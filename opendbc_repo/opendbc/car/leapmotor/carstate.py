from opendbc.car import DT_CTRL, Bus, structs
from opendbc.can.parser import CANParser
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.common.filter_simple import FirstOrderFilter
from opendbc.car.interfaces import CarStateBase
from opendbc.car.leapmotor.values import DBC, CarControllerParams

GearShifter = structs.CarState.GearShifter
GEARS = {0: GearShifter.park, 1: GearShifter.reverse, 2: GearShifter.neutral, 3: GearShifter.drive}


class SetSpeed:
  """openpilot's own ACC set speed (km/h) with openpilot longitudinal. The stock set speed stops at 30 km/h, but the
  knob (CLUSTER.ACC_SPEED_KNOB) keeps counting its detents, 1 km/h each, so openpilot follows the knob instead."""
  STOCK_MIN = 30.  # km/h, the lowest stock set speed
  MIN = 5.
  MAX = 150.
  RESYNC_FRAMES = 50  # the stock set speed follows a detent 20-30 ms later; any other change is the car's own

  MAX_DETENTS = 5  # per 10 ms frame; more is a jump of the reading (e.g. its first frame), not the knob

  def __init__(self):
    self.knob_last: int | None = None
    self.stock_last: float | None = None
    self.speed: float | None = None
    self.engaged_last = False
    self.frames_since_knob = self.RESYNC_FRAMES

  def update(self, engaged: bool, stock: float, knob: int) -> float:
    detents = 0 if self.knob_last is None else round((((knob - self.knob_last + 1024) % 2048) - 1024) / 100)
    detents = detents if abs(detents) <= self.MAX_DETENTS else 0
    self.knob_last = knob
    self.frames_since_knob = 0 if detents else self.frames_since_knob + 1

    if not engaged or not self.engaged_last or self.speed is None:
      # starts from the stock set speed at every engagement
      self.speed = stock
    elif detents:
      self.speed = min(max(self.speed + detents, self.MIN), self.MAX)
    elif stock != self.stock_last and self.frames_since_knob > self.RESYNC_FRAMES and stock > self.STOCK_MIN:
      # changed without the knob: follow the car (not at its 30 km/h floor, which says nothing about the knob)
      self.speed = stock
    self.stock_last = stock
    self.engaged_last = engaged
    return self.speed


class CarState(CarStateBase):
  def __init__(self, CP):
    super().__init__(CP)
    self.steering_angle_last = 0.0
    self.adas_lka: dict | None = None
    self.lka_received_seen = False
    self.driver_override_frames = 0
    self.driver_override = False
    # the EPS's own effort as it shows up in the driver-torque reading, which lags it (see EPS_TORQUE_LAG)
    self.eps_torque_lagged = FirstOrderFilter(0., CarControllerParams.EPS_TORQUE_LAG, DT_CTRL)
    self.powertrain_state: dict | None = None
    self.set_speed = SetSpeed()
    self.stopped_frames = 0

  def update(self, can_parsers) -> structs.CarState:
    cp = can_parsers[Bus.pt]
    cp_cam = can_parsers[Bus.cam]
    ret = structs.CarState()

    # speed
    self.parse_wheel_speeds(ret,
      cp.vl['WHEEL_SPEEDS']['WHEEL_SPEED_FL'],
      cp.vl['WHEEL_SPEEDS']['WHEEL_SPEED_FR'],
      cp.vl['WHEEL_SPEEDS']['WHEEL_SPEED_RL'],
      cp.vl['WHEEL_SPEEDS']['WHEEL_SPEED_RR'],
    )
    ret.vEgoCluster = cp.vl['CLUSTER']['DISPLAY_SPEED'] * CV.KPH_TO_MS
    ret.standstill = ret.vEgoRaw < 0.01
    ret.yawRate = cp.vl['IMU']['YAW_RATE'] * CV.DEG_TO_RAD

    # pedals
    ret.gasPressed = cp.vl['VEHICLE_STATE']['ACCEL_PEDAL'] > 0
    ret.brakePressed = bool(cp.vl['VEHICLE_STATE']['BRAKE_PRESSED'])

    # steering: the EPS speed field has no sign, so take the sign from the angle change
    angle = cp.vl['EPS_STEERING']['STEER_ANGLE']
    sign = 1.0 if angle > self.steering_angle_last else (-1.0 if angle < self.steering_angle_last else 0.0)
    ret.steeringAngleDeg = angle
    ret.steeringRateDeg = sign * cp.vl['EPS_STEERING']['STEER_SPEED_ABS']
    self.steering_angle_last = angle
    ret.steeringTorqueEps = cp.vl['EPS_STEERING']['STEER_TORQUE_EPS']
    # the driver's torque: the reading minus the share that follows the EPS's own effort
    ret.steeringTorque = cp.vl['EPS_STEERING']['STEER_TORQUE_DRIVER'] + \
                         self.eps_torque_lagged.update(ret.steeringTorqueEps) / CarControllerParams.EPS_TORQUE_PER_DRIVER_TORQUE
    # the torque reading is coarse: during an LCC override it only reached 2-3, but the EPS flagged the override itself.
    # The EPS also sets that flag whenever the car is above about 121 km/h, with or without the driver (99% of frames
    # above 121 km/h on a stock drive, and openpilot's own drive at 125 km/h), so above 118 km/h only the torque counts.
    eps_override = cp.vl['EPS_STEERING']['DRIVER_OVERRIDE'] and ret.vEgoRaw < 118 * CV.KPH_TO_MS
    self.driver_override_frames = self.driver_override_frames + 1 if eps_override else 0
    torque_pressed = self.update_steering_pressed(abs(ret.steeringTorque) > CarControllerParams.STEER_DRIVER_ALLOWANCE, 5)
    ret.steeringPressed = torque_pressed or self.driver_override_frames >= 2
    # When to let go of the wheel: on the EPS's own flag, like stock LCC, or on the driver's torque. Not on the raw
    # reading, which read 5-7 whenever the EPS pushed 65-95 in a curve: letting go on it dropped the EPS to 0 and back
    # every 0.7 s (41 times in 14 min). The flag alone is not enough: above 118 km/h it is always set, and at walking
    # pace it stayed off while openpilot pulled the wheel from +324 to -203 deg through the driver's hands.
    self.driver_override = self.driver_override_frames >= 2 or torque_pressed
    # The EPS reports whether it receives ADAS_LKA (0x83): losing it after it was received means our steering frame is
    # missing or rejected (camera-unplug test). Not set before the camera's first frame after ignition.
    lka_received = bool(cp.vl['EPS_STEERING']['ADAS_LKA_RECEIVED'])
    self.lka_received_seen = self.lka_received_seen or lka_received
    ret.steerFaultTemporary = self.lka_received_seen and not lka_received

    # gear
    ret.gearShifter = GEARS.get(int(cp.vl['VEHICLE_STATE']['GEAR']), GearShifter.unknown)

    # body
    ret.leftBlinker, ret.rightBlinker = self.update_blinker_from_lamp(50, cp.vl['BODY_CONTROL']['BLINKER_LEFT'] == 1,
                                                                      cp.vl['BODY_CONTROL']['BLINKER_RIGHT'] == 1)
    ret.doorOpen = bool(cp.vl['BODY_CONTROL']['DRIVER_DOOR_OPEN'])
    ret.seatbeltUnlatched = bool(cp.vl['SEATBELT']['DRIVER_SEATBELT_UNLATCHED'])

    # blind-spot radars: silent when stopped, so they are not timeout-checked (see get_can_parsers)
    ret.leftBlindspot = bool(cp.vl['BSD_LEFT']['OBJECT_DETECTED'])
    ret.rightBlindspot = bool(cp.vl['BSD_RIGHT']['OBJECT_DETECTED'])

    # stock ACC / LCC, read from the camera side
    ret.cruiseState.enabled = bool(cp.vl['POWERTRAIN_STATE']['ASSIST_ENGAGED'])
    # latest VCU frame, copied by the controller to cancel ACC; None until received
    self.powertrain_state = dict(cp.vl['POWERTRAIN_STATE']) if cp.ts_nanos['POWERTRAIN_STATE']['COUNTER'] > 0 else None
    # ACC_STATE: 1 not ready (Park, brake, accelerator override), 2 ready, 3 engaged. Engaged also counts as
    # available so that an accelerator override (state 1 while engaged) does not raise wrongCarMode
    ret.cruiseState.available = ret.cruiseState.enabled or int(cp_cam.vl['ADAS_LKA']['ACC_STATE']) in (2, 3)
    ret.cruiseState.speed = cp_cam.vl['ADAS_LKA']['ACC_SET_SPEED'] * CV.KPH_TO_MS
    if self.CP.openpilotLongitudinalControl:
      # openpilot's own set speed, which the knob can take below the stock minimum of 30 km/h
      ret.cruiseState.speed = self.set_speed.update(ret.cruiseState.enabled, cp_cam.vl['ADAS_LKA']['ACC_SET_SPEED'],
                                                    int(cp.vl['CLUSTER']['ACC_SPEED_KNOB'])) * CV.KPH_TO_MS
    ret.cruiseState.speedCluster = ret.cruiseState.speed
    # the standstill hold, after which openpilot waits for the driver (see CarControllerParams.STANDSTILL_RESUME_TIME)
    self.stopped_frames = self.stopped_frames + 1 if ret.standstill and not ret.gasPressed else 0
    ret.cruiseState.standstill = (ret.cruiseState.enabled and
                                  self.stopped_frames * DT_CTRL >= CarControllerParams.STANDSTILL_RESUME_TIME)
    # the camera's forward collision warning, shown by openpilot too
    ret.stockFcw = bool(cp_cam.vl['ROAD_SIGNS']['FCW'])
    # latest camera frame, rebuilt by the controller; None until the camera's 0x83 has been received
    self.adas_lka = dict(cp_cam.vl['ADAS_LKA']) if cp_cam.ts_nanos['ADAS_LKA']['COUNTER'] > 0 else None

    return ret

  @staticmethod
  def get_can_parsers(CP):
    return {
      Bus.pt: CANParser(DBC[CP.carFingerprint][Bus.pt], [("BSD_LEFT", float('nan')), ("BSD_RIGHT", float('nan'))], 0),
      Bus.cam: CANParser(DBC[CP.carFingerprint][Bus.pt], [], 2),
    }
