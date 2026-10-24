from mujoco_generator.controllers.pick_place_controller import PickPlaceController


class PickPlaceControllerV005(PickPlaceController):
    """Experiment 005: remove HOLD before TRANSFER."""

    T_HOLD = 11.0
    T_TRANSFER = 15.0
    T_TRANSFER_HOLD = 17.0
    T_PLACE = 20.0
    T_RELEASE = 22.0
    T_SETTLE = 23.0
    T_RETREAT = 26.0
    T_END = 27.0
