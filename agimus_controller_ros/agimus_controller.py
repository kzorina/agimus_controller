import rospy

from agimus_controller_ros.controller_base import ControllerBase


class AgimusControllerNodeParameters:
    def __init__(self) -> None:
        self.rate = rospy.get_param("~rate", 100)
        self.horizon_size = rospy.get_param("~horizon_size", 100)


class AgimusControllerNode(ControllerBase):
    def __init__(self) -> None:
        super().__init__()
