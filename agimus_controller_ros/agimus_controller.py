import rospy

from agimus_controller_ros.controller_base import ControllerBase

from agimus_controller_ros.hpp_subscriber import HPPSubscriber


class AgimusControllerNode(ControllerBase):
    def __init__(self) -> None:
        super().__init__()
        self.hpp_subscriber = HPPSubscriber()

    def get_next_trajectory_point(self):
        return self.hpp_subscriber.get_trajectory_point()
