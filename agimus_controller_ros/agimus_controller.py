import time
from agimus_controller_ros.controller_base import ControllerBase

from agimus_controller_ros.hpp_subscriber import HPPSubscriber


class AgimusControllerNode(ControllerBase):
    def __init__(self) -> None:
        super().__init__()
        self.hpp_subscriber = HPPSubscriber()

    def get_next_trajectory_point(self):
        return self.hpp_subscriber.get_trajectory_point()

    def update_state_machine(self):
        if self.hpp_subscriber.last_time_got_traj_point is None:
            return
        self.last_elapsed_time = self.elapsed_time
        self.elapsed_time = time.time() - self.hpp_subscriber.last_time_got_traj_point
        if self.last_elapsed_time is None:
            if self.elapsed_time is not None:
                self.change_state()
            return
        if (self.elapsed_time < 0.05 and self.last_elapsed_time >= 0.05) or (
            self.elapsed_time >= 0.05 and self.last_elapsed_time < 0.05
        ):
            self.change_state()

    def change_state(self):
        self.state_machine_timeline_idx = (self.state_machine_timeline_idx + 1) % len(
            self.state_machine_timeline
        )
        self.state_machine = self.state_machine_timeline[
            self.state_machine_timeline_idx
        ]
        print("changing state to ", self.state_machine)
