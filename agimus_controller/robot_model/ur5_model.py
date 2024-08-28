from math import pi
import numpy as np
from pathlib import Path
import rospkg
from agimus_controller.robot_model.robot_model import RobotModelParameters
from agimus_controller.robot_model.robot_model import RobotModel


class UR5RobotModelParameters(RobotModelParameters):
    def __init__(self) -> None:
        super().__init__()
        self._rospack = rospkg.RosPack()
        self._package_dir = Path(self._rospack.get_path("example-robot-data"))
        self.urdf = (
            self._package_dir
            / "robots"
            / "ur_description"
            / "urdf"
            / "ur5_gripper.urdf"
        )
        self.srdf = (
            self._package_dir
            / "robots"
            / "ur_description"
            / "srdf"
            / "ur5_gripper.srdf"
        )


class UR5RobotModel(RobotModel):
    @classmethod
    def load_model(cls, env=None):
        params = UR5RobotModelParameters()
        obj = super().load_model(params, env)
        obj._q0 = np.array([pi / 6, -pi / 2, pi / 2, -0.2, 0.02, 1])
        return obj
