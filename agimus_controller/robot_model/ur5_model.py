from math import pi
import numpy as np
import example_robot_data
from agimus_controller.robot_model.robot_model import RobotModelParameters
from agimus_controller.robot_model.robot_model import RobotModel


class UR5RobotModelParameters(RobotModelParameters):
    def __init__(self) -> None:
        super().__init__()
        self.model_name = "ur5"
        self.q0_name = "default"
        self.free_flyer = False
        self.locked_joint_names = []
        self.urdf = (
            "package://example-robot-data/robots/ur_description/urdf/ur5_gripper.urdf"
        )
        self.srdf = (
            "package://example-robot-data/robots/ur_description/srdf/ur5_gripper.srdf"
        )
        self.meshes = "package://example-robot-data/robots/ur_description/meshes"


class UR5RobotModel(RobotModel):
    @classmethod
    def load_model(cls, env=None):
        params = UR5RobotModelParameters()
        obj = cls()
        model_wrapper = example_robot_data.load(params.model_name)
        obj._model = model_wrapper.model
        obj._cmodel = model_wrapper.model
        obj._vmodel = model_wrapper.model
        obj._rmodel = model_wrapper.model
        obj._rcmodel = model_wrapper.model
        obj._rvmodel = model_wrapper.model
        obj._q0 = np.array(
            [pi / 6, -pi / 2, pi / 2, 0, 0, 0, -0.2, 0, 0.02, 0, 0, 0, 1]
        )
        obj._update_collision_model(
            env, params.collision_as_capsule, params.self_collision, params.srdf
        )
        return obj
