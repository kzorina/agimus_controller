from pathlib import Path
import rospkg
import xacro
from agimus_controller.robot_model.robot_model import RobotModelParameters
from agimus_controller.robot_model.robot_model import RobotModel


class PandaRobotModelParameters(RobotModelParameters):
    def __init__(self) -> None:
        super().__init__()
        self._rospack = rospkg.RosPack()
        self._package_dir = Path(self._rospack.get_path("franka_description"))
        self._xacro_file = self._package_dir / "robots" / "panda" / "panda.urdf.xacro"

        self.model_name = "panda"
        self.q0_name = "default"
        self.free_flyer = False
        self.collision_as_capsule = False
        self.self_collision = False
        self.locked_joint_names = ["panda_finger_joint1", "panda_finger_joint2"]
        self.urdf = xacro.process_file(self._xacro_file, mappings={}).toxml()
        self.srdf = self._package_dir / "robots" / "panda" / "panda.srdf"
        self.meshes = self._package_dir / "meshes"


class PandaRobotModel(RobotModel):
    @classmethod
    def load_model(cls, env=None, params=None):
        if params is not None:
            return super().load_model(params, env)
        else:
            return super().load_model(PandaRobotModelParameters(), env)
