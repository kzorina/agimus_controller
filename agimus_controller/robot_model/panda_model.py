from tempfile import TemporaryDirectory
from pathlib import Path
import rospkg
import xacro
from agimus_controller.robot_model.robot_model import RobotModelParameters
from agimus_controller.robot_model.robot_model import RobotModel


class PandaRobotModelParameters(RobotModelParameters):
    def __init__(self) -> None:
        super().__init__()
        self._temp_dir = TemporaryDirectory()
        self._rospack = rospkg.RosPack()
        self._package_dir = Path(self._rospack.get_path("franka_description"))
        self._xacro_file = self._package_dir / "robots" / "panda" / "panda.urdf.xacro"

        self.locked_joint_names = ["panda_finger_joint1", "panda_finger_joint2"]
        self.urdf = Path(self._temp_dir.name) / "panda.urdf"
        self.srdf = self._package_dir / "robots" / "panda" / "panda.srdf"
        self.ee_frame_name = "panda_leftfinger"

        # Parse the xacro file and dump it in a temporary folder.
        self._urdf_string = xacro.process_file(
            self._xacro_file, mappings={}
        ).toprettyxml(indent="  ")
        with open(str(self.urdf), "w") as urdf_file:
            urdf_file.write(self._urdf_string)

    def __del__(self):
        self._temp_dir.cleanup()


class PandaRobotModel(RobotModel):
    @classmethod
    def load_model(cls, env=None, params=None):
        if params is not None:
            return super().load_model(params, env)
        else:
            return super().load_model(PandaRobotModelParameters(), env)
