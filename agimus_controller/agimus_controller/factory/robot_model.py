from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import coal
import numpy as np
import numpy.typing as npt
import pinocchio as pin


@dataclass
class RobotModelParameters:
    q0: npt.NDArray[np.float64] = np.array(
        [], dtype=np.float64
    )  # Initial full configuration of the robot
    free_flyer: bool = False  # True if the robot has a free flyer
    moving_joint_names: list[str] = field(default_factory=list)
    urdf_path: Path = Path()  # Path to the URDF file
    urdf_xml: str | None = (
        None  # String containing URDF. If None, then ``urdf_path`` used
    )
    srdf_path: Path | None = None  # Path to the SRDF file
    urdf_meshes_dir: Path | None = (
        Path()  # Path to the directory containing the meshes and the URDF file.
    )
    collision_as_capsule: bool = (
        False  # True if the collision model should be reduced to capsules.
    )
    # By default, the collision model when convexified is a sum of spheres and cylinders, often representing capsules. Here, all the couples sphere cylinder sphere are replaced by coal capsules.
    self_collision: bool = (
        False  # If True, the collision model takes into account collisions pairs written in the srdf file.
    )
    armature: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )  # Default empty NumPy array
    collision_color: npt.NDArray[np.float64] = (
        np.array([249.0, 136.0, 126.0, 125.0]) / 255.0
    )  # Red color for the collision model

    def __post_init__(self):
        # Check full_q0 is not empty
        if len(self.q0) == 0:
            raise ValueError("q0 cannot be empty.")

        # Handle armature:
        if self.armature.size == 0:
            # Use a default armature filled with 0s, based on the size of moving_joint_names
            self.armature = np.zeros(len(self.moving_joint_names), dtype=np.float64)

        # Ensure armature has the same shape as moving_joint_names
        if (
            len(self.armature) != len(self.moving_joint_names) and not self.free_flyer
        ):  #! TODO: Do the same for free flyer
            raise ValueError(
                f"Armature must have the same shape as moving_joint_names. Got {self.armature.shape} and {len(self.moving_joint_names)}."
            )

        # Ensure URDF and SRDF are valid
        if self.urdf_xml is None:
            if self.urdf_path is not None:
                if self.urdf_path.is_file():
                    self.urdf_path = self.urdf_path.absolute().as_posix()
                else:
                    raise ValueError(
                        "urdf_path must be a valid file path. "
                        f"File: '{self.urdf_path}' doesn't exist!"
                    )
        elif self.urdf_xml == "":
            raise ValueError("urdf_xml can not be an empty string.")

        if self.srdf_path is not None:
            if self.srdf_path.is_file():
                self.srdf_path = self.srdf_path.absolute().as_posix()
            else:
                raise ValueError(
                    "srdf_path must be a valid file path. "
                    f"File: '{self.srdf_path}' doesn't exist!"
                )

        if self.urdf_meshes_dir is not None:
            if self.urdf_meshes_dir.exists():
                self.urdf_meshes_dir = self.urdf_meshes_dir.absolute().as_posix()
            else:
                raise ValueError(
                    "urdf_meshes_dir must be a valid folder path. "
                    f"Folder: '{self.urdf_meshes_dir}' doesn't exist!"
                )


class RobotModels:
    """Parse the robot model, reduce it and filter the collision model."""

    def __init__(self, param: RobotModelParameters):
        """Parse the robot model, reduce it and filter the collision model.

        Args:
            param (RobotModelParameters): Parameters to load the robot models.
        """
        self._params = param
        self._full_robot_model = None
        self._robot_model = None
        self._collision_model = None
        self._visual_model = None
        self._q0 = deepcopy(self._params.q0)
        self.load_models()  # Populate models

    @property
    def full_robot_model(self) -> pin.Model:
        """Full robot model."""
        if self._full_robot_model is None:
            raise AttributeError("Full robot model has not been initialized yet.")
        return self._full_robot_model

    @property
    def robot_model(self) -> pin.Model:
        """Robot model, reduced if specified in the parameters."""
        if self._robot_model is None:
            raise AttributeError("Robot model has not been computed yet.")
        return self._robot_model

    @property
    def visual_model(self) -> pin.GeometryModel:
        """Visual model of the robot."""
        if self._visual_model is None:
            raise AttributeError("Visual model has not been computed yet.")
        return self._visual_model

    @property
    def collision_model(self) -> pin.GeometryModel:
        """Collision model of the robot."""
        if self._collision_model is None:
            raise AttributeError("Visual model has not been computed yet.")
        return self._collision_model

    def load_models(self) -> None:
        """Load and prepare robot models based on parameters."""
        self._load_full_pinocchio_models()
        self._lock_joints()
        if self._params.collision_as_capsule:
            self._update_collision_model_to_capsules()
        if self._params.self_collision:
            self._update_collision_model_to_self_collision()

    def _load_full_pinocchio_models(self) -> None:
        """Load the full robot model, the visual model and the collision model."""
        try:
            geometry_types = [
                pin.GeometryType.COLLISION,
                (pin.GeometryType.VISUAL if self._params.urdf_meshes_dir else None),
            ]

            urdf_xml = self._params.urdf_xml
            if urdf_xml is None:
                with open(self._params.urdf_path, "r") as file:
                    urdf_xml = file.read().replace("\n", "")

            if self._params.free_flyer:
                self._full_robot_model = pin.buildModelFromXML(
                    urdf_xml, pin.JointModelFreeFlyer()
                )
            else:
                self._full_robot_model = pin.buildModelFromXML(urdf_xml)

            self._collision_model, self._visual_model = [
                (
                    pin.buildGeomFromUrdfString(
                        self._full_robot_model,
                        urdf_xml,
                        geometry_type,
                        package_dirs=[self._params.urdf_meshes_dir],
                    )
                    if geometry_type is not None
                    else None
                )
                for geometry_type in geometry_types
            ]
        except Exception as e:
            raise ValueError(
                f"Failed to load URDF models from {self._params.urdf_path}: {e}"
            )

    def _lock_joints(self) -> None:
        """Apply locked joints."""
        joints_to_lock = []
        for jn in self._full_robot_model.names:
            if jn == 'universe':
                continue
            if not jn in self._params.moving_joint_names:
                joints_to_lock.append(self._full_robot_model.getJointId(jn))

        self._robot_model = pin.buildReducedModel(
            self._full_robot_model, joints_to_lock, self._q0
        )

    def _update_collision_model_to_capsules(self) -> None:
        """Update the collision model to capsules."""
        cmodel = self._collision_model.copy()
        list_names_capsules = []
        # Iterate through geometry objects in the collision model
        for geom_object in cmodel.geometryObjects:
            geometry = geom_object.geometry
            # Remove superfluous suffix from the name
            base_name = "_".join(geom_object.name.split("_")[:-1])
            # Convert cylinders to capsules
            if isinstance(geometry, coal.Cylinder):
                name = self._generate_capsule_name(base_name, list_names_capsules)
                list_names_capsules.append(name)
                capsule = pin.GeometryObject(
                    name=name,
                    parent_frame=geom_object.parentFrame,
                    parent_joint=geom_object.parentJoint,
                    collision_geometry=coal.Capsule(
                        geometry.radius, geometry.halfLength
                    ),
                    placement=geom_object.placement,
                )
                capsule.meshColor = self._params.collision_color
                self._collision_model.addGeometryObject(capsule)
                self._collision_model.removeGeometryObject(geom_object.name)

            # Remove spheres associated with links
            elif isinstance(geometry, coal.Sphere) and "link" in geom_object.name:
                self._collision_model.removeGeometryObject(geom_object.name)

    def _update_collision_model_to_self_collision(self) -> None:
        """Update the collision model to self collision."""
        self._collision_model.addAllCollisionPairs()
        pin.removeCollisionPairs(
            self._robot_model, self._collision_model, str(self._params.srdf_path)
        )

    def _generate_capsule_name(self, base_name: str, existing_names: list[str]) -> str:
        """Generates a unique capsule name for a geometry object.
        Args:
            base_name (str): The base name of the geometry object.
            existing_names (list): List of names already assigned to capsules.
        Returns:
            str: Unique capsule name.
        """
        i = 0
        while f"{base_name}_capsule_{i}" in existing_names:
            i += 1
        return f"{base_name}_capsule_{i}"

    @property
    def armature(self) -> npt.NDArray[np.float64]:
        """Armature of the robot.
        Returns:
            npt.NDArray[np.float64]: Armature of the robot.
        """
        return self._params.armature
