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
    )  # Initial configuration of the robot
    free_flyer: bool = False  # True if the robot has a free flyer
    locked_joint_names: list[str] = field(default_factory=list)
    urdf_path: Path = Path()  # Path to the URDF file
    srdf_path: Path = Path()  # Path to the SRDF file
    urdf_meshes_dir: Path = (
        Path()
    )  # Path to the directory containing the meshes and the URDF file.
    collision_as_capsule: bool = (
        False  # True if the collision model should be reduced to capsules.
    )
    # By default, the collision model when convexified is a sum of spheres and cylinders, often representing capsules. Here, all the couples sphere cylinder sphere are replaced by hppfcl capsules.
    self_collision: bool = False  # If True, the collision model takes into account collisions pairs written in the srdf file.
    armature: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )  # Default empty NumPy array
    collision_color: npt.NDArray[np.float64] = (
        np.array([249.0, 136.0, 126.0, 125.0]) / 255.0
    )  # Red color for the collision model

    def __post_init__(self):
        # Check q0 is not empty
        if len(self.q0) == 0:
            raise ValueError("q0 cannot be empty.")

        # Handle armature:
        if self.armature.size == 0:
            # Use a default armature filled with 0s, based on the size of q0
            self.armature = np.zeros(len(self.q0), dtype=np.float64)

        # Ensure armature has the same shape as q0
        if self.armature.shape != self.q0.shape:
            raise ValueError(
                f"Armature must have the same shape as q0. Got {self.armature.shape} and {self.q0.shape}."
            )

        # Ensure paths are valid strings
        if not isinstance(self.urdf_path, str) or not self.urdf_path:
            raise ValueError("urdf_path must be a non-empty string.")

        if not isinstance(self.srdf_path, str):
            raise ValueError("srdf_path must be a string.")


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
            raise AttributeError("Robot model has not been initialized yet.")
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

    @property
    def q0(self) -> np.array:
        """Initial configuration of the robot."""
        return self._q0

    def load_models(self) -> None:
        """Load and prepare robot models based on parameters."""
        self._load_full_pinocchio_models()
        if self._params.locked_joint_names:
            self._apply_locked_joints()
        if self._params.collision_as_capsule:
            self._update_collision_model_to_capsules()
        if self._params.self_collision:
            self._update_collision_model_to_self_collision()

    def _load_full_pinocchio_models(self) -> None:
        """Load the full robot model, the visual model and the collision model."""
        try:
            (
                self._full_robot_model,
                self._collision_model,
                self._visual_model,
            ) = pin.buildModelsFromUrdf(
                self._params.urdf_path,
                self._params.urdf_meshes_dir,
                self._params.free_flyer,
            )
        except Exception as e:
            raise ValueError(
                f"Failed to load URDF models from {self._params.urdf_path}: {e}"
            )

    def _apply_locked_joints(self) -> None:
        """Apply locked joints."""
        joints_to_lock = []
        for jn in self._params.locked_joint_names:
            if self._full_robot_model.existJointName(jn):
                joints_to_lock.append(self._full_robot_model.getJointId(jn))
            else:
                raise ValueError(f"Joint {jn} not found in the robot model.")

        self._robot_model = pin.buildReducedModel(
            self._full_robot_model, joints_to_lock, self._q0
        )

    def _get_joints_to_lock(self) -> list[int]:
        """Get the joints ID to lock.

        Raises:
            ValueError: Joint name not found in the robot model.

        Returns:
            list[int]: List of joint IDs to lock.
        """
        joints_to_lock = []
        for jn in self._params.locked_joint_names:
            if self._full_robot_model.existJointName(jn):
                joints_to_lock.append(self._full_robot_model.getJointId(jn))
            else:
                raise ValueError(f"Joint {jn} not found in the robot model.")
        return joints_to_lock

    def _load_reduced_model(self) -> None:
        """Load the reduced robot model."""
        joints_to_lock = self._get_joints_to_lock()
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
                    parent_frame=int(geom_object.parentFrame),
                    parent_joint=int(geom_object.parentJoint),
                    collision_geometry=coal.Capsule(
                        geometry.radius, geometry.halfLength
                    ),
                    placement=geom_object.placement,
                )
                capsule.meshColor = np.array([249, 136, 126, 125]) / 255  # Red color
                self._collision_model.addGeometryObject(capsule)
                self._collision_model.removeGeometryObject(geom_object.name)

            # Remove spheres associated with links
            elif isinstance(geometry, coal.Sphere) and "link" in geom_object.name:
                self._collision_model.removeGeometryObject(geom_object.name)

    def _update_collision_model_to_self_collision(self) -> None:
        """Update the collision model to self collision."""
        pin.addAllCollisionPairs(self._collision_model)
        pin.removeCollisionPairs(
            self._robot_model, self._collision_model, self._params.srdf_path
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
