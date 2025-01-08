from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import coal
import numpy as np
import numpy.typing as npt
import pinocchio as pin


@dataclass
class RobotModelParameters:
    q0: npt.NDArray[np.float64]  # Initial configuration of the robot
    free_flyer = False  # True if the robot has a free flyer
    locked_joint_names = []  # List of joint names to lock
    urdf_path = Path() | str  # Path to the URDF file
    srdf_path = Path() | str  # Path to the SRDF file
    urdf_meshes_dir = (
        Path() | str
    )  # Path to the directory containing the meshes and the URDF file.
    collision_as_capsule = (
        False  # True if the collision model should be reduced to capsules.
    )
    # By default, the collision model when convexified is a sum of spheres and cylinders, often representing capsules. Here, all the couples sphere cylinder sphere are replaced by  HPPFCL capsules.
    self_collision = False  # If True, the collision model takes into account collisions pairs written in the srdf file.


class RobotModelFactory:
    """Parse the robot model, reduce it and filter the collision model."""

    def __init__(self, param: RobotModelParameters):
        """Parse the robot model, reduce it and filter the collision model.

        Args:
            param (RobotModelParameters): Parameters to load the robot models.
        """
        self._params = param
        self.load_models()

    @property
    def full_robot_model(self) -> pin.Model:
        return self.full_robot_model

    @property
    def robot_model(self) -> pin.Model:
        return self.robot_model

    @property
    def visual_model(self) -> pin.GeometryModel:
        return self.visual_model

    @property
    def collision_model(self) -> pin.GeometryModel:
        return self.collision_model

    @property
    def q0(self) -> np.array:
        return self.q0

    def load_models(self) -> None:
        self._load_full_pinocchio_models()
        if self._params.locked_joint_names is not None:
            self._load_reduced_model()
        else:
            self.robot_model = deepcopy(self.full_robot_model)
        if self._params.collision_as_capsule:
            self._update_collision_model_to_capsules()
        if self._params.self_collision:
            self._update_collision_model_to_self_collision()

    def _get_joints_to_lock(self) -> list[int]:
        """Get the joints ID to lock.

        Raises:
            ValueError: Joint name not found in the robot model.

        Returns:
            list[int]: List of joint IDs to lock.
        """
        joints_to_lock = []
        for jn in self._params.locked_joint_names:
            if self.full_robot_model.existJointName(jn):
                joints_to_lock.append(self.full_robot_model.getJointId(jn))
            else:
                raise ValueError(f"Joint {jn} not found in the robot model.")
        return joints_to_lock

    def _load_full_pinocchio_models(self) -> None:
        """Load the full robot model, the visual model and the collision model."""
        (
            self.full_robot_model,
            self.collision_model,
            self.visual_model,
        ) = pin.buildModelsFromUrdf(
            self._params.urdf_path,
            self._params.urdf_meshes_dir,
            self._params.free_flyer,
        )

    def _load_reduced_model(self) -> None:
        """Load the reduced robot model."""
        joints_to_lock = self._get_joints_to_lock()
        self.robot_model = pin.buildReducedModel(
            self.full_robot_model, joints_to_lock, self.q0
        )

    def _update_collision_model_to_capsules(self) -> None:
        """Update the collision model to capsules."""
        cmodel = self.collision_model.copy()
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
                self.collision_model.addGeometryObject(capsule)
                self.collision_model.removeGeometryObject(geom_object.name)

            # Remove spheres associated with links
            elif isinstance(geometry, coal.Sphere) and "link" in geom_object.name:
                self.collision_model.removeGeometryObject(geom_object.name)

    def _update_collision_model_to_self_collision(self) -> None:
        """Update the collision model to self collision."""
        pin.addAllCollisionPairs(self.collision_model)
        pin.removeCollisionPairs(
            self.robot_model, self.collision_model, self._params.srdf_path
        )

    def _generate_capsule_name(self, base_name: str, existing_names: list) -> str:
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
