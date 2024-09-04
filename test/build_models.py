import yaml
import pinocchio as pin
import numpy as np
import rospy
import example_robot_data
from panda_torque_mpc_pywrap import reduce_capsules_robot


from pathlib import Path
from hppfcl import Sphere, Box, Cylinder, Capsule
# from agimus_controller.utils.path_finder import get_project_root


class ObstacleParamsParser:
    def __init__(self, yaml_file, collision_model):
        with open(yaml_file, "r") as file:
            self.params = yaml.safe_load(file)
        self.collision_model = collision_model

    def add_collisions(self):
        obs_idx = 1

        while f"obstacle{obs_idx}" in self.params:
            obstacle_name = f"obstacle{obs_idx}"
            obstacle_config = self.params[obstacle_name]

            type_ = obstacle_config.get("type")
            translation_vect = obstacle_config.get("translation", [])

            if not translation_vect:
                print(
                    f"No obstacle translation declared for the obstacle named: {obstacle_name}"
                )
                return

            translation = np.array(translation_vect).reshape(3)

            rotation_vect = obstacle_config.get("rotation", [])
            if not rotation_vect:
                print(
                    f"No obstacle rotation declared for the obstacle named: {obstacle_name}"
                )
                return

            rotation = np.array(rotation_vect).reshape(4)

            geometry = None
            if type_ == "sphere":
                radius = obstacle_config.get("radius")
                if radius:
                    geometry = Sphere(radius)
                else:
                    print("No dimension or wrong dimensions in the obstacle config.")
                    return
            elif type_ == "box":
                x = obstacle_config.get("x")
                y = obstacle_config.get("y")
                z = obstacle_config.get("z")
                if x and y and z:
                    geometry = Box(x, y, z)
                else:
                    print("No dimension or wrong dimensions in the obstacle config.")
                    return
            elif type_ == "cylinder":
                radius = obstacle_config.get("radius")
                half_length = obstacle_config.get("halfLength")
                if radius and half_length:
                    geometry = Cylinder(radius, half_length)
                else:
                    print("No dimension or wrong dimensions in the obstacle config.")
                    return
            elif type_ == "capsule":
                radius = obstacle_config.get("radius")
                half_length = obstacle_config.get("halfLength")
                if radius and half_length:
                    geometry = Capsule(radius, half_length)
                else:
                    print("No dimension or wrong dimensions in the obstacle config.")
                    return
            else:
                print("No type or wrong type in the obstacle config.")
                return
            obstacle_pose = pin.XYZQUATToSE3(np.concatenate([translation, rotation]))
            obstacle_pose.translation = translation
            obstacle = pin.GeometryObject(obstacle_name, 0, 0, geometry, obstacle_pose)
            self.collision_model.addGeometryObject(obstacle)
            obs_idx += 1

        collision_pairs = self.params.get("collision_pairs", [])
        if collision_pairs:
            for pair in collision_pairs:
                if len(pair) == 2:
                    name_object1, name_object2 = pair
                    if self.collision_model.existGeometryName(
                        name_object1
                    ) and self.collision_model.existGeometryName(name_object2):
                        self.add_collision_pair(name_object1, name_object2)
                    else:
                        print(
                            f"Object {name_object1} or {name_object2} does not exist in the collision model."
                        )
                else:
                    print(f"Invalid collision pair: {pair}")
        else:
            print("No collision pairs.")

    def add_collision_pair(self, name_object1, name_object2):
        object1_id = self.collision_model.getGeometryId(name_object1)
        object2_id = self.collision_model.getGeometryId(name_object2)
        if object1_id is not None and object2_id is not None:
            self.collision_model.addCollisionPair(
                pin.CollisionPair(object1_id, object2_id)
            )
        else:
            print(
                f"Object ID not found for collision pair: {object1_id} and {object2_id}"
            )


class RobotModelConstructor:
    def __init__(self, load_from_ros: bool=False) -> None:
        # self.load_model(load_from_ros)
        pass

    def load_model(self, load_from_ros):
        # yaml_path = get_project_root() / "config" / "param.yaml"
        yaml_path = Path()
        # mesh_dir = str(Path(franka_description.__path__[0]) / "meshes")

        if load_from_ros:
            print("Load robot from ROS")

            # Getting urdf and srdf content
            urdf_string = rospy.get_param("robot_description")
            # srdf_string = rospy.get_param("robot_description_semantic")
            # self.construct_robot_model(urdf_string, srdf_string, mesh_dir)
            self.construct_collision_model(self.rmodel, urdf_string, yaml_path)
        else:
            print("Load robot from files")

            urdf_path = str(Path("urdf") / "robot.urdf")
            srdf_path = str(Path("srdf") / "demo.srdf")
            robot = example_robot_data.load("panda")
            self.set_robot_model(robot, urdf_path, srdf_path)
            self.set_collision_model(urdf_path, yaml_path)

    def set_robot_model(self, robot, urdf_path, srdf_path):
        self._model = pin.Model()
        pin.buildModelFromUrdf(urdf_path, self._model)
        pin.loadReferenceConfigurations(self._model, srdf_path, False)

        q0 = self._model.referenceConfigurations["default"]
        locked_joints = [
            robot.model.getJointId("panda_finger_joint1"),
            robot.model.getJointId("panda_finger_joint2"),
        ]
        self._rmodel = pin.buildReducedModel(self._model, locked_joints, q0)

    def set_collision_model(self, urdf_path, yaml_path):
        self._cmodel = pin.buildGeomFromUrdf(self._rmodel, urdf_path, pin.COLLISION)
        self._crmodel = reduce_capsules_robot(self._cmodel)
        parser = ObstacleParamsParser(yaml_path, self._crmodel)
        parser.add_collisions()
        self._crmodel = parser.collision_model

    def construct_robot_model(self, urdf_string, srdf_string):
        self._model = pin.buildModelFromXML(urdf_string)

        locked_joints = [
            self._model.getJointId("panda_finger_joint1"),
            self._model.getJointId("panda_finger_joint2"),
        ]
        pin.loadReferenceConfigurationsFromXML(self._model, srdf_string, False)
        self._cmodel = pin.buildGeomFromUrdfString(
            self._model, urdf_string, pin.COLLISION
        )
        q0 = self._model.referenceConfigurations["default"]
        self._rmodel, geometric_models_reduced = pin.buildReducedModel(
            self._model,
            list_of_geom_models=[self._cmodel],
            list_of_joints_to_lock=locked_joints,
            reference_configuration=q0,
        )
        self._crmodel, self._vrmodel = geometric_models_reduced

    def construct_collision_model(self, yaml_file):
        self._crmodel = self.transform_model_into_capsules(self._crmodel)
        parser = ObstacleParamsParser(yaml_file, self._crmodel)
        parser.add_collisions()
        return parser.collision_model

    def transform_model_into_capsules(self, model):
        """Modifying the collision model to transform the spheres/cylinders into capsules which makes it easier to have a fully constrained robot."""
        model_copy = model.copy()
        list_names_capsules = []

        # Going through all the goemetry objects in the collision model
        for geom_object in model_copy.geometryObjects:
            if isinstance(geom_object.geometry, Cylinder):
                # Sometimes for one joint there are two cylinders, which need to be defined by two capsules for the same link.
                # Hence the name convention here.
                if (geom_object.name[:-4] + "capsule_0") in list_names_capsules:
                    name = geom_object.name[:-4] + "capsule_" + "1"
                else:
                    name = geom_object.name[:-4] + "capsule_" + "0"
                list_names_capsules.append(name)
                placement = geom_object.placement
                parentJoint = geom_object.parentJoint
                parentFrame = geom_object.parentFrame
                geometry = geom_object.geometry
                geom = pin.GeometryObject(
                    name,
                    parentFrame,
                    parentJoint,
                    Capsule(geometry.radius, geometry.halfLength),
                    placement,
                )
                RED = np.array([249, 136, 126, 125]) / 255
                geom.meshColor = RED
                model_copy.addGeometryObject(geom)
                model_copy.removeGeometryObject(geom_object.name)
            elif (
                isinstance(geom_object.geometry, Sphere) and "link" in geom_object.name
            ):
                model_copy.removeGeometryObject(geom_object.name)
        return model_copy

    def get_robot_model(self):
        return self._model

    def get_robot_reduced_model(self):
        return self._rmodel

    def get_collision_reduced_model(self):
        return self._crmodel

    def get_visual_reduced_model(self):
        return self._vrmodel
