from copy import deepcopy
from os import environ
from os.path import dirname
import unittest
from pathlib import Path
import example_robot_data as robex
import numpy as np
import pinocchio as pin

from agimus_controller.factory.robot_model import RobotModelParameters, RobotModels


class TestRobotModelParameters(unittest.TestCase):
    def test_valid_initialization(self):
        """Test that the dataclass initializes correctly with valid input."""

        robot = robex.load("panda")
        urdf_path = Path(robot.urdf)
        srdf_path = Path(robot.urdf.replace("urdf", "srdf"))
        urdf_meshes_dir = urdf_path.parent.parent.parent.parent.parent
        free_flyer = False
        full_q0 = np.zeros(robot.model.nq)
        locked_joint_names = ["panda_joint1", "panda_joint2"]
        reduced_nq = robot.model.nq - len(locked_joint_names)
        q0 = np.zeros(reduced_nq)
        params = RobotModelParameters(
            full_q0=full_q0,
            q0=q0,
            free_flyer=free_flyer,
            locked_joint_names=locked_joint_names,
            urdf=urdf_path,
            srdf=srdf_path,
            urdf_meshes_dir=urdf_meshes_dir,
            collision_as_capsule=True,
            self_collision=True,
            armature=np.linspace(0.1, 0.9, reduced_nq),
        )
        self.assertTrue(np.array_equal(params.full_q0, full_q0))
        self.assertTrue(np.array_equal(params.q0, q0))
        self.assertEqual(params.free_flyer, free_flyer)
        self.assertEqual(params.locked_joint_names, ["panda_joint1", "panda_joint2"])
        self.assertEqual(params.urdf, urdf_path)
        self.assertEqual(params.srdf, srdf_path)
        self.assertEqual(params.urdf_meshes_dir, urdf_meshes_dir)
        self.assertTrue(params.collision_as_capsule)
        self.assertTrue(params.self_collision)
        self.assertTrue(
            np.array_equal(params.armature, np.linspace(0.1, 0.9, reduced_nq))
        )

    def test_empty_q0_raises_error(self):
        """Test that an empty q0 raises a ValueError."""
        with self.assertRaises(ValueError):
            RobotModelParameters(q0=np.array([]))

    def test_armature_default_value(self):
        """Test that the armature is set to default if not provided."""
        robot = robex.load("panda")
        urdf_path = Path(robot.urdf)
        srdf_path = Path(robot.urdf.replace("urdf", "srdf"))
        urdf_meshes_dir = urdf_path.parent.parent.parent.parent.parent
        locked_joint_names = ["panda_joint1", "panda_joint2"]
        free_flyer = False
        full_q0 = np.zeros(robot.model.nq)
        locked_joint_names = ["panda_joint1", "panda_joint2"]
        q0 = np.zeros(robot.model.nq - len(locked_joint_names))
        params = RobotModelParameters(
            full_q0=full_q0,
            q0=q0,
            free_flyer=free_flyer,
            locked_joint_names=locked_joint_names,
            urdf=urdf_path,
            srdf=srdf_path,
            urdf_meshes_dir=urdf_meshes_dir,
            collision_as_capsule=True,
            self_collision=True,
        )
        self.assertTrue(np.array_equal(params.armature, np.zeros_like(q0)))

    def test_armature_mismatched_shape_raises_error(self):
        """Test that a mismatched armature raises a ValueError."""
        full_q0 = np.array([0.0, 1.0, 2.0])
        armature = np.array([0.1, 0.2])  # Wrong shape
        with self.assertRaises(ValueError):
            RobotModelParameters(full_q0=full_q0, armature=armature)

    def test_invalid_urdf_raises_error(self):
        """Test that an invalid URDF path raises a ValueError."""
        full_q0 = np.array([0.0, 1.0, 2.0])
        with self.assertRaises(ValueError):
            RobotModelParameters(full_q0=full_q0, urdf=Path("invalid_path"))

    def test_invalid_urdf_string(self):
        """Test that an invalid URDF string raises a ValueError."""
        full_q0 = np.array([0.0, 1.0, 2.0])
        with self.assertRaises(ValueError):
            RobotModelParameters(full_q0=full_q0, urdf="")

    def test_invalid_srdf_type_raises_error(self):
        """Test that a non-string SRDF path raises a ValueError."""
        full_q0 = np.array([0.0, 1.0, 2.0])
        with self.assertRaises(ValueError):
            RobotModelParameters(full_q0=full_q0, srdf=Path("invalid_path"))

    def test_invalid_urdf_mesh_path_type_raises_error(self):
        """Test that a non-string SRDF path raises a ValueError."""
        full_q0 = np.array([0.0, 1.0, 2.0])
        with self.assertRaises(ValueError):
            RobotModelParameters(full_q0=full_q0, urdf_meshes_dir=Path("invalid_path"))


class TestRobotModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        This method sets up the shared environment for all test cases in the class.
        """
        # Load the example robot model using example robot data to get the URDF path.
        robot = robex.load("panda")
        urdf_path = Path(robot.urdf)
        srdf_path = Path(robot.urdf.replace("urdf", "srdf"))
        urdf_meshes_dir = urdf_path.parent.parent.parent.parent.parent
        free_flyer = False
        full_q0 = np.zeros(robot.model.nq)
        locked_joint_names = ["panda_joint1", "panda_joint2"]
        reduced_nq = robot.model.nq - len(locked_joint_names)
        q0 = np.zeros(reduced_nq)
        cls.params = RobotModelParameters(
            full_q0=full_q0,
            q0=q0,
            free_flyer=free_flyer,
            locked_joint_names=locked_joint_names,
            urdf=urdf_path,
            srdf=srdf_path,
            urdf_meshes_dir=urdf_meshes_dir,
            collision_as_capsule=True,
            self_collision=True,
            armature=np.linspace(0.1, 0.9, reduced_nq),
        )

    def setUp(self):
        """
        This method ensures that a fresh RobotModelParameters and RobotModels instance
        are created for each test case.
        """
        self.params = deepcopy(self.params)
        self.robot_models = RobotModels(self.params)

    def test_invalid_urdf_value(self):
        self.params.urdf = None
        with self.assertRaises(ValueError):
            RobotModels(self.params)

    def test_invalid_srdf_value(self):
        self.params.srdf = None
        with self.assertRaises(ValueError):
            RobotModels(self.params)

    def test_invalid_srdf_no_colision(self):
        self.params.srdf = None
        self.params.self_collision = False
        RobotModels(self.params)

    def test_no_meshes(self):
        self.params.urdf_meshes_dir = None
        # Clear paths where pinocchio searches for meshes
        environ["ROS_PACKAGE_PATH"] = ""
        environ["AMENT_PREFIX_PATH"] = ""
        with self.assertRaises(ValueError):
            RobotModels(self.params)

    def test_load_urdf_from_string(self):
        params = deepcopy(self.params)
        with open(Path(robex.load("panda").urdf), "r") as file:
            params.urdf = file.read().replace("\n", "")

        robot_models_str = RobotModels(params)
        self.assertEqual(
            robot_models_str.full_robot_model,
            self.robot_models.full_robot_model,
        )
        self.assertEqual(
            robot_models_str.robot_model,
            self.robot_models.robot_model,
        )
        self.assertTrue(np.array_equal(robot_models_str.q0, self.robot_models.q0))
        self.assertTrue(
            np.array_equal(robot_models_str.full_q0, self.robot_models.full_q0)
        )
        self.assertEqual(
            robot_models_str.collision_model.ngeoms,
            self.robot_models.collision_model.ngeoms,
        )
        self.assertEqual(
            robot_models_str.visual_model.ngeoms,
            self.robot_models.visual_model.ngeoms,
        )

    def test_initial_configuration(self):
        self.assertTrue(np.array_equal(self.robot_models.q0, self.params.q0))

    def test_load_models_populates_models(self):
        self.robot_models.load_models()
        self.assertIsNotNone(self.robot_models.full_robot_model)
        self.assertIsNotNone(self.robot_models.visual_model)
        self.assertIsNotNone(self.robot_models.collision_model)

    def test_reduced_robot_model(self):
        self.robot_models.load_models()
        self.assertTrue(
            self.robot_models.robot_model.nq
            == self.robot_models.full_robot_model.nq
            - len(self.params.locked_joint_names)
        )

    def test_invalid_joint_name_raises_value_error(self):
        # Modify a fresh instance of parameters for this test
        self.params.locked_joint_names = ["InvalidJoint"]
        with self.assertRaises(ValueError):
            self.robot_models._apply_locked_joints()

    def test_armature_property(self):
        self.assertTrue(
            np.array_equal(self.robot_models.armature, self.params.armature)
        )

    def test_collision_pairs(self):
        """Checking that the collision model has collision pairs."""
        self.assertTrue(
            len(self.robot_models.collision_model.collisionPairs) == 44
        )  # Number of collision pairs in the panda model

    def test_rnea(self):
        """Checking that the RNEA method works."""
        q = np.zeros(self.robot_models.robot_model.nq)
        v = np.zeros(self.robot_models.robot_model.nv)
        a = np.zeros(self.robot_models.robot_model.nv)
        robot_data = self.robot_models.robot_model.createData()
        pin.rnea(self.robot_models.robot_model, robot_data, q, v, a)


if __name__ == "__main__":
    unittest.main()
