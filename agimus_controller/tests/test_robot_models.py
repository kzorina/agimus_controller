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
    def setUp(self):
        robot = robex.load("panda")
        urdf_path = Path(robot.urdf)
        srdf_path = Path(robot.urdf.replace("urdf", "srdf"))
        self.valid_args = {
            "q0": np.array([0.0, 1.0, 2.0]),
            "urdf": urdf_path,
            "srdf": srdf_path,
            "urdf_meshes_dir": urdf_path.parent.parent.parent.parent.parent,
            "free_flyer": False,
            "moving_joint_names": [f"panda_joint{x}" for x in range(3, 8)],
            "collision_as_capsule": True,
            "self_collision": True,
        }
        reduced_nq = len(self.valid_args["moving_joint_names"])
        self.valid_args["armature"] = np.linspace(0.1, 0.9, reduced_nq)

    def test_valid_initialization(self):
        """Test that the dataclass initializes correctly with valid input."""

        self.valid_args["collision_as_capsule"] = True
        self.valid_args["self_collision"] = True

        params = RobotModelParameters(**deepcopy(self.valid_args))
        self.assertEqual(params.free_flyer, self.valid_args["free_flyer"])
        self.assertEqual(
            params.moving_joint_names, self.valid_args["moving_joint_names"]
        )
        self.assertEqual(params.urdf, self.valid_args["urdf"])
        self.assertEqual(params.srdf, self.valid_args["srdf"])
        self.assertEqual(params.urdf_meshes_dir, self.valid_args["urdf_meshes_dir"])
        self.assertTrue(params.collision_as_capsule)
        self.assertTrue(params.self_collision)
        np.testing.assert_array_equal(params.q0, self.valid_args["q0"])
        np.testing.assert_array_equal(params.armature, self.valid_args["armature"])


    def test_armature_default_value(self):
        """Test that the armature is set to default if not provided."""
        del self.valid_args["armature"]
        params = RobotModelParameters(**self.valid_args)
        np.testing.assert_array_equal(
            params.armature, np.zeros(len(self.valid_args["moving_joint_names"]))
        )

    def test_armature_mismatched_shape_raises_error(self):
        """Test that a mismatched armature raises a ValueError."""
        self.valid_args["q0"] = np.array([0.0, 1.0, 2.0])
        self.valid_args["armature"] = np.array([0.1, 0.2])  # Wrong shape
        with self.assertRaises(ValueError):
            RobotModelParameters(**self.valid_args)

    def test_invalid_urdf_raises_error(self):
        """Test that an invalid URDF path raises a ValueError."""
        for val in [Path("invalid_path"), ""]:
            self.valid_args["urdf"] = val
            with self.assertRaises(ValueError):
                RobotModelParameters(**self.valid_args)

    def test_invalid_srdf_type_raises_error(self):
        """Test that a non-string SRDF path raises a ValueError."""
        self.valid_args["srdf"] = Path("invalid_path")
        with self.assertRaises(ValueError):
            RobotModelParameters(**self.valid_args)

    def test_invalid_urdf_mesh_path_type_raises_error(self):
        """Test that a non-string SRDF path raises a ValueError."""
        self.valid_args["urdf_meshes_dir"] = Path("invalid_path")
        with self.assertRaises(ValueError):
            RobotModelParameters(**self.valid_args)


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
        q0 = np.zeros(robot.model.nq)
        moving_joint_names = [f"panda_joint{x}" for x in range(3, 8)]
        reduced_nq = len(moving_joint_names)
        cls.params = RobotModelParameters(
            q0=q0,
            free_flyer=free_flyer,
            moving_joint_names=moving_joint_names,
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
        np.testing.assert_array_equal(robot_models_str._q0, self.robot_models._q0)

        self.assertEqual(
            robot_models_str.collision_model.ngeoms,
            self.robot_models.collision_model.ngeoms,
        )
        self.assertEqual(
            robot_models_str.visual_model.ngeoms,
            self.robot_models.visual_model.ngeoms,
        )

    def test_initial_configuration(self):
        np.testing.assert_array_equal(self.robot_models._q0, self.params.q0)

    def test_load_models_populates_models(self):
        self.robot_models.load_models()
        self.assertIsNotNone(self.robot_models.full_robot_model)
        self.assertIsNotNone(self.robot_models.visual_model)
        self.assertIsNotNone(self.robot_models.collision_model)

    def test_reduced_robot_model(self):
        self.robot_models.load_models()
        self.assertTrue(
            self.robot_models.robot_model.nq == len(self.params.moving_joint_names)
        )

    def test_armature_property(self):
        np.testing.assert_array_equal(self.robot_models.armature, self.params.armature)

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
