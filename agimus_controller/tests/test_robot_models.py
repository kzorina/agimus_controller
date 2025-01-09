from os.path import dirname
from pathlib import Path
import unittest

import example_robot_data as robex
import numpy as np

from agimus_controller.factory.robot_model import RobotModelParameters, RobotModels


class TestRobotModelParameters(unittest.TestCase):
    def test_valid_initialization(self):
        """Test that the dataclass initializes correctly with valid input."""

        robot = robex.load("panda")
        urdf_path = robot.urdf
        srdf_path = robot.urdf.replace("urdf", "srdf")
        urdf_meshes_dir = dirname(dirname(robot.urdf))
        q0 = np.zeros(robot.model.nq)
        params = RobotModelParameters(
            q0=q0,
            free_flyer=True,
            locked_joint_names=["panda_joint1", "panda_joint2"],
            urdf_path=urdf_path,
            srdf_path=srdf_path,
            urdf_meshes_dir=urdf_meshes_dir,
            collision_as_capsule=True,
            self_collision=True,
            armature=np.linspace(0.1, 0.9, 9),
        )

        self.assertTrue(np.array_equal(params.q0, q0))
        self.assertEqual(params.free_flyer, True)
        self.assertEqual(params.locked_joint_names, ["panda_joint1", "panda_joint2"])
        self.assertEqual(params.urdf_path, urdf_path)
        self.assertEqual(params.srdf_path, srdf_path)
        self.assertEqual(params.urdf_meshes_dir, urdf_meshes_dir)
        self.assertTrue(params.collision_as_capsule)
        self.assertTrue(params.self_collision)
        self.assertTrue(np.array_equal(params.armature, np.linspace(0.1, 0.9, 9)))

    def test_empty_q0_raises_error(self):
        """Test that an empty q0 raises a ValueError."""
        with self.assertRaises(ValueError):
            RobotModelParameters(q0=np.array([]))

    def test_armature_default_value(self):
        """Test that the armature is set to default if not provided."""
        q0 = np.array([0.0, 1.0, 2.0])
        params = RobotModelParameters(q0=q0)
        self.assertTrue(np.array_equal(params.armature, np.zeros_like(q0)))

    def test_armature_mismatched_shape_raises_error(self):
        """Test that a mismatched armature raises a ValueError."""
        q0 = np.array([0.0, 1.0, 2.0])
        armature = np.array([0.1, 0.2])  # Wrong shape
        with self.assertRaises(ValueError):
            RobotModelParameters(q0=q0, armature=armature)

    def test_invalid_urdf_path_raises_error(self):
        """Test that an invalid URDF path raises a ValueError."""
        q0 = np.array([0.0, 1.0, 2.0])
        with self.assertRaises(ValueError):
            RobotModelParameters(q0=q0, urdf_path=None)

    def test_invalid_srdf_path_type_raises_error(self):
        """Test that a non-string SRDF path raises a ValueError."""
        q0 = np.array([0.0, 1.0, 2.0])
        with self.assertRaises(ValueError):
            RobotModelParameters(q0=q0, srdf_path=12345)

    def test_collision_color_default(self):
        """Test that the default collision color is set correctly."""
        q0 = np.array([0.0, 1.0, 2.0])
        params = RobotModelParameters(q0=q0)
        self.assertTrue(
            np.array_equal(
                params.collision_color, np.array([249.0, 136.0, 126.0, 125.0]) / 255.0
            )
        )
