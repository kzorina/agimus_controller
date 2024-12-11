import unittest

import numpy as np

from agimus_controller.warm_start_reference import WarmStartReference
from agimus_controller.trajectory import TrajectoryPoint


class TestWarmStart(unittest.TestCase):
    def test_initialization(self):
        ws = WarmStartReference()
        self.assertEqual(ws._previous_solution, [])

    def test_generate(self):
        ws = WarmStartReference()
        num_points = 10
        random_qs = np.random.randn(num_points, 7)
        random_vs = np.random.randn(num_points, 7)
        reference_trajectory = [
            TrajectoryPoint(robot_configuration=q, robot_velocity=v)
            for q, v in zip(random_qs, random_vs)
        ]
        # Create the expected stacked array
        expected_x0 = np.hstack((random_qs[0], random_vs[0]))
        expected_x_init = np.hstack((random_qs[1:], random_vs[1:]))
        expected_u_init = np.zeros_like(random_vs)

        # Act
        x0, x_init, u_init = ws.generate(reference_trajectory)
        x_init = np.array(x_init)
        u_init = np.array(u_init)

        # Assert
        # Check shapes
        self.assertEqual(x_init.shape, expected_x_init.shape)
        self.assertEqual(u_init.shape, expected_u_init.shape)

        # Check values (assuming `generate` would use these random inputs)
        np.testing.assert_array_equal(x0, expected_x0)
        np.testing.assert_array_equal(x_init, expected_x_init)
        np.testing.assert_array_equal(u_init, expected_u_init)

        # Additional sanity checks
        self.assertTrue(np.all(u_init == 0))
