from copy import deepcopy
import numpy as np
from random import randint
import unittest


from agimus_controller.trajectory import (
    TrajectoryBuffer,
    TrajectoryPoint,
    TrajectoryPointWeights,
    WeightedTrajectoryPoint,
)


class TestTrajectoryBuffer(unittest.TestCase):
    """
    TestOCPParamsCrocoBase unittests parameters settters and getters of OCPParamsBaseCroco class.
    """

    def __init__(self, methodName="runTest"):
        super().__init__(methodName)
        self.nv = randint(10, 100)  # Number of dof in the robot velocity
        self.nq = self.nv + 1  # Number of dof in the robot configuration

        self.trajectory_size = 200
        self.dt = 0.01
        self.dt_ns = int(1e9 * self.dt)
        self.horizon_dts = list()

    def generate_random_weighted_states(self, time_ns):
        """
        Generate random data for the TrajectoryPointWeights.
        """
        return WeightedTrajectoryPoint(
            point=TrajectoryPoint(
                time_ns=time_ns,
                robot_configuration=np.random.random(self.nq),
                robot_velocity=np.random.random(self.nv),
                robot_acceleration=np.random.random(self.nv),
                robot_effort=np.random.random(self.nv),
            ),
            weight=TrajectoryPointWeights(
                w_robot_configuration=np.random.random(self.nv),
                w_robot_velocity=np.random.random(self.nv),
                w_robot_acceleration=np.random.random(self.nv),
                w_robot_effort=np.random.random(self.nv),
            ),
        )

    def test_append_data(self):
        """
        Test adding points to the buffer.
        """
        obj = TrajectoryBuffer()
        times_ns = np.arange(
            0, 30 * self.trajectory_size * self.dt_ns, self.dt_ns, dtype=int
        )
        for time_ns in times_ns:
            obj.append(self.generate_random_weighted_states(time_ns))

        self.assertEqual(len(obj), times_ns.size)

    def test_clear_past(self):
        """
        Test adding points to the buffer.
        """
        obj = TrajectoryBuffer()
        times_ns = np.arange(
            0, 30 * self.trajectory_size * self.dt_ns, self.dt_ns, dtype=int
        )
        for time_ns in times_ns:
            obj.append(self.generate_random_weighted_states(time_ns))

        obj.clear_past(times_ns[-1] / 2, self.dt_ns)
        self.assertEqual(len(obj), times_ns.size / 2 + 2)

    def test_horizon_no_horizon_dts(self):
        """
        Test accessing the horizon from a given trajectory.
        """
        obj = TrajectoryBuffer()
        times_ns = np.arange(
            0, 30 * self.trajectory_size * self.dt_ns, self.dt_ns, dtype=int
        )
        for time_ns in times_ns:
            obj.append(self.generate_random_weighted_states(time_ns))

        horizon_size = 10
        horizon = obj.horizon(horizon_size, self.dt_ns)
        np.testing.assert_array_equal(deepcopy(horizon), obj[:horizon_size])


if __name__ == "__main__":
    unittest.main()
