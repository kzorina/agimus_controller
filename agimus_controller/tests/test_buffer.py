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

        self.trajectory_size = 1000
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
        Test clearing the past of the buffer.
        """
        obj = TrajectoryBuffer()
        times_ns = np.arange(
            0, 30 * self.trajectory_size * self.dt_ns, self.dt_ns, dtype=int
        )
        for time_ns in times_ns:
            obj.append(self.generate_random_weighted_states(time_ns))

        obj.clear_past(times_ns[-1] / 2, self.dt_ns)
        self.assertEqual(len(obj), times_ns.size / 2 + 2)

    def test_find_next_index(self):
        """
        Test find the first index from the current time.
        """
        obj = TrajectoryBuffer()
        times_ns = np.arange(
            0, 30 * self.trajectory_size * self.dt_ns, self.dt_ns, dtype=int
        )
        for time_ns in times_ns:
            obj.append(self.generate_random_weighted_states(time_ns))

        for i in range(min(10, times_ns.size)):
            start_out = obj.find_next_index(i * self.dt_ns)
            start_test = i + 1
            self.assertEqual(start_out, start_test)

    def test_compute_horizon_index(self):
        """
        Test computing the time indexes from dt_factor_n_seq.
        """
        obj = TrajectoryBuffer()
        dt_factor_n_seq = [(1, 2), (2, 2), (3, 2), (4, 2), (5, 2)]
        indexes_out = obj.compute_horizon_indexes(dt_factor_n_seq)
        indexes_test = [0, 1, 3, 5, 8, 11, 15, 19, 24, 29]
        np.testing.assert_equal(indexes_out, indexes_test)

    def test_horizon_with_simple_horizon_dts(self):
        """
        Test computing the horizon from the horizon_dts format.
        """
        obj = TrajectoryBuffer()
        times_ns = np.arange(
            0, 30 * self.trajectory_size * self.dt_ns, self.dt_ns, dtype=int
        )
        for time_ns in times_ns:
            obj.append(self.generate_random_weighted_states(time_ns))

        horizon_size = 10
        horizon = obj.horizon(
            current_time_ns=0,
            dt_factor_n_seq=[(1, horizon_size)],
        )
        np.testing.assert_array_equal(
            deepcopy(horizon),
            obj[1 : horizon_size + 1],
        )

    def test_horizon_with_simple_horizon_dts_and_offsets(self):
        """
        Test computing the horizon from the horizon_dts format
        with a time offset.
        """
        obj = TrajectoryBuffer()
        times_ns = np.arange(
            0, 30 * self.trajectory_size * self.dt_ns, self.dt_ns, dtype=int
        )
        for time_ns in times_ns:
            obj.append(self.generate_random_weighted_states(time_ns))

        horizon_size = 10
        time_factor = randint(0, 5)
        current_time_ns = time_factor * self.dt_ns
        horizon = obj.horizon(
            current_time_ns=current_time_ns,
            dt_factor_n_seq=[(1, horizon_size)],
        )
        np.testing.assert_array_equal(
            deepcopy(horizon),
            obj[time_factor + 1 : time_factor + 1 + horizon_size],
        )

    def test_horizon_with_more_complex_horizon_dts(self):
        """
        Test computing the horizon from complex horizon_dts.
        """
        obj = TrajectoryBuffer()
        times_ns = np.arange(
            0, 30 * self.trajectory_size * self.dt_ns, self.dt_ns, dtype=int
        )
        for time_ns in times_ns:
            obj.append(self.generate_random_weighted_states(time_ns))

        horizon = obj.horizon(
            current_time_ns=0,
            dt_factor_n_seq=[(1, 2), (2, 2), (3, 2), (4, 2), (5, 2)],
        )
        np.testing.assert_array_equal(
            deepcopy(horizon),
            [obj[index + 1] for index in [0, 1, 3, 5, 8, 11, 15, 19, 24, 29]],
        )

    def test_horizon_with_more_complex_horizon_dts_with_offset(self):
        """
        Test computing the horizon from complex horizon_dts
        with a time offset.
        """
        obj = TrajectoryBuffer()
        times_ns = np.arange(
            0, 30 * self.trajectory_size * self.dt_ns, self.dt_ns, dtype=int
        )
        for time_ns in times_ns:
            obj.append(self.generate_random_weighted_states(time_ns))

        time_factor = randint(0, 5)
        current_time_ns = time_factor * self.dt_ns
        horizon = obj.horizon(
            current_time_ns=current_time_ns,
            dt_factor_n_seq=[(1, 2), (2, 2), (3, 2), (4, 2), (5, 2)],
        )
        np.testing.assert_array_equal(
            deepcopy(horizon),
            [
                obj[time_factor + 1 + index]
                for index in [0, 1, 3, 5, 8, 11, 15, 19, 24, 29]
            ],
        )


if __name__ == "__main__":
    unittest.main()
