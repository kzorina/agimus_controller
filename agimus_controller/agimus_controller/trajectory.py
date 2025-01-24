import bisect
from dataclasses import dataclass
from itertools import accumulate
import numpy as np
import numpy.typing as npt
from pinocchio import SE3, Force


@dataclass
class TrajectoryPoint:
    """Trajectory point aiming at being a reference for the MPC."""

    time_ns: int | None = None
    robot_configuration: npt.NDArray[np.float64] | None = None
    robot_velocity: npt.NDArray[np.float64] | None = None
    robot_acceleration: npt.NDArray[np.float64] | None = None
    robot_effort: npt.NDArray[np.float64] | None = None
    forces: dict[Force] | None = None  # Dictionary of pinocchio.Force
    end_effector_poses: dict[SE3] | None = None  # Dictionary of pinocchio.SE3

    def __eq__(self, other):
        if not isinstance(other, TrajectoryPoint):
            return False

        # Compare scalar values directly
        if self.time_ns != other.time_ns:
            return False

        # Compare numpy arrays (ignoring None values)
        if (
            self.robot_configuration is not None
            and other.robot_configuration is not None
        ):
            if not np.array_equal(self.robot_configuration, other.robot_configuration):
                return False
        elif (
            self.robot_configuration is not None
            or other.robot_configuration is not None
        ):
            return False

        if self.robot_velocity is not None and other.robot_velocity is not None:
            if not np.array_equal(self.robot_velocity, other.robot_velocity):
                return False
        elif self.robot_velocity is not None or other.robot_velocity is not None:
            return False

        if self.robot_acceleration is not None and other.robot_acceleration is not None:
            if not np.array_equal(self.robot_acceleration, other.robot_acceleration):
                return False
        elif (
            self.robot_acceleration is not None or other.robot_acceleration is not None
        ):
            return False

        if self.robot_effort is not None and other.robot_effort is not None:
            if not np.array_equal(self.robot_effort, other.robot_effort):
                return False
        elif self.robot_effort is not None or other.robot_effort is not None:
            return False

        # Compare dictionaries (forces and end_effector_poses)
        if self.forces != other.forces:
            return False

        if self.end_effector_poses != other.end_effector_poses:
            return False

        return True


@dataclass
class TrajectoryPointWeights:
    """Trajectory point weights aiming at being set in the MPC costs."""

    w_robot_configuration: npt.NDArray[np.float64] | None = None
    w_robot_velocity: npt.NDArray[np.float64] | None = None
    w_robot_acceleration: npt.NDArray[np.float64] | None = None
    w_robot_effort: npt.NDArray[np.float64] | None = None
    w_forces: dict[npt.NDArray[np.float64]] | None = None
    w_end_effector_poses: dict[npt.NDArray[np.float64]] | None = None

    def __eq__(self, other):
        if not isinstance(other, TrajectoryPointWeights):
            return False

        # Compare numpy arrays (weights)
        if (
            self.w_robot_configuration is not None
            and other.w_robot_configuration is not None
        ):
            if not np.array_equal(
                self.w_robot_configuration, other.w_robot_configuration
            ):
                return False
        elif (
            self.w_robot_configuration is not None
            or other.w_robot_configuration is not None
        ):
            return False

        if self.w_robot_velocity is not None and other.w_robot_velocity is not None:
            if not np.array_equal(self.w_robot_velocity, other.w_robot_velocity):
                return False
        elif self.w_robot_velocity is not None or other.w_robot_velocity is not None:
            return False

        if (
            self.w_robot_acceleration is not None
            and other.w_robot_acceleration is not None
        ):
            if not np.array_equal(
                self.w_robot_acceleration, other.w_robot_acceleration
            ):
                return False
        elif (
            self.w_robot_acceleration is not None
            or other.w_robot_acceleration is not None
        ):
            return False

        if self.w_robot_effort is not None and other.w_robot_effort is not None:
            if not np.array_equal(self.w_robot_effort, other.w_robot_effort):
                return False
        elif self.w_robot_effort is not None or other.w_robot_effort is not None:
            return False

        if self.w_forces != other.w_forces:
            return False

        if self.w_end_effector_poses != other.w_end_effector_poses:
            return False

        return True


@dataclass
class WeightedTrajectoryPoint:
    """Trajectory point and it's corresponding weights."""

    point: TrajectoryPoint
    weight: TrajectoryPointWeights

    def __eq__(self, other):
        if not isinstance(other, WeightedTrajectoryPoint):
            return False

        # Compare the 'point' and 'weight' attributes
        if self.point != other.point:
            return False

        if self.weight != other.weight:
            return False

        return True


class TrajectoryBuffer(list):
    """List of variable size in which the HPP trajectory nodes will be."""

    def clear_past(self, current_time_ns, dt_ns):
        while len(self) > 2 and self[0].point.time_ns + 2 * dt_ns < current_time_ns:
            self.pop(0)

    def find_next_index(self, current_time_ns):
        # Use bisect_right directly on the time_ns values without extracting them
        index = bisect.bisect_right(
            [wpoint.point.time_ns for wpoint in self], current_time_ns
        )

        # Ensure that the index is within bounds
        if index < len(self):
            return index
        else:
            raise LookupError(
                "current_time_ns is likely greater than the trajectory horizon time."
            )

    def horizon(self, current_time_ns, horizon_size, horizon_dts=list()):
        # Starting index:
        start = self.find_next_index(current_time_ns)

        # Ensure sizes match when `horizon_dts` is provided
        if horizon_dts:
            assert len(horizon_size) == len(
                horizon_dts
            ), "Size of horizon_size and horizon_dts must match."

            # Instead of creating an intermediate list,
            # directly compute the cumulative sum
            # and use it in the loop
            cumulative_sum = 0
            for dt in horizon_dts:
                cumulative_sum += dt - 1
                yield self[start + cumulative_sum]  # Yielding on-the-fly
        else:
            # Generate range directly and access elements in one go
            for i in range(horizon_size):
                yield self[start + i]  # Yielding on-the-fly
