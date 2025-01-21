from collections import deque
from dataclasses import dataclass
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


@dataclass
class TrajectoryPointWeights:
    """Trajectory point weights aiming at being set in the MPC costs."""

    w_robot_configuration: npt.NDArray[np.float64] | None = None
    w_robot_velocity: npt.NDArray[np.float64] | None = None
    w_robot_acceleration: npt.NDArray[np.float64] | None = None
    w_robot_effort: npt.NDArray[np.float64] | None = None
    w_forces: dict[npt.NDArray[np.float64]] | None = None
    w_end_effector_poses: dict[npt.NDArray[np.float64]] | None = None


@dataclass
class WeightedTrajectoryPoint:
    """Trajectory point and it's corresponding weights."""

    point: TrajectoryPoint
    weight: TrajectoryPointWeights


class TrajectoryBuffer(list):
    """List of variable size in which the HPP trajectory nodes will be."""

    def clear_past(self, current_time_ns):
        while self and self[0].point.time_ns < current_time_ns:
            self.remove(0)

    def horizon(self, horizon_size, horizon_dt):
        # TBD improve this implementation in case the dt_mpc != dt_ocp
        if len(horizon_dt) != 1:
            assert len(horizon_size) == len(horizon_dt) and "Size must match."

        indexes = [0] * horizon_size
        for index, factor in zip(indexes, horizon_dt):
            index += factor
        return self[indexes]
