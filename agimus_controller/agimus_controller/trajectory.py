import numpy as np
from collections import deque
from dataclasses import dataclass
from pinocchio import SE3, Force


@dataclass
class TrajectoryPoint:
    """Trajectory point aiming at being a reference for the MPC."""

    time_ns: int
    robot_configuration: np.ndarray
    robot_velocity: np.ndarray
    robot_acceleration: np.ndarray
    robot_effort: np.ndarray
    forces: dict[Force]  # Dictionary of pinocchio.Force
    end_effector_poses: dict[SE3]  # Dictionary of pinocchio.SE3


@dataclass
class TrajectoryPointWeights:
    """Trajectory point weights aiming at being set in the MPC costs."""

    w_robot_configuration: np.ndarray
    w_robot_velocity: np.ndarray
    w_robot_acceleration: np.ndarray
    w_robot_effort: np.ndarray
    w_forces: dict[np.ndarray]
    w_end_effector_poses: dict[np.ndarray]


@dataclass
class WeightedTrajectoryPoint:
    """Trajectory point and it's corresponding weights."""

    point: TrajectoryPoint
    weight: TrajectoryPointWeights


class TrajectoryBuffer(deque):
    """List of variable size in which the HPP trajectory nodes will be."""
