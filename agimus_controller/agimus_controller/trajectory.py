import numpy as np
from collections import deque
from dataclasses import dataclass
from pinocchio import SE3, Force


@dataclass
class TrajectoryPoint:
    """Trajectory point aiming at being a reference for the MPC."""

    robot_configuration: np.ndarray
    robot_velocity: np.ndarray
    robot_acceleration: np.ndarray
    robot_effort: np.ndarray
    forces: dict[Force]  # Dictionary of pinocchio.Force
    end_effector_poses: dict[SE3]  # Dictionary of pinocchio.SE3


class TrajectoryBuffer(deque):
    """List of variable size in which the HPP trajectory nodes will be."""

    def __init__(self) -> None:
        super().__init__()

    def add_trajectory_point(self, trajectory_point: TrajectoryPoint):
        """
        Add trajectory point to the buffer.
        """
        self.append(trajectory_point)

    def get_size(self):
        """
        Returns the size of the buffer until the first invalid TrajectoryPoint.
        """
        return self.__len__()

    def get_points(self, nb_points: int):
        """Get nb_points of valid TrajectoryPoints from the buffer."""
        buffer_size = self.get_size()
        if nb_points > buffer_size:
            raise Exception(
                f"the buffer size is {buffer_size} and you ask for {nb_points} points"
            )
        else:
            return [self.popleft() for _ in range(nb_points)]

    def get_last_point(self):
        """Return last point in buffer without removing it from buffer."""
        if not self.get_size():
            return None
        else:
            return self._buffer[-1]
