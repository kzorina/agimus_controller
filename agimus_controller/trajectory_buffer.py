from __future__ import annotations
import numpy as np
from collections import deque
from __future__ import annotations

from agimus_controller.trajectory_point import TrajectoryPoint, PointAttribute


class TrajectoryBuffer:
    """List of variable size in which the HPP trajectory nodes will be."""

    def __init__(self):
        self._buffer: deque[TrajectoryPoint] = deque()
        print("type ", type(self._buffer))
        self.size = 0
        self.nq = 0
        self.nv = 0
        self.nx = 0

    def initialize(self, start: TrajectoryPoint):
        self._buffer.append(start)
        self.nq = start.q.shape[0]
        self.nv = start.v.shape[0]
        self.nx = self.nq + self.nv

    def add_trajectory_point(self, trajectory_point: TrajectoryPoint):
        """Add trajectory point to the buffer if it matches the size of q and v"""
        assert self.nq == trajectory_point.q.shape[0]
        assert self.nv == trajectory_point.v.shape[0]
        self._buffer.append(trajectory_point)

    def get_size(self, attributes: list[PointAttribute]):
        """Returns the size of the buffer until the first invalid TrajectoryPoint"""
        for idx, point in enumerate(self._buffer):
            for attribute in attributes:
                if not point.attribute_is_valid(attribute):
                    print(
                        f"buffer point at index {idx} is not valid for attribute {attribute}"
                    )
                    return idx
        return len(self._buffer)

    def get_points(self, nb_points: int, attributes: list[PointAttribute]):
        """Get nb_points of valid TrajectoryPoints from the buffer"""
        buffer_size = self.get_size(attributes)
        if nb_points > buffer_size:
            raise Exception(
                "the buffer size is {buffer_size} and you ask for {nb_points}"
            )
        else:
            return [self._buffer.popleft() for _ in range(nb_points)]

    def get_state_horizon_planning(self):
        """Return the state planning for the horizon, state is composed of joints positions and velocities"""
        x_plan = np.zeros([len(self._buffer), self.nx])
        for idx, point in enumerate(self._buffer):
            x_plan[idx, :] = np.concatenate([point.q, point.v])
        return x_plan

    def get_joint_acceleration_horizon(self):
        """Return the acceleration reference for the horizon, state is composed of joints positions and velocities"""
        a_plan = np.zeros([len(self._buffer), self.nv])
        for idx, point in enumerate(self._buffer):
            a_plan[idx, :] = point.a
        return a_plan

    def get_buffer(self):
        return self._buffer.copy()
