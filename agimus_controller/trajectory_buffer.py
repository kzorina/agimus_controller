import numpy as np

from agimus_controller.trajectory_point import TrajectoryPoint


class TrajectoryBuffer:
    """List of variable size in which the HPP trajectory nodes will be."""

    def __init__(self, rmodel, T):
        self.rmodel = rmodel
        self.T = T
        self.trajectory: list[TrajectoryPoint] = []

    def add_trajectory_point(self, point: TrajectoryPoint):
        if len(self.trajectory) >= self.T:
            del self.trajectory[0]
        self.trajectory.append(point)

    def get_state_horizon_planning(self):
        """Return the state planning for the horizon, state is composed of joints positions and velocities"""
        x_plan = np.zeros([self.T, self.rmodel.nx])
        for idx, point in enumerate(self.trajectory):
            x_plan[idx, :] = np.concatenate([point.q, point.v])
        return x_plan

    def get_joint_acceleration_horizon(self):
        """Return the acceleration reference for the horizon, state is composed of joints positions and velocities"""
        a_plan = np.zeros([self.T, self.rmodel.nv])
        for idx, point in enumerate(self.trajectory):
            a_plan[idx, :] = point.a
        return a_plan
