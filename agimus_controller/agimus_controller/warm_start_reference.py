from typing import Tuple

import numpy as np

from agimus_controller.warm_start_base import WarmStartBase
from agimus_controller.trajectory import TrajectoryPoint


class WarmStartReference(WarmStartBase):
    def generate(self, reference_trajectory: list[TrajectoryPoint]):
        """
        Fills the warmstart from the reference
        Assumes that state `x` is [q, v] and
        control `u` is initialized to zeros of same shape as v
        """
        qs = np.array([point.robot_configuration for point in reference_trajectory])
        vs = np.array([point.robot_velocity for point in reference_trajectory])

        x_init = np.hstack([qs, vs])
        u_init = np.zeros_like(vs)
        return x_init, u_init
