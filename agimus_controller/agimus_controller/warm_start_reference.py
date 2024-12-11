import numpy as np
import numpy.typing as npt

from agimus_controller.warm_start_base import WarmStartBase
from agimus_controller.trajectory import TrajectoryPoint


class WarmStartReference(WarmStartBase):
    def generate(
        self, reference_trajectory: list[TrajectoryPoint]
    ) -> tuple[
        npt.NDArray[np.float64],
        list[npt.NDArray[np.float64]],
        list[npt.NDArray[np.float64]],
    ]:
        """
        Fills the warmstart from the reference
        Assumes that state `x` is [q, v] and
        control `u` is initialized to zeros of same shape as v
        """
        qs = np.array([point.robot_configuration for point in reference_trajectory])
        vs = np.array([point.robot_velocity for point in reference_trajectory])

        x_init = list(np.hstack([qs, vs]))
        u_init = list(np.zeros_like(vs))
        return x_init[0], x_init[1:], u_init
