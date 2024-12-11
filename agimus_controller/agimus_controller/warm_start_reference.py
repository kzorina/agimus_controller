import numpy as np
import numpy.typing as npt

import pinocchio as pin

from agimus_controller.warm_start_base import WarmStartBase
from agimus_controller.trajectory import TrajectoryPoint


class WarmStartReference(WarmStartBase):
    def __init__(self) -> None:
        super().__init__()
        self._rmodel: pin.Model | None = None
        self._rdata: pin.Data | None = None

    def setup(self, rmodel: pin.Model) -> None:
        self._rmodel = rmodel
        self._rdata = self._rmodel.createData()

    def generate(
        self,
        initial_state: TrajectoryPoint,
        reference_trajectory: list[TrajectoryPoint],
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
        assert self._rmodel is not None
        x0 = np.concatenate(
            [initial_state.robot_configuration, initial_state.robot_velocity]
        )
        qs = np.array([point.robot_configuration for point in reference_trajectory])
        vs = np.array([point.robot_velocity for point in reference_trajectory])
        acs = np.array([point.robot_acceleration for point in reference_trajectory])

        x_init = list(np.hstack([qs, vs]))
        u_init = [
            pin.rnea(self._rmodel, self._rdata, q, v, a) for q, v, a in zip(qs, vs, acs)
        ]
        return x0, x_init, u_init
