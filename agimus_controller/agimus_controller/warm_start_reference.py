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
        Generate initial values for a warm-start of the optimization problem.

        This method uses the provided initial state and reference trajectory to compute:
        - `x0`: The initial state vector `[q, v]`, where:
            - `q` is the robot's joint configuration.
            - `v` is the robot's joint velocity.
        - `init_xs`: A list of state vectors `[q, v]` constructed from the reference trajectory.
        - `init_us`: A list of control inputs computed using inverse dynamics (RNEA)
            based on the reference trajectory.
        
        Args:
            initial_state (TrajectoryPoint): The initial state of the robot,
                containing `robot_configuration` and `robot_velocity`.
            reference_trajectory (list[TrajectoryPoint]): A list of `TrajectoryPoint` objects
                representing the reference trajectory.

        Returns:
            tuple:
                - x0 (npt.NDArray[np.float64]): The initial state vector `[q, v]`.
                - init_xs (list[npt.NDArray[np.float64]]): List of state vectors `[q, v]`
                for each point in the reference trajectory.
                - init_us (list[npt.NDArray[np.float64]]): List of control inputs computed using RNEA.
        """
        # Ensure the robot model (_rmodel) is initialized before proceeding
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
