import numpy as np
from abc import ABC, abstractmethod
from agimus_controller.trajectory import ReferenceTrajectory, ReferenceTrajectoryPoint
from agimus_controller.mpc_output import MPCOutput


class OCPBase(ABC):
    def __init__(self) -> None:
        self.mpc_output_data = MPCOutput()
        self.mpc_debug_data = MPCOutput()

    @abstractmethod
    def update_horizon_from_reference(
        self, reference_trajectory: ReferenceTrajectory
    ) -> bool:
        return False

    @abstractmethod
    def get_debug_infos(self):
        return

    @abstractmethod
    def get_x_from_trajectory_point(
        self, trajectory_point: ReferenceTrajectoryPoint
    ) -> np.ndarray:
        pass

    @abstractmethod
    def get_u_from_trajectory_point(
        self, trajectory_point: ReferenceTrajectoryPoint
    ) -> np.ndarray:
        pass

    @abstractmethod
    def get_horizon_size() -> int:
        pass

    # @abstractmethod ?
    def solve(
        self, x0: np.ndarray, x_init: list[np.ndarray], u_init: list[np.ndarray]
    ) -> bool:
        return False

    def get_output(self) -> MPCOutput:
        self.mpc_output.feed_forward_terms = list()
        self.mpc_output.ricatti_gains = list()
        self.mpc_output.x = list()
        self.mpc_output.u = list()
