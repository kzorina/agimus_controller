from abc import ABC, abstractmethod
import numpy as np

from agimus_controller.mpc_data import OCPResults, OCPDebugData
from agimus_controller.trajectory import WeightedTrajectoryPoint


class OCPBase(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def set_reference_horizon(
        self, reference_trajectory: list[WeightedTrajectoryPoint]
    ) -> None:
        ...

    @abstractmethod
    @property
    def horizon_size() -> int:
        ...

    @abstractmethod
    @property
    def dt() -> int:
        ...

    @abstractmethod
    def solve(
        self, x0: np.ndarray, x_init: list[np.ndarray], u_init: list[np.ndarray]
    ) -> None:
        ...

    @abstractmethod
    @property
    def ocp_results(self) -> OCPResults:
        ...

    @abstractmethod
    @property
    def debug_data(self) -> OCPDebugData:
        ...