from abc import ABC, abstractmethod
from typing import List

import numpy as np
import numpy.typing as npt

from agimus_controller.mpc_data import OCPResults, OCPDebugData
from agimus_controller.trajectory import WeightedTrajectoryPoint


class OCPBase(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def set_reference_horizon(
        self, reference_trajectory: list[WeightedTrajectoryPoint]
    ) -> None:
        pass

    @abstractmethod
    @property
    def horizon_size() -> int:
        pass

    @abstractmethod
    @property
    def dt() -> int:
        pass

    @abstractmethod
    @property
    def x0() -> npt.NDArray[np.float64]:
        pass

    @abstractmethod
    def solve(
        self,
        x_init: List[npt.NDArray[np.float64]],
        u_init: List[npt.NDArray[np.float64]],
    ) -> None:
        pass

    @abstractmethod
    @property
    def ocp_results(self) -> OCPResults:
        pass

    @abstractmethod
    @property
    def debug_data(self) -> OCPDebugData:
        pass
