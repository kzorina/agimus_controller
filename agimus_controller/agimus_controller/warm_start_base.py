from abc import ABC, abstractmethod
import numpy as np

from agimus_controller.trajectory import TrajectoryPoint


class WarmStartBase(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def generate(
        self,
        reference_trajectory: list[TrajectoryPoint],
        previous_solution: list[TrajectoryPoint],
    ) -> tuple(np.ndarray, np.ndarray):
        """Returns x_init, u_init."""
        ...
