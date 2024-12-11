from dataclasses import dataclass
from typing import List

import numpy as np
import numpy.typing as npt

from agimus_controller.trajectory import WeightedTrajectoryPoint

@dataclass
class OCPParamsBase:
    """Input data structure of the OCP."""

    # Data relevant to solve the OCP
    dt: float # Integration step of the OCP
    T: int # Number of time steps in the horizon
    qp_iters: int # Number of QP iterations
    solver_iters: int # Number of solver iterations
    
    # Weights, costs & helpers for the costs & constraints
    WeightedTrajectoryPoints: List[WeightedTrajectoryPoint] # List of weighted trajectory points    
    armature: npt.ArrayLike # Armature of the robot
    ee_name: str # Name of the end-effector
    p_target: npt.ArrayLike # Target position of the end-effector in R3
        
