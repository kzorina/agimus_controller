from dataclasses import dataclass

import numpy as np


@dataclass
class OCPParamsBaseCroco:
    """Input data structure of the OCP."""

    # Data relevant to solve the OCP
    dt: float  # Integration step of the OCP
    horizon_size: int  # Number of time steps in the horizon
    solver_iters: int  # Number of solver iterations
    qp_iters: int = 200  # Number of QP iterations (must be a multiple of 25).
    termination_tolerance: float = (
        1e-3  # Termination tolerance (norm of the KKT conditions)
    )
    eps_abs: float = 1e-6  # Absolute tolerance of the solver
    eps_rel: float = 0.0  # Relative tolerance of the solver
    callbacks: bool = False  # Flag to enable/disable callbacks
    use_filter_line_search = False  # Flag to enable/disable the filter line searchs
