import numpy as np
import numpy.typing as npt
from dataclasses import dataclass


@dataclass
class MPCResults:
    """Output data structure of the MPC."""

    states: list[npt.NDArray[np.float64]]
    ricatti_gains: list[npt.NDArray[np.float64]]
    feed_forward_terms: list[npt.NDArray[np.float64]]


@dataclass
class OCPDebugData:
    # Solver infos
    problem_solved: bool = False

    # Debug data
    robot_configurations: list[npt.NDArray[np.float64]]
    robot_velocities: list[npt.NDArray[np.float64]]
    robot_accelerations: list[npt.NDArray[np.float64]]
    robot_efforts: list[npt.NDArray[np.float64]]
    operational_frame_forces: list[npt.NDArray[np.float64]]
    operational_frame_poses: list[npt.NDArray[np.float64]]
    robot_configurations_ref: list[npt.NDArray[np.float64]]
    robot_velocities_ref: list[npt.NDArray[np.float64]]
    robot_accelerations_ref: list[npt.NDArray[np.float64]]
    robot_efforts_ref: list[npt.NDArray[np.float64]]
    operational_frame_forces_ref: list[npt.NDArray[np.float64]]
    operational_frame_poses_ref: list[npt.NDArray[np.float64]]
    kkt_norms: list[np.float64]
    collision_distance_residuals: list[dict[np.float64]]


@dataclass
class MPCDebugData:
    ocp: OCPDebugData
    # Timers
    duration_iteration: float = 0.0
    duration_horizon_update: float = 0.0
    duration_generate_warm_start: float = 0.0
    duration_ocp_solve: float = 0.0
