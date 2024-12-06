import time
from agimus_controller.ocp_base import OCPBase
from agimus_controller.warm_start_base import WarmStartBase
from agimus_controller.trajectory import (
    TrajectoryPoint,
    WeightedTrajectoryPoint,
    TrajectoryBuffer,
)
from agimus_controller.mpc_data import MPCResults, MPCDebugData


class MPC(object):
    def __init__(self) -> None:
        self._ocp = None
        self._warm_start = None
        self._mpc_results: MPCResults = None
        self._mpc_debug_data: MPCDebugData = None
        self._buffer = TrajectoryBuffer()

    def setup(self, ocp: OCPBase, warm_start: WarmStartBase) -> None:
        self._ocp = ocp
        self._warm_start = warm_start

    def run(self, initial_state: TrajectoryPoint, current_time_ns: int) -> MPCResults:
        assert self._ocp is not None
        assert self._warm_start is not None
        timer1 = time.perf_counter_ns()
        self._clear_buffer_past(current_time_ns)
        reference_trajectory = self._extract_horizon_from_buffer()
        self._ocp.set_reference_horizon(reference_trajectory)
        timer2 = time.perf_counter_ns()
        x0, x_init, u_init = self._warm_start.generate(
            initial_state, reference_trajectory
        )
        timer3 = time.perf_counter_ns()
        self._ocp.solve(x0, x_init, u_init)
        timer4 = time.perf_counter_ns()

        # Extract the solution.
        self._mpc_results = self._ocp.mpc_results
        self._mpc_debug_data = self._ocp.debug_data
        self._mpc_debug_data.duration_iteration_ns = timer4 - timer1
        self._mpc_debug_data.duration_horizon_update_ns = timer2 - timer1
        self._mpc_debug_data.duration_generate_warm_start_ns = timer3 - timer2
        self._mpc_debug_data.duration_ocp_solve_ns = timer4 - timer3

    def add_trajectory_point(self, trajectory_point: WeightedTrajectoryPoint):
        self._buffer.append(trajectory_point)

    def add_trajectory_points(self, trajectory_points: list[WeightedTrajectoryPoint]):
        for trajectory_point in trajectory_points:
            self.add_trajectory_point(trajectory_point)

    def _clear_buffer_past(self, current_time_ns: int):
        self._buffer.clear_past(current_time_ns)

    def _extract_horizon_from_buffer(self):
        return self._buffer[: self._ocp.horizon_size]
