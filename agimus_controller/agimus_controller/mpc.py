import time
from agimus_controller.ocp_base import OCPBase
from agimus_controller.warm_start_base import WarmStartBase
from agimus_controller.trajectory import TrajectoryBuffer, TrajectoryPoint
from mpc_output import MPCOutputData, MPCDebugData


class MPC(object):

    def __init__(self) -> None:
        self._ocp = None
        self._warm_start = None
        self._trajectory_buffer = TrajectoryBuffer()
        self._mpc_output_data = MPCOutputData()
        self._mpc_debug_data = MPCDebugData()
        self._duration_iteration = 0.0
        self._duration_horizon_update = 0.0
        self._duration_generate_warm_start = 0.0
        self._duration_ocp_solve = 0.0
        

    def setup(self, ocp: OCPBase, warm_start: WarmStartBase) -> None:
        self._ocp = ocp
        self._warm_start = warm_start

    def run(self, initial_state: TrajectoryPoint, trajectory_buffer: TrajectoryBuffer) -> MPCOutputData:
        assert self._ocp is not None
        assert self._warm_start is not None
        timer1 = time.perf_counter()
        reference_trajectory = trajectory_buffer.get_points(
            self._ocp.get_horizon_size())
        self._ocp.update_horizon_from_reference(reference_trajectory)
        timer2 = time.perf_counter()
        x_init, u_init = self._warm_start.generate(reference_trajectory)
        timer3 = time.perf_counter()
        x0 = self._ocp.get_x_from_trajectory_point(initial_state)
        self._ocp.solve(x0, x_init, u_init)
        timer4 = time.perf_counter_ns()

        self._duration_iteration = timer4 - timer1
        self._duration_horizon_update = timer2 - timer1
        self._duration_generate_warm_start = timer3 - timer2
        self._duration_ocp_solve= timer4 - timer3
