from abc import abstractmethod

import crocoddyl
import mim_solvers
import numpy as np
import numpy.typing as npt

from agimus_controller.factory.robot_model import RobotModels
from agimus_controller.mpc_data import OCPResults, OCPDebugData
from agimus_controller.ocp_base import OCPBase
from agimus_controller.ocp_param_base import OCPParamsBaseCroco
from agimus_controller.trajectory import TrajectoryPointWeights


class OCPBaseCroco(OCPBase):
    def __init__(
        self,
        robot_models: RobotModels,
        ocp_params: OCPParamsBaseCroco,
    ) -> None:
        """Defines common behavior for all OCP using croccodyl. This is an abstract class with some helpers to design OCPs in a more friendly way.

        Args:
            robot_models (RobotModelFactory): All models of the robot.
            ocp_params (OCPParamsBaseCroco): Input data structure of the OCP.
        """
        # Setting the robot model
        self._robot_models = robot_models
        self._robot_model = self._robot_models.robot_model
        self._collision_model = self._robot_models.collision_model
        self._armature = self._robot_models._params.armature

        # Stat and actuation model
        self._state = crocoddyl.StateMultibody(self._robot_model)
        self._actuation = crocoddyl.ActuationModelFull(self._state)

        # Setting the OCP parameters
        self._ocp_params = ocp_params
        self._ocp = None
        self._ocp_results = None

    @property
    def horizon_size(self) -> int:
        """Number of time steps in the horizon."""
        return self._ocp_params.horizon_size

    @property
    def dt(self) -> float:
        """Integration step of the OCP."""
        return self._ocp_params.dt

    @abstractmethod
    def create_running_model_list(self) -> list[crocoddyl.ActionModelAbstract]:
        """Create the list of running models."""
        pass

    @abstractmethod
    def create_terminal_model(self) -> crocoddyl.ActionModelAbstract:
        """Create the terminal model."""
        pass

    @abstractmethod
    def update_crocoddyl_problem(
        self,
        x0: npt.NDArray[np.float64],
        weighted_trajectory_points: list[TrajectoryPointWeights],
    ) -> None:
        """Update the Crocoddyl problem's references, weights and x0."""
        pass

    def solve(
        self,
        x0: npt.NDArray[np.float64],
        x_warmstart: list[npt.NDArray[np.float64]],
        u_warmstart: list[npt.NDArray[np.float64]],
    ) -> None:
        """Solves the OCP.
        The results can be accessed through the ocp_results property.

        Args:
            x0 (npt.NDArray[np.float64]): Current state of the robot.
            x_warmstart (list[npt.NDArray[np.float64]]): Predicted states for the OCP.
            u_warmstart (list[npt.NDArray[np.float64]]): Predicted control inputs for the OCP.
        """
        ### Creation of the state and actuation models

        if self._ocp is None:
            # Create the running models
            self.running_model_list = self.create_running_model_list()
            # Create the terminal model
            self.terminal_model = self.create_terminal_model()
            # Create the shooting problem
            self._problem = crocoddyl.ShootingProblem(
                x0, self.running_model_list, self.terminal_model
            )
            # Create solver + callbacks
            self._ocp = mim_solvers.SolverCSQP(self._problem)

            # Merit function
            self._ocp.use_filter_line_search = self._ocp_params.use_filter_line_search

            # Parameters of the solver
            self._ocp.termination_tolerance = self._ocp_params.termination_tolerance
            self._ocp.max_qp_iters = self._ocp_params.qp_iters
            self._ocp.eps_abs = self._ocp_params.eps_abs
            self._ocp.eps_rel = self._ocp_params.eps_rel
            self._ocp.with_callbacks = self._ocp_params.callbacks

        # Creating the warmstart lists for the solver
        # Solve the OCP
        self._ocp.solve([x0] + x_warmstart, u_warmstart, self._ocp_params.solver_iters)

        # Store the results
        self.ocp_results = OCPResults(
            states=self._ocp.xs,
            ricatti_gains=self._ocp.K,
            feed_forward_terms=self._ocp.us,
        )

    @property
    def ocp_results(self) -> OCPResults:
        """Output data structure of the OCP.

        Returns:
            OCPResults: Output data structure of the OCP. It contains the states, Ricatti gains, and feed-forward terms.
        """
        return self._ocp_results

    @ocp_results.setter
    def ocp_results(self, value: OCPResults) -> None:
        """Set the output data structure of the OCP.

        Args:
            value (OCPResults): New output data structure of the OCP.
        """
        self._ocp_results = value

    @property
    def debug_data(self) -> OCPDebugData:
        pass

    @debug_data.setter
    def debug_data(self, value: OCPDebugData) -> None:
        pass
