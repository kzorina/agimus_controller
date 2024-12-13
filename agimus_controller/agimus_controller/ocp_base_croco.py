import crocoddyl
import mim_solvers
import numpy as np
import numpy.typing as npt
import pinocchio as pin

from agimus_controller.mpc_data import OCPResults, OCPDebugData
from agimus_controller.ocp_base import OCPBase
from agimus_controller.ocp_param_base import OCPParamsCrocoBase
from agimus_controller.factory.robot_model import RobotModelFactory

class OCPCrocoBase(OCPBase):
    def __init__(
        self,
        robot_model: RobotModelFactory,
        OCPParams: OCPParamsCrocoBase,
    ) -> None:
        """Defines common behavior for all OCP using croccodyl. This is an abstract class with some helpers to design OCPs in a more friendly way.

        Args:
            rmodel (pin.Model): Robot model.
            cmodel (pin.GeometryModel): Collision Model of the robot.
            OCPParams (OCPParamsBase): Input data structure of the OCP.
        """
        # Setting the robot model
        self._robot_model = robot_model
        self._rmodel = self._robot_model._rmodel
        self._cmodel = self._robot_model._complete_collision_model
        self._armature = self._robot_model._params.armature
        
        # Setting the OCP parameters
        self._OCPParams = OCPParams

    @property
    def horizon_size(self) -> int:
        """Number of time steps in the horizon."""
        return self._OCPParams.T

    @property
    def dt(self) -> np.float64:
        """Integration step of the OCP."""
        return self._OCPParams.dt

    def solve(
        self,
        x0: npt.NDArray[np.float64],
        x_warmstart: list[npt.NDArray[np.float64]],
        u_warmstart: list[npt.NDArray[np.float64]],
    ) -> bool:
        """Solves the OCP. Returns True if the OCP was solved successfully, False otherwise.

        Args:
            x0 (npt.NDArray[np.float64]): Current state of the robot.
            x_warmstart (list[npt.NDArray[np.float64]]): Warmstart values for the state. This doesn't include the current state.
            u_warmstart (list[npt.NDArray[np.float64]]): Warmstart values for the control inputs.

        Returns:
            bool: True if the OCP was solved successfully, False otherwise.
        """
        ### Creation of the state and actuation models

        # Stat and actuation model
        self._state = crocoddyl.StateMultibody(self._rmodel)
        self._actuation = crocoddyl.ActuationModelFull(self._state)

        self._runningModelList = []
        # Running & terminal cost models
        for t in range(self.horizon_size):
            ### Creation of cost terms
            # State Regularization cost
            xResidual = crocoddyl.ResidualModelState(self._state, self.x0)
            xRegCost = crocoddyl.CostModelResidual(self._state, xResidual)

            # Control Regularization cost
            uResidual = crocoddyl.ResidualModelControl(self._state)
            uRegCost = crocoddyl.CostModelResidual(self._state, uResidual)

            # End effector frame cost
            frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(
                self._state,
                self._OCPParams.ee_name,
                self._OCPParams.WeightedTrajectoryPoints[t]
                .point.end_effector_poses[self._OCPParams.ee_name]
                .translation,
            )

            goalTrackingCost = crocoddyl.CostModelResidual(
                self._state, frameTranslationResidual
            )

            # Adding costs to the models
            if t < self.horizon_size - 1:
                runningCostModel = crocoddyl.CostModelSum(self._state)
                runningCostModel.addCost(
                    "stateReg",
                    xRegCost,
                    np.concatenate(
                        [
                            self._OCPParams.WeightedTrajectoryPoints[
                                t
                            ].weight.w_robot_configuration,
                            self._OCPParams.WeightedTrajectoryPoints[
                                t
                            ].weight.w_robot_velocity,
                        ]
                    ),
                )
                runningCostModel.addCost(
                    "ctrlRegGrav",
                    uRegCost,
                    self._OCPParams.WeightedTrajectoryPoints[t].weight.w_robot_effort,
                )
                runningCostModel.addCost(
                    "gripperPoseRM",
                    goalTrackingCost,
                    self._OCPParams.WeightedTrajectoryPoints[
                        t
                    ].weight.w_end_effector_poses,
                )
                # Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions
                self._running_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(
                    self._state,
                    self._actuation,
                    runningCostModel,
                )
                # Create Integrated Action Model (IAM), i.e. Euler integration of continuous dynamics and cost
                runningModel = crocoddyl.IntegratedActionModelEuler(
                    self._running_DAM, self.dt
                )
                runningModel.differential.armature = OCPParamsCrocoBase.armature
                self._runningModelList.append(runningModel)
            else:
                terminalCostModel = crocoddyl.CostModelSum(self._state)
                terminalCostModel.addCost(
                    "stateReg",
                    xRegCost,
                    np.concatenate(
                        [
                            self._OCPParams.WeightedTrajectoryPoints[
                                t
                            ].weight.w_robot_configuration,
                            self._OCPParams.WeightedTrajectoryPoints[
                                t
                            ].weight.w_robot_velocity,
                        ]
                    ),
                )
                terminalCostModel.addCost(
                    "gripperPose",
                    goalTrackingCost,
                    self._OCPParams.WeightedTrajectoryPoints[
                        t
                    ].weight.w_end_effector_poses,
                )

                self._terminal_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(
                    self._state, self._actuation, terminalCostModel
                )

                self._terminalModel = crocoddyl.IntegratedActionModelEuler(
                    self._terminal_DAM, 0.0
                )
                self._terminalModel.differential.armature = OCPParamsCrocoBase.armature

        problem = crocoddyl.ShootingProblem(
            self.x0, self._runningModelList, self._terminalModel
        )
        # Create solver + callbacks
        ocp = mim_solvers.SolverSQP(problem)

        # Merit function
        ocp.use_filter_line_search = False

        # Parameters of the solver
        ocp.termination_tolerance = OCPParamsCrocoBase.termination_tolerance
        ocp.max_qp_iters = OCPParamsCrocoBase.qp_iters
        ocp.eps_abs = OCPParamsCrocoBase.eps_abs
        ocp.eps_rel = OCPParamsCrocoBase.eps_rel

        ocp.with_callbacks = OCPParamsCrocoBase.callbacks

        result = ocp.solve(self.x_init, self.u_init, OCPParamsCrocoBase.solver_iters)

        self.ocp_results = OCPResults(
            states=ocp.xs,
            ricatti_gains=ocp.Ks,  # Not sure about this one
            feed_forward_terms=ocp.us,
        )

        return result

    @property
    def ocp_results(self) -> OCPResults:
        """Output data structure of the OCP.

        Returns:
            OCPResults: Output data structure of the OCP. It contains the states, Ricatti gains and feed-forward terms.
        """
        return self._ocp_results

    @property
    def debug_data(self) -> OCPDebugData:
        ...
