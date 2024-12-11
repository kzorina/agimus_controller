from abc import ABC, abstractmethod
from typing import List
import numpy as np
import pinocchio as pin
import crocoddyl

from agimus_controller.mpc_data import OCPResults, OCPDebugData
from agimus_controller.trajectory import WeightedTrajectoryPoint
from agimus_controller.ocp_base import OCPBase
from agimus_controller.OCPCrocoBase import OCPCrocoBase

class OCPCrocoBase(OCPBase):
    def __init__(self, rmodel: pin.Model, cmodel: pin.GeometryModel, OCPParams: OCPParamsBase) -> None:
        """Creates an instance of the OCPCrocoBase class. This is an example of an OCP class that inherits from OCPBase, 
        for a simple goal reaching task. The class is used to solve the Optimal Control Problem (OCP) using the Crocoddyl library.

        Args:
            rmodel (pin.Model): Robot model.
            cmodel (pin.GeometryModel): Collision Model of the robot.
            OCPParams (OCPParamsBase): Input data structure of the OCP.
        """
        self._rmodel = rmodel
        self._cmodel = cmodel
        self._OCPParams = OCPParams
        
        
    @abstractmethod
    @property
    def horizon_size() -> int:
        """Number of time steps in the horizon."""
        return self._OCPParams.T

    @abstractmethod
    @property
    def dt() -> float64:
        """Integration step of the OCP."""
        return self._OCPParams.dt
    
    @abstractmethod
    @property
    def x0() -> np.ndarray:
        """Initial state of the robot."""
        return self._OCPParams.WeightedTrajectoryPoints[0].point.robot_configuration

    @abstractmethod
    def solve(
        self, x_init: List[np.ndarray], u_init: List[np.ndarray]
    ) -> bool:
        """Solves the OCP. Returns True if the OCP was solved successfully, False otherwise.

        Args:
            x_init (List[np.ndarray]): List of the states for the initial trajectory.
            u_init (List[np.ndarray]): List of the controls for the initial trajectory.

        Returns:
            bool: True if the OCP was solved successfully, False otherwise.
        """
        ### Creation of the state and actuation models        
        # Stat and actuation model
        self._state = crocoddyl.StateMultibody(self._rmodel)
        self._actuation = crocoddyl.ActuationModelFull(self._state)

        # Running & terminal cost models
        self._runningCostModel = crocoddyl.CostModelSum(self._state)
        self._terminalCostModel = crocoddyl.CostModelSum(self._state)

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
            self._OCPParams.p_target,
        )

        goalTrackingCost = crocoddyl.CostModelResidual(
            self._state, frameTranslationResidual
        )
        

    @abstractmethod
    @property
    def ocp_results(self) -> OCPResults:
        ...

    @abstractmethod
    @property
    def debug_data(self) -> OCPDebugData:
        ...
