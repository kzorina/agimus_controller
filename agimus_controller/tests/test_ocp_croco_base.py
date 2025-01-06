import unittest
from unittest.mock import MagicMock

import crocoddyl
import example_robot_data as robex
import numpy as np
import pinocchio as pin

from agimus_controller.ocp_base_croco import OCPBaseCroco
from agimus_controller.ocp_param_base import OCPParamsBaseCroco
from agimus_controller.factory.robot_model import RobotModelFactory


class TestOCPBaseCroco(unittest.TestCase):
    def setUp(self):
        # Mock the RobotModelFactory and OCPParamsCrocoBase
        self.mock_robot_model_factory = RobotModelFactory()

        mock_robot = robex.load("panda")
        mock_robot_model = mock_robot.model
        mock_collision_model = mock_robot.collision_model
        mock_armature = np.array([])

        self.mock_robot_model_factory._rmodel = mock_robot_model
        self.mock_robot_model_factory._complete_collision_model = mock_collision_model
        self.mock_robot_model_factory._armature = mock_armature

        self.mock_ocp_params = MagicMock(spec=OCPParamsBaseCroco)

        # Set mock parameters
        self.mock_ocp_params.T = 10
        self.mock_ocp_params.dt = 0.1
        self.mock_ocp_params.use_filter_line_search = True
        self.mock_ocp_params.termination_tolerance = 1e-6
        self.mock_ocp_params.qp_iters = 10
        self.mock_ocp_params.eps_abs = 1e-8
        self.mock_ocp_params.eps_rel = 1e-6
        self.mock_ocp_params.callbacks = True
        self.mock_ocp_params.solver_iters = 100

        # Create a concrete implementation of OCPBaseCroco
        class TestOCPCroco(OCPBaseCroco):
            @property
            def running_model_list(self):
                return None

            @property
            def terminal_model(self):
                return None

            def set_reference_horizon(self, horizon_size):
                return None

        self.ocp = TestOCPCroco(self.mock_robot_model_factory, self.mock_ocp_params)

    def test_horizon_size(self):
        """Test the horizon_size property."""
        self.assertEqual(self.ocp.horizon_size, self.mock_ocp_params.T)

    def test_dt(self):
        """Test the dt property."""
        self.assertAlmostEqual(self.ocp.dt, self.mock_ocp_params.dt)


class TestSimpleOCPCroco(unittest.TestCase):
    class TestOCPCroco(OCPBaseCroco):
        @property
        def running_model_list(self):
            # Running cost model
            running_cost_model = crocoddyl.CostModelSum(self._state)

            ### Creation of cost terms
            # State Regularization cost
            x_residual = crocoddyl.ResidualModelState(
                self._state,
                np.concatenate((pin.neutral(self._rmodel), np.zeros(self._rmodel.nv))),
            )
            x_reg_cost = crocoddyl.CostModelResidual(self._state, x_residual)

            # Control Regularization cost
            u_residual = crocoddyl.ResidualModelControl(self._state)
            u_reg_cost = crocoddyl.CostModelResidual(self._state, u_residual)

            # End effector frame cost
            frame_placement_residual = crocoddyl.ResidualModelFramePlacement(
                self._state,
                self._rmodel.getFrameId("panda_hand_tcp"),
                pin.SE3(np.eye(3), np.array([1.0, 1.0, 1.0])),
            )

            goal_tracking_cost = crocoddyl.CostModelResidual(
                self._state, frame_placement_residual
            )
            running_cost_model.addCost("stateReg", x_reg_cost, 0.1)
            running_cost_model.addCost("ctrlRegGrav", u_reg_cost, 0.0001)
            running_cost_model.addCost("gripperPoseRM", goal_tracking_cost, 1.0)
            # Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions
            running_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(
                self._state,
                self._actuation,
                running_cost_model,
            )
            running_model = crocoddyl.IntegratedActionModelEuler(
                running_DAM,
            )
            running_model.differential.armature = self._robot_model.armature

            running_model_list = [running_model] * (self._ocp_params.T - 1)
            return running_model_list

        @property
        def terminal_model(self):
            # Terminal cost models
            terminal_cost_model = crocoddyl.CostModelSum(self._state)

            ### Creation of cost terms
           ### Creation of cost terms
            # State Regularization cost
            x_residual = crocoddyl.ResidualModelState(
                self._state,
                np.concatenate((pin.neutral(self._rmodel), np.zeros(self._rmodel.nv))),
            )
            x_reg_cost = crocoddyl.CostModelResidual(self._state, x_residual)

            # End effector frame cost
            frame_placement_residual = crocoddyl.ResidualModelFramePlacement(
                self._state,
                self._rmodel.getFrameId("panda_hand_tcp"),
                pin.SE3(np.eye(3), np.array([1.0, 1.0, 1.0])),
            )

            goal_tracking_cost = crocoddyl.CostModelResidual(
                self._state, frame_placement_residual
            )
            terminal_cost_model.addCost("stateReg", x_reg_cost, 0.1)
            terminal_cost_model.addCost("gripperPose", goal_tracking_cost, 50)

            # Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions
            terminal_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(
                self._state,
                self._actuation,
                terminal_cost_model,
            )

            terminal_model = crocoddyl.IntegratedActionModelEuler(terminal_DAM, 0.0)
            terminal_model.differential.armature = self._robot_model.armature

            return terminal_model

        def set_reference_horizon(self, horizon_size):
            ### Not implemented in this OCP example.
            return None

    def setUp(self):
        # Mock the RobotModelFactory and OCPParamsCrocoBase
        self.robot_model_factory = RobotModelFactory()

        robot = robex.load("panda")
        robot_model = robot.model
        collision_model = robot.collision_model
        armature = np.full(robot_model.nq, 0.1)

        self.robot_model_factory._rmodel = robot_model
        self.robot_model_factory._complete_collision_model = collision_model
        self.robot_model_factory.armature = armature

        # Set mock parameters
        self.ocp_params = OCPParamsBaseCroco(
            dt=0.1, T=10, solver_iters=100, callbacks=True
        )
        self.state_reg = np.concatenate(
            (pin.neutral(robot_model), np.zeros(robot_model.nv))
        )
        self.state_warmstart = [np.zeros(robot_model.nq + robot_model.nv)] * (
            self.ocp_params.T - 1
        )  # The first state is the current state
        self.control_warmstart = [np.zeros(robot_model.nq)] * (self.ocp_params.T - 1)
        # Create a concrete implementation of OCPBaseCroco
        self.ocp = self.TestOCPCroco(self.robot_model_factory, self.ocp_params)
        self.ocp.solve(self.state_reg, self.state_warmstart, self.control_warmstart)
        # self.save_results()

    def save_results(self):
        results = np.array(
            [
                self.ocp.ocp_results.states.tolist(),
                self.ocp.ocp_results.ricatti_gains.tolist(),
                self.ocp.ocp_results.feed_forward_terms.tolist(),
            ],
            dtype=object,  # Ensure the array is dtype=object before saving
        )
        np.save(
            "ressources/simple_ocp_croco_results.npy",
            results,
        )

    def test_check_results(self):
        results = np.load("ressources/simple_ocp_croco_results.npy", allow_pickle=True)
        # Checking the states
        for iter, state in enumerate(results[0]):
            np.testing.assert_array_almost_equal(
                state, self.ocp.ocp_results.states.tolist()[iter], err_msg="States are not equal"
            )

        # Checking the ricatti gains
        for iter, gain in enumerate(results[1]):
            np.testing.assert_array_almost_equal(
                gain, self.ocp.ocp_results.ricatti_gains.tolist()[iter], err_msg="Ricatti gains are not equal"
            )

        # Checking the feed forward terms
        for iter, term in enumerate(results[2]):
            np.testing.assert_array_almost_equal(
                term, self.ocp.ocp_results.feed_forward_terms.tolist()[iter], err_msg="Feed forward term are not equal"
            )


if __name__ == "__main__":
    unittest.main()
