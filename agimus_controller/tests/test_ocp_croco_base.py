from os.path import dirname
import unittest
from unittest.mock import MagicMock
from pathlib import Path
import crocoddyl
import example_robot_data as robex
import numpy as np
import pinocchio as pin

from agimus_controller.ocp_base_croco import OCPBaseCroco
from agimus_controller.ocp_param_base import OCPParamsBaseCroco
from agimus_controller.factory.robot_model import RobotModels, RobotModelParameters


class TestOCPBaseCroco(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # Mock the RobotModelFactory and OCPParamsCrocoBase
        robot = robex.load("panda")
        urdf_path = Path(robot.urdf)
        srdf_path = Path(robot.urdf.replace("urdf", "srdf"))
        urdf_meshes_dir = urdf_path.parent.parent.parent.parent.parent
        free_flyer = False
        q0 = np.zeros(robot.model.nq)
        armature = np.full(robot.model.nq, 0.1)

        # Store shared initial parameters
        self.params = RobotModelParameters(
            q0=q0,
            free_flyer=free_flyer,
            locked_joint_names=["panda_joint1", "panda_joint2"],
            urdf_path=urdf_path,
            srdf_path=srdf_path,
            urdf_meshes_dir=urdf_meshes_dir,
            collision_as_capsule=True,
            self_collision=True,
            armature=armature,
        )

        self.robot_models = RobotModels(self.params)
        self.robot_model = self.robot_models.robot_model
        self.collision_model = self.robot_models.collision_model

        self.ocp_params = OCPParamsBaseCroco(
            dt=0.1, horizon_size=10, solver_iters=100, callbacks=True
        )

        # Create a concrete implementation of OCPBaseCroco
        class TestOCPCroco(OCPBaseCroco):
            def create_running_model_list(self):
                return None

            def create_terminal_model(self):
                return None

            def update_crocoddyl_problem(self, x0, trajectory_points_list):
                return None

            def set_reference_horizon(self, horizon_size):
                return None

        self.ocp = TestOCPCroco(self.robot_models, self.ocp_params)

    def test_horizon_size(self):
        """Test the horizon_size property."""
        self.assertEqual(self.ocp.horizon_size, self.ocp_params.horizon_size)

    def test_dt(self):
        """Test the dt property."""
        self.assertAlmostEqual(self.ocp.dt, self.ocp_params.dt)


class TestSimpleOCPCroco(unittest.TestCase):
    class TestOCPCroco(OCPBaseCroco):
        def create_running_model_list(self):
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

            running_model_list = [running_model] * (self._ocp_params.horizon_size - 1)
            return running_model_list

        def create_terminal_model(self):
            # Terminal cost models
            terminal_cost_model = crocoddyl.CostModelSum(self._state)

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

        def update_crocoddyl_problem(self, x0, trajectory_points_list):
            ### Not implemented in this OCP example.
            return None

    @classmethod
    def setUpClass(self):
        # Mock the RobotModelFactory and OCPParamsCrocoBase
        robot = robex.load("panda")
        urdf_path = Path(robot.urdf)
        srdf_path = Path(robot.urdf.replace("urdf", "srdf"))
        urdf_meshes_dir = urdf_path.parent.parent.parent.parent.parent
        free_flyer = False
        q0 = np.zeros(robot.model.nq)

        # Store shared initial parameters
        self.params = RobotModelParameters(
            q0=q0,
            free_flyer=free_flyer,
            locked_joint_names=["panda_joint1", "panda_joint2"],
            urdf_path=urdf_path,
            srdf_path=srdf_path,
            urdf_meshes_dir=urdf_meshes_dir,
            collision_as_capsule=True,
            self_collision=True,
            armature=np.linspace(0.1, 0.9, robot.model.nq),
        )

        self.robot_models = RobotModels(self.params)
        self.robot_model = self.robot_models.robot_model
        self.collision_model = self.robot_models.collision_model

        # Set mock parameters
        self.ocp_params = OCPParamsBaseCroco(
            dt=0.1, horizon_size=10, solver_iters=100, callbacks=True
        )
        self.state_reg = np.concatenate(
            (pin.neutral(self.robot_model), np.zeros(self.robot_model.nv))
        )
        self.state_warmstart = [np.zeros(self.robot_model.nq + self.robot_model.nv)] * (
            self.ocp_params.horizon_size - 1
        )  # The first state is the current state
        self.control_warmstart = [np.zeros(self.robot_model.nq)] * (
            self.ocp_params.horizon_size - 1
        )
        # Create a concrete implementation of OCPBaseCroco
        self.ocp = self.TestOCPCroco(self.robot_models, self.ocp_params)
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
                state,
                self.ocp.ocp_results.states.tolist()[iter],
                err_msg="States are not equal",
            )

        # Checking the ricatti gains
        for iter, gain in enumerate(results[1]):
            np.testing.assert_array_almost_equal(
                gain,
                self.ocp.ocp_results.ricatti_gains.tolist()[iter],
                err_msg="Ricatti gains are not equal",
            )

        # Checking the feed forward terms
        for iter, term in enumerate(results[2]):
            np.testing.assert_array_almost_equal(
                term,
                self.ocp.ocp_results.feed_forward_terms.tolist()[iter],
                err_msg="Feed forward term are not equal",
            )


if __name__ == "__main__":
    unittest.main()
