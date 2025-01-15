from os.path import dirname
import unittest
from pathlib import Path
import crocoddyl
import example_robot_data as robex
import numpy as np
import pinocchio as pin
import pickle
from agimus_controller.ocp_base_croco import OCPBaseCroco
from agimus_controller.ocp_param_base import OCPParamsBaseCroco
from agimus_controller.factory.robot_model import RobotModels, RobotModelParameters


class OCPCrocoExp3(OCPBaseCroco):
    def create_running_model_list(self):
        running_model_list = []
        for t in range(self._ocp_params.horizon_size):
            # Running cost model
            running_cost_model = crocoddyl.CostModelSum(self._state)

            ### Creation of cost terms
            # State Regularization cost
            x_residual = crocoddyl.ResidualModelState(
                self._state,
                np.concatenate(
                    (
                        pin.neutral(self._robot_models.robot_model),
                        np.zeros(self._robot_models.robot_model.nv),
                    )
                ),
            )
            x_reg_cost = crocoddyl.CostModelResidual(self._state, x_residual)
            # Control Regularization cost
            u_residual = crocoddyl.ResidualModelControl(self._state)
            u_reg_cost = crocoddyl.CostModelResidual(self._state, u_residual)
            running_cost_model.addCost("stateReg", x_reg_cost, 0.1)
            running_cost_model.addCost("ctrlRegGrav", u_reg_cost, 0.0001)
            # Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions
            running_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(
                self._state,
                self._actuation,
                running_cost_model,
            )
            running_model = crocoddyl.IntegratedActionModelEuler(
                running_DAM,
            )
            running_model.differential.armature = self._robot_models.armature

            running_model_list.append(running_model)
        return running_model_list

    def create_terminal_model(self):
        # Terminal cost models
        terminal_cost_model = crocoddyl.CostModelSum(self._state)

        ### Creation of cost terms
        # State Regularization cost
        x_residual = crocoddyl.ResidualModelState(
            self._state,
            np.concatenate(
                (
                    pin.neutral(self._robot_models.robot_model),
                    np.zeros(self._robot_models.robot_model.nv),
                )
            ),
        )
        x_reg_cost = crocoddyl.CostModelResidual(self._state, x_residual)
        terminal_cost_model.addCost("stateReg", x_reg_cost, 0.1)

        # Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions
        terminal_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            self._state,
            self._actuation,
            terminal_cost_model,
        )

        terminal_model = crocoddyl.IntegratedActionModelEuler(terminal_DAM, 0.0)
        terminal_model.differential.armature = self._robot_models.armature
        return terminal_model

    def set_reference_trajectory(self, horizon_size):
        ### Not implemented in this OCP example.
        return None

    def update_crocoddyl_problem(self, x0, trajectory_points_list):
        ### Not implemented in this OCP example.
        return None


### Loading the robot
robot = robex.load("panda")
urdf_path = Path(robot.urdf)
srdf_path = Path(robot.urdf.replace("urdf", "srdf"))
urdf_meshes_dir = urdf_path.parent.parent.parent.parent.parent
free_flyer = False
locked_joint_names = ["panda_finger_joint1", "panda_finger_joint2"]
reduced_nq = robot.model.nq - len(locked_joint_names)
full_q0 = np.zeros(robot.model.nq)
q0 = np.zeros(reduced_nq)
armature = np.full(reduced_nq, 0.1)
# Store shared initial parameters
params = RobotModelParameters(
    q0=q0,
    full_q0=full_q0,
    free_flyer=free_flyer,
    locked_joint_names=locked_joint_names,
    urdf_path=urdf_path,
    srdf_path=srdf_path,
    urdf_meshes_dir=urdf_meshes_dir,
    collision_as_capsule=True,
    self_collision=True,
    armature=armature,
)

robot_models = RobotModels(params)
robot_model = robot_models.robot_model
collision_model = robot_models.collision_model

# Set mock parameters
ocp_params = OCPParamsBaseCroco(
    dt=0.1, horizon_size=10, solver_iters=100, callbacks=True
)
