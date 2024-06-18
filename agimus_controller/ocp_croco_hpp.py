import crocoddyl
import pinocchio as pin
import numpy as np
import example_robot_data
import mim_solvers


class OCPCrocoHPP:
    def __init__(self, robot_name: str):
        """Class to define the OCP linked witha HPP generated trajectory.

        Args:
            robot_name (str): Name of the robot, can be chosen between 'ur3', 'ur5', 'ur10' and 'panda'.
            The collision avoidance OCP only works for the panda.

        Raises:
            Exception: Unkown robot.
        """
        
        # Weights of the different costs
        self._weight_x_reg = 1e-1
        self._weight_u_reg = 1e-4
        self._weight_ee_placement = 1e6
        self._weight_vel_reg = 0
        
        # Using the constraints ?
        self.use_constraints = False
        
        # Loading the robot
        self.robot = example_robot_data.load(robot_name)
        
        # Determining the last joint of the robot depending on their name
        if robot_name in ["ur3", "ur5", "ur10"]:
            self.last_joint_name = "wrist_3_joint"
            print("Collision avoidance is not taken into account for the URs.")

        # The panda 
        elif robot_name == "panda":
            self.last_joint_name = "panda_joint7"
            locked_joints = [
                self.robot.model.getJointId("panda_finger_joint1"),
                self.robot.model.getJointId("panda_finger_joint2"),
            ]
            robot_model_reduced = pin.buildReducedModel(
                self.robot.model, locked_joints, self.robot.q0
            )
            self.robot.model = robot_model_reduced
        else:
            raise Exception("Unkown robot!")
        self.robot_data = self.robot.model.createData()
        self.state = crocoddyl.StateMultibody(self.robot.model)
        self.actuation = crocoddyl.ActuationModelFull(self.state)
        self.DT = 1e-2
        self.nq = self.robot.nq
        self.nv = self.robot.nv

        self.x_plan = None
        self.a_plan = None
        self.u_ref = None
        self.T = None
        self.running_models = None
        self.terminal_model = None
        self.solver = None

    def get_uref(self, x_plan, a_plan):
        """Return the reference of control u_ref that compensates gravity."""
        u_ref = np.zeros([x_plan.shape[0] - 1, self.nv])
        """
        for x in self.hpp_paths[path_idx].x_plan:
            pin.computeGeneralizedGravity(
                self.robot.model,
                self.robot_data,
                x[: self.nq],
            )
            u_ref.append(self.robot_data.g.copy())"""
        for idx in range(x_plan.shape[0] - 1):
            x = x_plan[idx, :]
            a = a_plan[idx, :]
            tau = self.get_inverse_dynamic_control(x, a)
            u_ref[idx, :] = tau[: self.nq]
        return u_ref

    def set_weights(
        self, weight_ee_placement, weight_x_reg, weight_u_reg, weight_vel_reg
    ):
        """Set costs of the ocp."""
        self._weight_ee_placement = weight_ee_placement
        self._weight_x_reg = weight_x_reg
        self._weight_u_reg = weight_u_reg
        self._weight_vel_reg = weight_vel_reg

    def set_models(self, x_plan, a_plan):
        """Set running models and terminal model for the ocp."""
        self.x_plan = x_plan
        self.a_plan = a_plan
        self.T = x_plan.shape[0]
        self.u_ref = self.get_uref(x_plan, a_plan)
        goal_placement_residual = self.get_placement_residual(
            self.x_plan[-1, : self.nq]
        )
        self.set_running_models()
        self.set_terminal_model(goal_placement_residual)

    def set_running_models(self):
        """Set running models based on state and acceleration reference trajectory."""

        running_models = []
        for idx in range(self.T - 1):
            running_cost_model = crocoddyl.CostModelSum(self.state)
            x_ref = self.x_plan[idx, :]
            x_residual = self.get_state_residual(x_ref)
            u_residual = self.get_control_residual(self.u_ref[idx, :])
            frame_velocity_residual = self.get_velocity_residual(self.last_joint_name)
            placemment_residual = self.get_placement_residual(x_ref[: self.nq])
            running_cost_model.addCost("xReg", x_residual, self._weight_x_reg)
            running_cost_model.addCost("uReg", u_residual, self._weight_u_reg)
            running_cost_model.addCost(
                "velReg", frame_velocity_residual, self._weight_vel_reg
            )
            running_cost_model.addCost(
                "gripperPose", placemment_residual, 0
            )  # useful for mpc to reset ocp
            running_models.append(
                crocoddyl.IntegratedActionModelEuler(
                    crocoddyl.DifferentialActionModelFreeFwdDynamics(
                        self.state, self.actuation, running_cost_model
                    ),
                    self.DT,
                )
            )
        self.running_models = running_models

    def set_terminal_model(self, goal_placement_residual):
        """Set terminal model."""
        if self.use_constraints:
            last_model = self.get_terminal_model_with_constraints(
                goal_placement_residual, self.x_plan[-1, :], self.u_ref[-1, :]
            )
        else:
            last_model = self.get_terminal_model_without_constraints(
                goal_placement_residual, self.x_plan[-1, :], self.u_ref[-1, :]
            )
        self.terminal_model = last_model

    def get_terminal_model_without_constraints(
        self, goal_placement_residual, x_ref, u_ref
    ):
        """Return last model without constraints."""
        running_cost_model = crocoddyl.CostModelSum(self.state)
        running_cost_model.addCost(
            "gripperPose", goal_placement_residual, self._weight_ee_placement
        )
        vel_cost = self.get_velocity_residual(self.last_joint_name)
        if np.linalg.norm(x_ref[self.nq :]) < 1e-9:
            running_cost_model.addCost(
                "velReg", vel_cost, self._weight_ee_placement
            )
        else:
            running_cost_model.addCost("velReg", vel_cost, 0)
        x_residual = self.get_state_residual(x_ref)
        running_cost_model.addCost("xReg", x_residual, 0)

        u_reg_cost = self.get_control_residual(u_ref)
        running_cost_model.addCost("uReg", u_reg_cost, 0)
        return crocoddyl.IntegratedActionModelEuler(
            crocoddyl.DifferentialActionModelFreeFwdDynamics(
                self.state, self.actuation, running_cost_model
            ),
            self.DT,
        )

    def get_terminal_model_with_constraints(self, placement_residual, x_ref, u_ref):
        """Return terminal model with constraints for mim_solvers."""
        running_cost_model = crocoddyl.CostModelSum(self.state)
        constraints = crocoddyl.ConstraintModelManager(self.state, self.nq)
        x_residual = self.get_state_residual(x_ref)
        u_reg_cost = self.get_control_residual(u_ref)
        vel_cost = self.get_velocity_residual(self.last_joint_name)
        running_cost_model.addCost("xReg", x_residual, 0)
        running_cost_model.addCost("velReg", vel_cost, 0)
        running_cost_model.addCost("gripperPose", placement_residual, 0)
        running_cost_model.addCost("uReg", u_reg_cost, 0)

        # add torque constraint

        torque_residual = crocoddyl.ResidualModelJointEffort(
            self.state,
            self.actuation,
            np.array([0] * 6),  # np.array([0] * 6)
        )
        torque_constraint = crocoddyl.ConstraintModelResidual(
            self.state,
            torque_residual,
            np.array([-87] * 4 + [-12] * 3),
            np.array([87] * 4 + [12] * 3),
        )
        constraints.addConstraint("JointsEfforts", torque_constraint)

        return crocoddyl.IntegratedActionModelEuler(
            crocoddyl.DifferentialActionModelFreeFwdDynamics(
                self.state, self.actuation, running_cost_model, constraints
            ),
            self.DT,
        )

    def get_placement_residual(self, q):
        """Return placement residual to the last position of the sub path."""
        target = self.robot.placement(q, self.nq).copy()
        return crocoddyl.CostModelResidual(
            self.state,
            crocoddyl.ResidualModelFramePlacement(
                self.state, self.robot.model.getFrameId(self.last_joint_name), target
            ),
        )

    def get_velocity_residual(self, joint_name):
        """Return velocity residual of desired joint."""
        vref = pin.Motion.Zero()
        return crocoddyl.CostModelResidual(
            self.state,
            crocoddyl.ResidualModelFrameVelocity(
                self.state,
                self.robot.model.getFrameId(joint_name),
                vref,
                pin.WORLD,
            ),
        )

    def get_control_residual(self, uref):
        """Return control residual with uref the control reference."""
        return crocoddyl.CostModelResidual(
            self.state, crocoddyl.ResidualModelControl(self.state, uref)
        )

    def get_state_residual(self, xref):
        """Return state residual with xref the state reference."""
        return crocoddyl.CostModelResidual(
            self.state,  # x_reg_weights,
            crocoddyl.ResidualModelState(self.state, xref, self.actuation.nu),
        )

    def get_xlimit_residual(self):
        """Return state limit residual."""
        return crocoddyl.CostModelResidual(
            self.state,
            crocoddyl.ActivationModelQuadraticBarrier(
                crocoddyl.ActivationBounds(self.state.lb, self.state.ub)
            ),
            crocoddyl.ResidualModelState(
                self.state,
                np.array([0] * (self.nq + self.nv)),
                self.actuation.nu,
            ),
        )

    def get_translation_residual(self):
        """Return translation residual to the last position of the sub path."""
        q_final = self.x_plan[-1, : self.nq]
        target = self.robot.placement(q_final, self.nq)
        return crocoddyl.ResidualModelFrameTranslation(
            self.state,
            self.robot.model.getFrameId(self.last_joint_name),
            target.translation,
        )

    def get_inverse_dynamic_control(self, x, a):
        """Return inverse dynamic control for a given state and acceleration."""
        return pin.rnea(
            self.robot.model, self.robot.data, x[: self.nq], x[self.nq :], a
        ).copy()

    def update_cost(self, model, new_model, cost_name, update_weight=True):
        """Update model's cost reference and weight by copying new_model's cost."""
        model.differential.costs.costs[cost_name].cost.residual.reference = (
            new_model.differential.costs.costs[cost_name].cost.residual.reference.copy()
        )
        if update_weight:
            new_weight = new_model.differential.costs.costs[cost_name].weight
            model.differential.costs.costs[cost_name].weight = new_weight
        if model.differential.costs.costs[cost_name].weight == 0:
            model.differential.costs.changeCostStatus(cost_name, False)
        else:
            model.differential.costs.changeCostStatus(cost_name, True)

    def update_model(self, model, new_model, update_weight):
        """update model's costs by copying new_model's costs."""
        self.update_cost(model, new_model, "xReg", update_weight)
        self.update_cost(model, new_model, "gripperPose", update_weight)
        self.update_cost(model, new_model, "velReg", update_weight)
        self.update_cost(model, new_model, "uReg", update_weight)

    def reset_ocp(self, x, x_ref, u_ref):
        """Reset ocp problem using next reference in state and control."""
        self.solver.problem.x0 = x
        runningModels = list(self.solver.problem.runningModels)
        for node_idx in range(len(runningModels) - 1):
            self.update_model(
                runningModels[node_idx], runningModels[node_idx + 1], True
            )
        self.update_model(runningModels[-1], self.solver.problem.terminalModel, False)
        if self.use_constraints:
            terminal_model = self.get_terminal_model_with_constraints(
                self.get_placement_residual(x_ref[: self.nq]), x_ref, u_ref
            )
        else:
            terminal_model = self.get_terminal_model_without_constraints(
                self.get_placement_residual(x_ref[: self.nq]), x_ref, u_ref
            )
        self.update_model(self.solver.problem.terminalModel, terminal_model, True)

    def build_ocp_from_plannif(self, x_plan, a_plan, x0):
        """Set models based on state and acceleration planning, create crocoddyl problem from it."""
        self.set_models(x_plan, a_plan)
        return crocoddyl.ShootingProblem(x0, self.running_models, self.terminal_model)

    def run_solver(self, problem, xs_init, us_init, max_iter, set_callback=False):
        """
        Run FDDP or CSQP solver
        problem : crocoddyl ocp problem.
        xs_init : xs warm start.
        us_init : us warm start.
        max_iter : max number of iteration for the solver
        set_callback : activate solver callback
        """
        # Creating the solver for this OC problem, defining a logger
        if self.use_constraints:
            solver = mim_solvers.SolverCSQP(problem)
            solver.use_filter_line_search = True
            solver.termination_tolerance = 1e-3
            solver.max_qp_iters = 100
        else:
            solver = crocoddyl.SolverFDDP(problem)
            solver.use_filter_line_search = True
            solver.termination_tolerance = 1e-3
        if set_callback:
            solver.setCallbacks([crocoddyl.CallbackVerbose()])
        solver.solve(xs_init, us_init, max_iter)
        self.solver = solver
