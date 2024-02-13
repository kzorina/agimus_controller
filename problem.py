from hpp.corbaserver import wrap_delete
import crocoddyl
import pinocchio as pin
import numpy as np
import example_robot_data


class SubPath:
    def __init__(self, path):
        self.path = path
        self.T = int(20 * self.path.length())
        self.x_plan = self.set_xplan()
        self.u_ref = None
        self.running_models = None
        self.terminal_model = None

    def set_xplan(self):
        """Set x_plan, the state trajectory of hpp."""
        x_plan = []
        total_time = self.path.length()
        if self.T == 0:
            return x_plan
        for iter in range(self.T + 1):
            iter_time = total_time * iter / self.T
            q_t = np.array(self.path.call(iter_time)[0][:6])
            v_t = np.array(self.path.derivative(iter_time, 1)[:6])
            x_plan.append(np.concatenate([q_t, v_t]))
        return x_plan


class Problem:
    def __init__(self, ps, robot_name):
        self.DT = 1e-3  # integration step for crocoddyl
        self.x_cost = 1e-1
        self.u_cost = 1e-4
        self.grip_cost = 1e6
        self.null_speed_path_idxs = [1, 4]
        self.p = wrap_delete(ps.client.basic.problem.getPath(2))
        self.robot = example_robot_data.load(robot_name)
        self.robot_data = self.robot.model.createData()
        self.set_costs(1e1, 1e-3, 1e-5)
        self.hpp_paths = []
        self.DT = 1e-3
        self.nq = self.robot.nq
        self.nv = self.robot.nv
        for i in range(self.p.numberPaths()):
            new_path = SubPath(self.p.pathAtRank(i))
            if new_path.T > 0:
                self.hpp_paths.append(new_path)
        self.nb_paths = len(self.hpp_paths)
        self.ddp = None

    def get_uref(self, path_idx):
        """Return the reference of control u_ref that compensates gravity."""
        u_ref = []
        for x in self.hpp_paths[path_idx].x_plan:
            pin.computeGeneralizedGravity(
                self.robot.model,
                self.robot_data,
                x[: self.nq],
            )
            u_ref.append(self.robot_data.g)
        return u_ref

    def set_costs(self, grip_cost, x_cost, u_cost):
        """Set costs of the ddp problem."""
        self.x_cost = x_cost
        self.u_cost = u_cost
        self.grip_cost = grip_cost

    def set_models(self, terminal_paths_idxs):
        """Set running models and terminal model of the ddp problem."""
        state = crocoddyl.StateMultibody(self.robot.model)
        actuation_model = crocoddyl.ActuationModelFull(state)
        goal_tracking_costs = self.get_tracking_costs(state)
        for path_idx in range(self.nb_paths):
            self.hpp_paths[path_idx].u_ref = self.get_uref(path_idx)
            self.set_sub_path_running_models(
                path_idx,
                terminal_paths_idxs,
                state,
                actuation_model,
                goal_tracking_costs[path_idx],
            )

        self.set_terminal_models(
            terminal_paths_idxs, state, actuation_model, goal_tracking_costs
        )

    def get_tracking_costs(self, state):
        """Return vector of tracking costs for each sub path."""
        goal_tracking_costs = []
        for path_idx in range(self.nb_paths):
            q_final = self.hpp_paths[path_idx].x_plan[-1][: self.nq]
            target = self.robot.placement(q_final, self.nq)
            goal_tracking_costs.append(
                crocoddyl.CostModelResidual(
                    state,
                    crocoddyl.ResidualModelFramePlacement(
                        state, self.robot.model.getFrameId("wrist_3_joint"), target
                    ),
                )
            )
        """goalTrackingCost = crocoddyl.CostModelResidual(
            state,
            crocoddyl.ResidualModelFrameTranslation(
                state, self.robot.model.getFrameId("wrist_3_joint"), target.translation
            ),
        )"""
        return goal_tracking_costs

    def set_sub_path_running_models(
        self, path_idx, terminal_paths_idxs, state, actuation_model, goalTrackingCost
    ):
        """Set running models for one sub path of the ddp problem."""

        running_models = []
        xRegWeights = crocoddyl.ActivationModelWeightedQuad(
            np.array([10] * self.nq + [1] * self.nv)
        )

        for idx in range(self.hpp_paths[path_idx].T):
            running_cost_model = crocoddyl.CostModelSum(state)
            x_reg_cost = crocoddyl.CostModelResidual(
                state,
                xRegWeights,
                crocoddyl.ResidualModelState(
                    state, self.hpp_paths[path_idx].x_plan[idx]
                ),
            )
            u_reg_cost = crocoddyl.CostModelResidual(
                state,
                crocoddyl.ResidualModelControl(
                    state, self.hpp_paths[path_idx].u_ref[idx]
                ),
            )
            running_cost_model.addCost("xReg", x_reg_cost, self.x_cost)
            running_cost_model.addCost("uReg", u_reg_cost, self.u_cost)
            running_models.append(
                crocoddyl.IntegratedActionModelEuler(
                    crocoddyl.DifferentialActionModelFreeFwdDynamics(
                        state, actuation_model, running_cost_model
                    ),
                    self.DT,
                )
            )
        if path_idx not in terminal_paths_idxs:
            running_models = self.set_sub_path_last_running_model(
                state, actuation_model, goalTrackingCost, running_models, path_idx
            )

        self.hpp_paths[path_idx].running_models = running_models

    def set_sub_path_last_running_model(
        self, state, actuation_model, goalTrackingCost, running_models, path_idx
    ):
        """Set last running model for one sub path of the ddp problem."""
        running_cost_model = crocoddyl.CostModelSum(state)
        running_cost_model.addCost("gripperPose", goalTrackingCost, self.grip_cost)
        if path_idx in self.null_speed_path_idxs:
            vref = pin.Motion.Zero()
            for joint_name in self.robot.model.names:
                vel_cost = crocoddyl.CostModelResidual(
                    state,
                    crocoddyl.ResidualModelFrameVelocity(
                        state, self.robot.model.getFrameId(joint_name), vref, pin.LOCAL
                    ),
                )
                running_cost_model.addCost(
                    f"vel_{joint_name}", vel_cost, self.grip_cost
                )
        running_models.append(
            crocoddyl.IntegratedActionModelEuler(
                crocoddyl.DifferentialActionModelFreeFwdDynamics(
                    state, actuation_model, running_cost_model
                ),
                self.DT,
            )
        )
        return running_models

    def set_terminal_models(
        self, terminal_paths_idxs, state, actuation_model, goal_tracking_costs
    ):
        """Set terminal models of the ddp problem."""
        for path_idx in terminal_paths_idxs:
            self.hpp_paths[path_idx].u_ref.pop()
            terminal_cost_model = crocoddyl.CostModelSum(state)
            terminal_cost_model.addCost(
                "gripperPose", goal_tracking_costs[path_idx], self.grip_cost
            )
            vref = pin.Motion.Zero()
            for joint_name in self.robot.model.names:
                vel_cost = crocoddyl.CostModelResidual(
                    state,
                    crocoddyl.ResidualModelFrameVelocity(
                        state, self.robot.model.getFrameId(joint_name), vref, pin.LOCAL
                    ),
                )
                terminal_cost_model.addCost(
                    f"vel_{joint_name}", vel_cost, self.grip_cost
                )
            terminal_model = crocoddyl.IntegratedActionModelEuler(
                crocoddyl.DifferentialActionModelFreeFwdDynamics(
                    state, actuation_model, terminal_cost_model
                )
            )
            self.hpp_paths[path_idx].terminal_model = terminal_model

    def run_ddp(self, start_idx, terminal_idx, set_callback=False):
        """
        Run ddp solver
        start_idx : hpp's sub path idx from which we start to run ddp on
        terminal_idx : hpp's sub path idx from which we end to run ddp.
        """
        # Create the problem
        x0 = self.hpp_paths[0].x_plan[0]
        final_running_model = []
        final_x_plan = []
        final_u_ref = []
        for path_idx in range(start_idx, terminal_idx + 1):
            final_running_model += self.hpp_paths[path_idx].running_models
            final_x_plan += self.hpp_paths[path_idx].x_plan
            final_u_ref += self.hpp_paths[path_idx].u_ref
        problem = crocoddyl.ShootingProblem(
            x0, final_running_model, self.hpp_paths[terminal_idx].terminal_model
        )
        # Creating the DDP solver for this OC problem, defining a logger
        ddp = crocoddyl.SolverDDP(problem)
        if set_callback:
            ddp.setCallbacks([crocoddyl.CallbackVerbose()])

        ddp.setCandidate(final_x_plan, final_u_ref, False)
        # Solving it with the DDP algorithm
        ddp.solve()
        self.ddp = ddp
