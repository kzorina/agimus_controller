import time
import numpy as np
from .ocp import OCPCrocoHPP


class TrajectoryBuffer:
    def __init__(self, model):
        self.model = model
        self.x_plan = []
        self.a_plan = []

    def add_trajectory_point(self, q, v, a=None):
        if len(q) != self.model.nq:
            raise Exception(
                f"configuration vector size : {len(q)} doesn't match model's nq : {self.model.nq}"
            )
        if len(v) != self.model.nv:
            raise Exception(
                f"velocity vector size : {len(v)} doesn't match model's nv : {self.model.nv}"
            )
        x = np.concatenate([np.array(q), np.array(v)])
        self.x_plan.append(x)
        if a is not None:
            if len(a) != self.model.nv:
                raise Exception(
                    f"acceleration vector size : {len(a)} doesn't match model's nv : {self.model.nv}"
                )
            self.a_plan.append(np.array(a))

    def get_joint_state_horizon(self):
        """Return the state reference for the horizon, state is composed of joints positions and velocities"""
        return self.x_plan

    def get_joint_acceleration_horizon(self):
        """Return the acceleration reference for the horizon, state is composed of joints positions and velocities"""
        return self.a_plan


class MPC:
    def __init__(self, x_plan, a_plan, robot_name):
        self.prob = OCPCrocoHPP(robot_name)
        self.whole_x_plan = x_plan
        self.whole_a_plan = a_plan
        self.robot = self.prob.robot
        self.nq = self.robot.nq
        self.croco_xs = None
        self.croco_us = None
        self.results = {}
        self.results["xs"] = []
        self.results["us"] = []
        self.results["max_us"] = []
        self.results["max_increase_us"] = []
        self.results["combination"] = []
        self.whole_traj_T = x_plan.shape[0]

    def get_trajectory_difference(self, configuration_traj=True):
        """Compute at each node the absolute difference in position either in cartesian or configuration space and sum it."""
        if configuration_traj:
            traj_croco = self.croco_xs[:, : self.nq]
            traj_hpp = self.prob.x_plan[:, : self.nq]
        else:
            traj_croco, traj_hpp = self.get_cartesian_trajectory()
        diffs = []
        for idx in range(len(traj_croco)):
            array_diff = np.abs(np.array(traj_croco[idx]) - np.array(traj_hpp[idx]))
            diffs.append(np.sum(array_diff))
        return sum(diffs)

    def max_increase_us(self):
        """Return control max increase"""
        increases = np.zeros([self.croco_us.shape[0] - 1, self.robot.nq])
        for joint_idx in range(self.robot.nq):
            for idx in range(self.croco_us.shape[0] - 1):
                increases[idx, joint_idx] = (
                    self.croco_us[idx + 1, joint_idx] - self.croco_us[idx, joint_idx]
                )
        return np.max(np.abs(increases)), np.unravel_index(
            np.argmax(np.abs(increases), axis=None), increases.shape
        )

    def get_cartesian_trajectory(self):
        """Return the vector of gripper pose for both trajectories found by hpp and crocoddyl."""
        pose_croco = [[] for _ in range(3)]
        pose_hpp = [[] for _ in range(3)]
        for idx in range(self.croco_xs.shape[0]):
            q = self.croco_xs[idx, : self.nq]
            pose = self.robot.placement(q, self.nq).translation
            for idx in range(3):
                pose_croco[idx].append(pose[idx])
        for idx in range(self.whole_x_plan.shape[0]):
            q = self.whole_x_plan[idx, : self.nq]
            pose = self.robot.placement(q, self.nq).translation
            for idx in range(3):
                pose_hpp[idx].append(pose[idx])
        return pose_croco, pose_hpp

    def search_best_costs(
        self, terminal_idx, use_mim=False, configuration_traj=False, is_mpc=False
    ):
        """Search costs that minimize the gap between hpp and crocoddyl trajectories."""
        self.best_combination = None
        self.best_croco_xs = None
        self.best_croco_us = None
        self.best_diff = 1e6
        grip_cost = 0
        self.prob.use_mim = use_mim
        if use_mim:
            for x_exponent in range(0, 8, 2):
                for u_exponent in range(-32, -26, 2):
                    start = time.time()
                    _, x_cost, u_cost, _, _ = self.get_cost_from_exponent(
                        0, x_exponent, u_exponent, 0, 0
                    )
                    self.try_new_costs(
                        grip_cost,
                        x_cost,
                        u_cost,
                        terminal_idx,
                        0,
                        0,
                        configuration_traj=configuration_traj,
                        vel_cost=0,
                        xlim_cost=0,
                        is_mpc=is_mpc,
                    )
                    end = time.time()
                    print("iteration duration ", end - start)
        else:
            for grip_exponent in range(25, 50, 5):
                print("grip expo ", grip_exponent)
                for x_exponent in range(0, 15, 5):
                    for u_exponent in range(-35, -25, 5):
                        # for vel_exponent in range(-6, 14, 4):
                        grip_cost, x_cost, u_cost, _, _ = self.get_cost_from_exponent(
                            grip_exponent, x_exponent, u_exponent, 0, 0
                        )
                        self.prob.set_costs(grip_cost, x_cost, u_cost, 0, 0)
                        start = time.time()
                        self.try_new_costs(
                            grip_cost,
                            x_cost,
                            u_cost,
                            terminal_idx,
                            configuration_traj=configuration_traj,
                            vel_cost=0,
                            xlim_cost=0,
                            is_mpc=is_mpc,
                        )
                        end = time.time()
                        print("iteration duration ", end - start)

        self.croco_xs = self.best_croco_xs
        self.croco_us = self.best_croco_us
        print("best diff ", self.best_diff)
        print("best combination ", self.best_combination)
        print("max torque ", np.max(np.abs(self.croco_us)))

    def get_cost_from_exponent(
        self, grip_exponent, x_exponent, u_exponent, vel_exponent=0, xlim_exponent=0
    ):
        return (
            10 ** (grip_exponent / 10),
            10 ** (x_exponent / 10),
            10 ** (u_exponent / 10),
            10 ** (vel_exponent / 10),
            10 ** (xlim_exponent / 10),
        )

    def try_new_costs(
        self,
        grip_cost,
        x_cost,
        u_cost,
        terminal_idx,
        vel_cost=0,
        xlim_cost=0,
        configuration_traj=False,
        is_mpc=False,
    ):
        """Set problem, run solver, add result in dict and check if we found a better solution."""
        if is_mpc:
            print("doing mpc")
            self.do_mpc(terminal_idx, 100)
        else:
            print("doing ocp")
            self.set_problem_run_solver(
                terminal_idx,
            )
        max_us = np.max(np.abs(self.croco_us))
        max_increase_us, _ = self.max_increase_us()
        self.results["xs"].append(self.croco_xs)
        self.results["us"].append(self.croco_us)
        self.results["max_us"].append(max_us)
        self.results["max_increase_us"].append(max_increase_us)
        self.results["combination"].append(
            [grip_cost, x_cost, u_cost, vel_cost, xlim_cost]
        )
        diff = self.get_trajectory_difference(terminal_idx, configuration_traj)
        if diff < self.best_diff and max_us < 100 and max_increase_us < 50:
            self.best_combination = [grip_cost, x_cost, u_cost, vel_cost, xlim_cost]
            self.best_diff = diff
            self.best_croco_xs = self.croco_xs
            self.best_croco_us = self.croco_us

    def set_problem_run_solver(self, terminal_idx):
        """Set OCP problem with new costs then run solver."""

        self.prob.set_models()
        problem = self.prob.build_ocp_from_plannif(
            self.whole_x_plan, self.whole_a_plan, self.whole_x_plan[0, :]
        )
        self.prob.run_solver(problem, self.prob.x_plan, self.prob.u_ref, 10)
        self.croco_xs = np.array(self.prob.solver.xs)
        self.croco_us = np.array(self.prob.solver.us)

    def compute_next_step(self, x, problem):
        m = problem.runningModels[0]
        d = m.createData()
        m.calc(d, x, self.prob.solver.us[0])
        return d.xnext.copy()

    def do_mpc(self, T, node_idx_breakpoint=None):
        mpc_xs = np.zeros([self.whole_traj_T, 2 * self.nq])
        mpc_us = np.zeros([self.whole_traj_T - 1, self.nq])
        x0 = self.whole_x_plan[0, :]
        mpc_xs[0, :] = x0

        x_plan = self.whole_x_plan[:T, :]
        a_plan = self.whole_a_plan[:T, :]
        x, u0 = self.mpc_first_step(x_plan, a_plan, x0, T)
        mpc_xs[1, :] = x
        mpc_us[0, :] = u0

        next_node_idx = T

        for idx in range(1, self.whole_traj_T - 1):
            x_plan = self.update_planning(x_plan, self.whole_x_plan[next_node_idx, :])
            a_plan = self.update_planning(a_plan, self.whole_a_plan[next_node_idx, :])
            x, u = self.mpc_step(x, x_plan, a_plan)
            if next_node_idx < self.whole_x_plan.shape[0] - 1:
                next_node_idx += 1
            mpc_xs[idx + 1, :] = x
            mpc_us[idx, :] = u

            if idx == node_idx_breakpoint:
                breakpoint()
        self.croco_xs = mpc_xs
        self.croco_us = mpc_us

    def update_planning(self, planning_vec, next_value):
        planning_vec = np.delete(planning_vec, 0, 0)
        return np.r_[planning_vec, next_value[np.newaxis, :]]

    def mpc_first_step(self, x_plan, a_plan, x0, T):
        problem = self.prob.build_ocp_from_plannif(x_plan, a_plan, x0)
        self.prob.run_solver(
            problem, list(x_plan), list(self.prob.u_ref[: T - 1]), 1000
        )
        x = self.compute_next_step(x0, self.prob.solver.problem)
        return x, self.prob.solver.us[0]

    def mpc_step(self, x, x_plan, a_plan):
        u_ref_terminal_node = self.prob.get_inverse_dynamic_control(
            x_plan[-1], a_plan[-1]
        )
        self.prob.reset_ocp(x, x_plan[-1], u_ref_terminal_node)
        xs_init = list(self.prob.solver.xs[1:]) + [self.prob.solver.xs[-1]]
        xs_init[0] = x
        us_init = list(self.prob.solver.us[1:]) + [self.prob.solver.us[-1]]
        self.prob.solver.problem.x0 = x
        self.prob.run_solver(self.prob.solver.problem, xs_init, us_init, 1)
        x = self.compute_next_step(x, self.prob.solver.problem)
        return x, self.prob.solver.us[0]
