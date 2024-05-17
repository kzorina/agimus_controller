from .problem import *
import matplotlib.pyplot as plt


class CrocoHppConnection:
    def __init__(self, ps, robot_name, vf, ball_init_pose):
        self.ball_init_pose = ball_init_pose
        if vf is not None:
            self.v = vf.createViewer()
        self.prob = Problem(ps, robot_name)
        self.robot = self.prob.robot
        self.nq = self.robot.nq
        self.DT = self.prob.DT
        self.croco_xs = None
        self.croco_us = None
        self.hpp_paths = self.prob.hpp_paths

    def plot_traj(self, terminal_idx):
        """Plot both trajectories of hpp and crocoddyl for the gripper pose."""
        pose_croco, pose_hpp = self.get_cartesian_trajectory(terminal_idx)
        path_time = self.get_path_length(terminal_idx)
        t = np.linspace(0, path_time, len(self.croco_xs))
        axis_string = ["x", "y", "z"]
        for idx in range(3):
            plt.subplot(2, 2, idx + 1)
            plt.plot(t, pose_croco[idx])
            plt.plot(t, pose_hpp[idx])
            plt.xlabel("time (s)")
            plt.ylabel("effector " + axis_string[idx] + " position")
            plt.legend(["crocoddyl", "hpp warm start"], loc="best")
        plt.show()

    def plot_traj_configuration(self, terminal_idx):
        """Plot both trajectories of hpp and crocoddyl in configuration space."""
        q_crocos, q_hpp = self.get_configuration_trajectory(terminal_idx)
        path_time = self.get_path_length(terminal_idx)
        t = np.linspace(0, path_time, len(self.croco_xs))
        for idx in range(self.nq):
            plt.subplot(self.nq, 1, idx + 1)
            plt.plot(t, q_crocos[idx])
            plt.plot(t, q_hpp[idx])
            plt.xlabel("time (s)")
            plt.ylabel(f"q{idx} position")
            plt.legend(["crocoddyl", "hpp warm start"], loc="best")
        plt.show()

    def plot_traj_velocity(self, terminal_idx):
        """Plot both velocities of hpp and crocoddyl."""
        v_crocos, v_hpp = self.get_velocity_trajectory(terminal_idx)
        path_time = self.get_path_length(terminal_idx)
        t = np.linspace(0, path_time, len(self.croco_xs))
        for idx in range(self.nq):
            plt.subplot(self.robot.nq, 1, idx + 1)
            plt.plot(t, v_crocos[idx])
            plt.plot(t, v_hpp[idx])
            plt.xlabel("time (s)")
            plt.ylabel(f"velocity q{idx}")
            plt.legend(["crocoddyl", "hpp warm start"], loc="best")
        plt.show()

    def plot_integrated_configuration(self, terminal_idx):
        """Plot both trajectories of hpp and crocoddyl in configuration space by integrating velocities."""

        v_crocos, v_hpp = self.get_velocity_trajectory(terminal_idx)
        q_crocos = [[] for _ in range(self.nq)]
        q_hpps = [[] for _ in range(self.nq)]

        # add initial configuration
        x0_croco = self.croco_xs[0]
        x0_hpp = self.hpp_paths[0].x_plan[0]
        for idx in range(self.nq):
            q_crocos[idx].append(x0_croco[idx])
            q_hpps[idx].append(x0_hpp[idx])

        # compute next configurations by integrating velocities
        for idx in range(len(self.croco_xs)):
            for joint_idx in range(self.nq):
                q_croco = q_crocos[joint_idx][-1] + v_crocos[joint_idx][idx] * self.DT
                q_crocos[joint_idx].append(q_croco)
                q_hpp = q_hpps[joint_idx][-1] + v_hpp[joint_idx][idx] * self.DT
                q_hpps[joint_idx].append(q_hpp)

        path_time = self.get_path_length(terminal_idx)
        t = np.linspace(0, path_time, len(self.croco_xs) + 1)
        for idx in range(self.nq):
            plt.subplot(3, 2, idx + 1)
            plt.plot(t, q_crocos[idx])
            plt.plot(t, q_hpps[idx])
            plt.xlabel("time (s)")
            plt.ylabel(f"q{idx} integrated")
            plt.legend(["crocoddyl", "hpp warm start"], loc="best")
        plt.show()

    def plot_control(self, terminal_idx):
        """Plot control for each joint."""
        us = self.get_configuration_control()
        path_time = self.get_path_length(terminal_idx)
        t = np.linspace(0, path_time, len(self.croco_us))
        for idx in range(self.nq):
            plt.subplot(self.nq, 1, idx + 1)
            plt.plot(t, us[idx])
            plt.xlabel("time (s)")
            plt.ylabel(f"q{idx} control")
        plt.show()

    def display_path(self, terminal_idx):
        """Display in Gepetto Viewer the trajectory found with crocoddyl."""
        path_time = self.get_path_length(terminal_idx)
        DT = path_time / len(self.croco_xs)
        for x in self.croco_xs:
            self.v(list(x)[: self.nq])  # + self.ball_init_pose
            time.sleep(DT)

    def print_final_placement(self, terminal_idx):
        """Print final gripper position for both hpp and crocoddyl trajectories."""
        q_final_hpp = self.hpp_paths[terminal_idx].x_plan[-1][: self.nq]
        hpp_placement = self.robot.placement(q_final_hpp, self.nq)
        print("Last node placement ")
        print(
            "hpp rot ",
            pin.log(hpp_placement.rotation),
            " translation ",
            hpp_placement.translation,
        )
        q_final_croco = self.croco_xs[-1][: self.nq]
        croco_placement = self.robot.placement(q_final_croco, self.nq)
        print(
            "croco rot ",
            pin.log(croco_placement.rotation),
            " translation ",
            croco_placement.translation,
        )

    def get_path_length(self, terminal_idx):
        length = 0
        for idx in range(terminal_idx):
            length += self.prob.hpp_paths[idx].path.length()
        return length

    def get_trajectory_difference(self, terminal_idx, configuration_traj=True):
        """Compute at each node the absolute difference in position either in cartesian or configuration space and sum it."""
        if configuration_traj:
            traj_croco, traj_hpp = self.get_configuration_trajectory(terminal_idx)
        else:
            traj_croco, traj_hpp = self.get_cartesian_trajectory(terminal_idx)
        diffs = []
        for idx in range(len(traj_croco)):
            array_diff = np.abs(np.array(traj_croco[idx]) - np.array(traj_hpp[idx]))
            diffs.append(np.sum(array_diff))
        return sum(diffs)

    def get_cartesian_trajectory(self, terminal_idx):
        """Return the vector of gripper pose for both trajectories found by hpp and crocoddyl."""
        pose_croco = [[] for _ in range(3)]
        pose_hpp = [[] for _ in range(3)]
        for x in self.croco_xs:
            q = x[: self.nq]
            pose = self.robot.placement(q, self.nq).translation
            for idx in range(3):
                pose_croco[idx].append(pose[idx])
        for path_idx in range(terminal_idx + 1):
            for x in self.hpp_paths[path_idx].x_plan:
                q = x[: self.nq]
                pose = self.robot.placement(q, self.nq).translation
                for idx in range(3):
                    pose_hpp[idx].append(pose[idx])
        return pose_croco, pose_hpp

    def get_configuration_trajectory(self, terminal_idx):
        """Return the vector of configuration for both trajectories found by hpp and crocoddyl."""
        q_crocos = [[] for _ in range(self.nq)]
        q_hpp = [[] for _ in range(self.nq)]
        for x in self.croco_xs:
            for idx in range(self.nq):
                q_crocos[idx].append(x[idx])
        for path_idx in range(terminal_idx + 1):
            for x in self.hpp_paths[path_idx].x_plan:
                for idx in range(self.nq):
                    q_hpp[idx].append(x[idx])
        return q_crocos, q_hpp

    def get_velocity_trajectory(self, terminal_idx):
        """Return the vector of velocity for both trajectories found by hpp and crocoddyl."""
        v_crocos = [[] for _ in range(self.nq)]
        v_hpp = [[] for _ in range(self.nq)]
        for x in self.croco_xs:
            for idx in range(self.nq):
                v_crocos[idx].append(x[idx + self.nq])
        for path_idx in range(terminal_idx + 1):
            for x in self.hpp_paths[path_idx].x_plan:
                for idx in range(self.nq):
                    v_hpp[idx].append(x[idx + self.nq])
        return v_crocos, v_hpp

    def get_configuration_control(self):
        """Return the vector of configuration for both trajectories found by hpp and crocoddyl."""
        us = [[] for _ in range(self.nq)]
        for u in self.croco_us:
            for idx in range(self.nq):
                us[idx].append(u[idx])
        return us

    def search_best_costs(
        self, terminal_idx, use_mim=False, configuration_traj=False, is_mpc=False
    ):
        """Search costs that minimize the gap between hpp and crocoddyl trajectories."""
        self.best_combination = None
        self.best_diff = 1e6
        self.best_solver = None
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
                        vel_exponent=0,
                        xlim_exponent=0,
                        is_mpc=is_mpc,
                    )
                    end = time.time()
                    print("iteration duration ", end - start)
        else:
            for grip_exponent in range(25, 50, 5):
                print("grip expo ", grip_exponent)
                for x_exponent in range(5, 25, 5):
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
                            vel_exponent=0,
                            xlim_exponent=0,
                            is_mpc=is_mpc,
                        )
                        end = time.time()
                        print("iteration duration ", end - start)

        self.prob.solver = self.best_solver
        if is_mpc == False:
            self.croco_xs = self.prob.solver.xs
            self.croco_us = self.prob.solver.us
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
        grip_exponent,
        x_exponent,
        u_exponent,
        terminal_idx,
        vel_exponent=0,
        xlim_exponent=0,
        configuration_traj=False,
        is_mpc=False,
    ):
        """Set problem, run solver and check if we found a better solution."""
        if is_mpc:
            print("doing mpc")
            self.do_mpc(terminal_idx, 100)
        else:
            print("doing ocp")
            self.set_problem_run_solver(
                terminal_idx,
            )
        diff = self.get_trajectory_difference(terminal_idx, configuration_traj)
        if diff < self.best_diff:
            self.best_combination = [
                grip_exponent,
                x_exponent,
                u_exponent,
                vel_exponent,
                xlim_exponent,
            ]
            self.best_diff = diff
            self.best_solver = self.prob.solver

    def set_problem_run_solver(self, terminal_idx):
        """Set OCP problem with new costs then run solver."""

        self.prob.set_models([terminal_idx])
        self.prob.create_whole_problem()
        self.prob.set_xplan_and_uref(0, terminal_idx)
        self.prob.run_solver(
            self.prob.whole_problem, self.prob.x_plan, self.prob.u_ref, 10
        )
        self.croco_xs = self.prob.solver.xs
        self.croco_us = self.prob.solver.us

    def compute_next_step(self, x, problem):
        m = problem.runningModels[0]
        # m.dt = 1e-2
        d = m.createData()
        m.calc(d, x, self.prob.solver.us[0])
        return d.xnext.copy()

    def do_mpc(self, path_terminal_idx, T):
        self.prob.set_models([path_terminal_idx])
        self.prob.create_whole_problem()
        self.prob.set_xplan_and_uref(0, path_terminal_idx)

        problem = self.prob.create_problem(T)
        problem.x0 = self.hpp_paths[0].x_plan[
            0
        ]  # np.concatenate([self.robot.q0, np.array([0, 0, 0, 0, 0, 0])])
        self.prob.run_solver(
            problem, self.prob.x_plan[:T], self.prob.u_ref[: T - 1], 1000
        )
        next_node_idx = T
        xs = [problem.x0]
        us = [self.prob.solver.us[0]]
        x = self.compute_next_step(problem.x0, problem)
        self.croco_xs = self.prob.solver.xs
        self.croco_us = self.prob.solver.us

        xs.append(x)

        for _ in range(len(self.prob.whole_problem.runningModels.tolist()) - 1):
            self.prob.reset_ocp(x, next_node_idx)
            xs_init = list(self.prob.solver.xs[1:]) + [self.prob.solver.xs[-1]]
            xs_init[0] = x
            us_init = list(self.prob.solver.us[1:]) + [self.prob.solver.us[-1]]

            self.prob.run_solver(self.prob.solver.problem, xs_init, us_init, 1)
            x = self.compute_next_step(x, self.prob.solver.problem)
            xs.append(x)
            us.append(self.prob.solver.us[0])
            next_node_idx += 1
            self.problem = self.prob.solver.problem.copy()
        self.croco_xs = xs
        self.croco_us = us
