from problem import *
import time
import matplotlib.pyplot as plt


class CrocoHppConnection:
    def __init__(self, ps, robot_name, vf, ball_init_pose):
        self.ball_init_pose = ball_init_pose
        self.v = vf.createViewer()
        self.prob = Problem(ps, robot_name)
        self.robot = example_robot_data.load(robot_name)
        self.nq = self.robot.nq
        self.croco_xs = None
        self.hpp_paths = None

    def plot_traj(self, terminal_idx):
        """Plot both trajectories of hpp and crocoddyl for the gripper pose."""
        pose_croco, pose_hpp = self.get_cartesian_trajectory(terminal_idx)
        t = np.linspace(0, self.prob.p.length(), len(self.croco_xs))
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
        t = np.linspace(0, self.prob.p.length(), len(self.croco_xs))
        for idx in range(self.nq):
            plt.subplot(3, 2, idx + 1)
            plt.plot(t, q_crocos[idx])
            plt.plot(t, q_hpp[idx])
            plt.xlabel("time (s)")
            plt.ylabel(f"q{idx} position")
            plt.legend(["crocoddyl", "hpp warm start"], loc="best")
        plt.show()

    def plot_control(self):
        """Plot control for each joint."""
        us = self.get_configuration_control()
        t = np.linspace(0, self.prob.p.length(), len(self.prob.ddp.us))
        for idx in range(self.nq):
            plt.subplot(3, 2, idx + 1)
            plt.plot(t, us[idx])
            plt.xlabel("time (s)")
            plt.ylabel(f"q{idx} control")
            # plt.legend(["crocoddyl", "hpp warm start"], loc="best")
        plt.show()

    def display_path(self):
        """Display in Gepetto Viewer the trajectory found with crocoddyl."""
        DT = self.prob.p.length() / len(self.croco_xs)
        for x in self.croco_xs:
            self.v(list(x)[: self.nq] + self.ball_init_pose)
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

    def get_configuration_control(self):
        """Return the vector of configuration for both trajectories found by hpp and crocoddyl."""
        us = [[] for _ in range(self.nq)]
        for u in self.prob.ddp.us:
            for idx in range(self.nq):
                us[idx].append(u[idx])
        return us

    def search_best_costs(self, terminal_idx, configuration_traj=True):
        """Search costs that minimize the gap between hpp and crocoddyl trajectories."""
        best_combination = None
        best_diff = 1e6
        best_ddp = None
        for grip_exponent in range(80, 100, 2):
            for x_exponent in range(0, 20, 2):
                self.set_problem_run_ddp(
                    10 ** (grip_exponent / 10),
                    10 ** (x_exponent / 10),
                    1e-3,
                    terminal_idx,
                )
                diff = self.get_trajectory_difference(terminal_idx, configuration_traj)
                if diff < best_diff:
                    best_combination = [grip_exponent, x_exponent]
                    best_diff = diff
                    best_ddp = self.prob.ddp
        self.prob.ddp = best_ddp
        self.croco_xs = self.prob.ddp.xs
        print("best diff ", best_diff)
        print("best combination ", best_combination)

    def set_problem_run_ddp(self, grip_cost, x_cost, u_cost, terminal_idx):
        """Set ddp problem with new costs then run ddp."""
        self.prob.set_costs(grip_cost, x_cost, u_cost)
        self.prob.set_models([terminal_idx])
        self.prob.run_ddp(0, terminal_idx)
        self.croco_xs = self.prob.ddp.xs
        self.hpp_paths = self.prob.hpp_paths
