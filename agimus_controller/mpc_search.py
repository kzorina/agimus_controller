import time
import numpy as np
from .mpc import MPC


class MPCSearch:
    def __init__(self, mpc: MPC):
        self.mpc = mpc
        self.whole_x_plan = self.mpc.whole_x_plan
        self.whole_a_plan = self.mpc.whole_a_plan
        self.robot = self.mpc.robot
        self.nq = self.robot.nq
        self.croco_xs = None
        self.croco_us = None
        self.results = {}
        self.results["xs"] = []
        self.results["us"] = []
        self.results["max_us"] = []
        self.results["max_increase_us"] = []
        self.results["combination"] = []

    def get_trajectory_difference(self, configuration_traj=True):
        """Compute at each node the absolute difference in position either in cartesian or configuration space and sum it."""
        if configuration_traj:
            traj_croco = self.croco_xs[:, : self.nq]
            traj_hpp = self.whole_x_plan[:, : self.nq]
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

    def search_best_costs(self, use_constraints=False, configuration_traj=False):
        """Search costs that minimize the gap between hpp and crocoddyl trajectories."""
        self.best_combination = None
        self.best_croco_xs = None
        self.best_croco_us = None
        self.best_diff = 1e6
        grip_cost = 0
        self.mpc.prob.use_constraints = use_constraints
        if use_constraints:
            for x_exponent in range(0, 8, 2):
                for u_exponent in range(-32, -26, 2):
                    start = time.time()
                    _, x_cost, u_cost, _, _ = self.get_cost_from_exponent(
                        0, x_exponent, u_exponent, 0, 0
                    )
                    print(" x : ", x_cost, " u : ", u_cost)
                    self.try_new_costs(
                        grip_cost,
                        x_cost,
                        u_cost,
                        configuration_traj=configuration_traj,
                        vel_cost=0,
                        xlim_cost=0,
                    )
                    end = time.time()
                    print("mpc simulation duration ", end - start)
        else:
            for grip_exponent in range(25, 50, 5):
                for x_exponent in range(0, 15, 5):
                    for u_exponent in range(-35, -25, 5):
                        grip_cost, x_cost, u_cost, _, _ = self.get_cost_from_exponent(
                            grip_exponent, x_exponent, u_exponent, 0, 0
                        )
                        print(
                            "weights pose : ",
                            grip_cost,
                            " x : ",
                            x_cost,
                            " u : ",
                            u_cost,
                        )
                        self.mpc.prob.set_costs(grip_cost, x_cost, u_cost, 0, 0)
                        start = time.time()
                        self.try_new_costs(
                            grip_cost,
                            x_cost,
                            u_cost,
                            configuration_traj=configuration_traj,
                            vel_cost=0,
                            xlim_cost=0,
                        )
                        end = time.time()
                        print("mpc simulation duration ", end - start)

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
        vel_cost=0,
        xlim_cost=0,
        configuration_traj=False,
    ):
        """Set problem, run solver, add result in dict and check if we found a better solution."""
        self.mpc.simulate_mpc(100)
        self.croco_xs = self.mpc.croco_xs
        self.croco_us = self.mpc.croco_us
        max_us = np.max(np.abs(self.croco_us))
        max_increase_us, _ = self.max_increase_us()
        self.results["xs"].append(self.croco_xs)
        self.results["us"].append(self.croco_us)
        self.results["max_us"].append(max_us)
        self.results["max_increase_us"].append(max_increase_us)
        self.results["combination"].append(
            [grip_cost, x_cost, u_cost, vel_cost, xlim_cost]
        )
        diff = self.get_trajectory_difference(configuration_traj)
        if diff < self.best_diff and max_us < 100 and max_increase_us < 50:
            self.best_combination = [grip_cost, x_cost, u_cost, vel_cost, xlim_cost]
            self.best_diff = diff
            self.best_croco_xs = self.croco_xs
            self.best_croco_us = self.croco_us
