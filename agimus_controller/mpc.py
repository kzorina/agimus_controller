import numpy as np
from .ocp_croco_hpp import OCPCrocoHPP


class TrajectoryBuffer:
    """List of variable size in which the HPP trajectory nodes will be.
    """
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
    """Create the MPC problem
    """
    def __init__(self, ocp, x_plan:np.ndarray, a_plan:np.ndarray, rmodel, cmodel):
        """Initiate the MPC problem.

        Args:
            ocp (OCP class): OCP describing the problem.
            x_plan (np.ndarray): State planification of HPP.
            a_plan (np.ndarray): Acceleration planification of HPP.
            rmodel (pin.Model): Pinocchio model of the robot
            cmodel (pin.CollisionModel): Pinocchio collision model.w
        """
        self._ocp = ocp
        self.whole_x_plan = x_plan
        self.whole_a_plan = a_plan
        self.rmodel = rmodel
        self.cmodel = cmodel
        self.nq = self.rmodel.nq
        self.croco_xs = None
        self.croco_us = None
        self.whole_traj_T = x_plan.shape[0]

    def get_next_state(self, x, problem):
        """Get state at the next step by doing a crocoddyl integration."""
        m = problem.runningModels[0]
        d = m.createData()
        m.calc(d, x, self._ocp.solver.us[0])
        return d.xnext.copy()

    def simulate_mpc(self, T, use_constraints=False, node_idx_breakpoint=None):
        """Simulate mpc behavior using crocoddyl integration as a simulator."""
        self._ocp.use_constraints = use_constraints
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
        """Update numpy array by removing the first value and adding next_value at the end."""
        planning_vec = np.delete(planning_vec, 0, 0)
        return np.r_[planning_vec, next_value[np.newaxis, :]]

    def mpc_first_step(self, x_plan, a_plan, x0, T):
        """Create crocoddyl problem from planning, run solver and get new state."""
        problem = self._ocp.build_ocp_from_plannif(x_plan, a_plan, x0)
        self._ocp.run_solver(
            problem, list(x_plan), list(self._ocp.u_ref[: T - 1]), 1000
        )
        x = self.get_next_state(x0, self._ocp.solver.problem)
        return x, self._ocp.solver.us[0]

    def mpc_step(self, x, x_plan, a_plan):
        """Reset ocp, run solver and get new state."""
        u_ref_terminal_node = self._ocp.get_inverse_dynamic_control(
            x_plan[-1], a_plan[-1]
        )
        self._ocp.reset_ocp(x, x_plan[-1], u_ref_terminal_node[: self.nq])
        xs_init = list(self._ocp.solver.xs[1:]) + [self._ocp.solver.xs[-1]]
        xs_init[0] = x
        us_init = list(self._ocp.solver.us[1:]) + [self._ocp.solver.us[-1]]
        self._ocp.solver.problem.x0 = x
        self._ocp.run_solver(self._ocp.solver.problem, xs_init, us_init, 1)
        x = self.get_next_state(x, self._ocp.solver.problem)
        return x, self._ocp.solver.us[0]
