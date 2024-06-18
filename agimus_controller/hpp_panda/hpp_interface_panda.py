import time
import numpy as np

from agimus_controller.hpp_panda.planner import Planner
from agimus_controller.hpp_panda.scenes import Scene
from agimus_controller.hpp_panda.wrapper_panda import PandaWrapper
from agimus_controller.hpp_interface import HppInterface
from agimus_controller.trajectory_point import TrajectoryPoint


class HppInterfacePanda:
    def __init__(self) -> None:
        # Creating the robot

        self.T = 20
        self.robot_wrapper = PandaWrapper(capsule=True, auto_col=True)
        self.rmodel, self.cmodel, self.vmodel = self.robot_wrapper()

        self.name_scene = "wall"
        self.scene = Scene(self.name_scene)
        self.rmodel, self.cmodel, self.target, self.target2, self.q0 = (
            self.scene.create_scene_from_urdf(self.rmodel, self.cmodel)
        )
        self.time_calc = []
        self.results = []
        self.planner = Planner(self.rmodel, self.cmodel, self.scene, self.T)
        self.start = time.process_time()
        self.q_init, self.q_goal, self.X = self.planner.solve_and_optimize()
        self.t_solve = time.process_time() - self.start
        self.time_calc.append(self.t_solve)
        self.results.append([self.q_init, self.q_goal, self.X])

    def _get_hpp_plan(self, DT, nq, ps):
        p = ps.client.problem.getPath(ps.numberPaths() - 1)
        path = p.pathAtRank(0)
        T = int(np.round(path.length() / DT))
        x_plan, a_plan, subpath = self._get_xplan_aplan(T, path, nq)
        trajectory = []
        whole_traj_T = 0
        for path_idx in range(1, p.numberPaths()):
            path = p.pathAtRank(path_idx)
            T = int(np.round(path.length() / DT))
            if T == 0:
                continue
            subpath_x_plan, subpath_a_plan, subpath = self._get_xplan_aplan(T, path, nq)
            x_plan = np.concatenate([x_plan, subpath_x_plan], axis=0)
            a_plan = np.concatenate([a_plan, subpath_a_plan], axis=0)
            trajectory += subpath
            whole_traj_T += T
        return x_plan, a_plan, whole_traj_T

    def _get_xplan_aplan(self, T, path, nq):
        """Return x_plan the state and a_plan the acceleration of hpp's trajectory."""
        x_plan = np.zeros([T, 2 * nq])
        a_plan = np.zeros([T, nq])
        subpath = []
        trajectory_point = TrajectoryPoint()
        trajectory_point.q = np.zeros(nq)
        trajectory_point.v = np.zeros(nq)
        trajectory_point.a = np.zeros(nq)
        subpath = [trajectory_point]
        if T == 0:
            pass
        elif T == 1:
            time = path.length()
            q_t = np.array(path.call(time)[0][:nq])
            v_t = np.array(path.derivative(time, 1)[:nq])
            x_plan[0, :] = np.concatenate([q_t, v_t])
            a_t = np.array(path.derivative(time, 2)[:nq])
            a_plan[0, :] = a_t
            subpath[0].q[:] = q_t[:]
            subpath[0].v[:] = v_t[:]
            subpath[0].a[:] = a_t[:]
        else:
            total_time = path.length()
            subpath = [TrajectoryPoint(t, nq, nq) for t in range(T)]
            for iter in range(T):
                iter_time = total_time * iter / (T - 1)  # iter * DT
                q_t = np.array(path.call(iter_time)[0][:nq])
                v_t = np.array(path.derivative(iter_time, 1)[:nq])
                x_plan[iter, :] = np.concatenate([q_t, v_t])
                a_t = np.array(path.derivative(iter_time, 2)[:nq])
                a_plan[iter, :] = a_t
                subpath[iter].q[:] = q_t[:]
                subpath[iter].v[:] = v_t[:]
                subpath[iter].a[:] = a_t[:]
        return x_plan, a_plan, subpath

    def display_path(self, croco_xs, v, nq, DT):
        """Display in Gepetto Viewer the trajectory found with crocoddyl."""
        for x in croco_xs:
            self.planner._v(list(x)[:nq] + [0] * 5 + [1])  # + self.ball_init_pose
            time.sleep(DT)

    def get_hpp_plan(self, dt, nq):
        hpp_interface = HppInterface()
        hpp_interface.ps = self.planner._ps
        return self._get_hpp_plan(dt, nq, hpp_interface.ps)