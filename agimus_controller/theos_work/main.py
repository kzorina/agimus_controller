#!/usr/bin/env python

from agimus_controller.theos_work.croco_hpp import CrocoHppConnection
from agimus_controller.hpp_interface import HppInterface
import time


class TrajectoryPoint:
    def __init__(self, x, a, com_pose=None):
        self.x = x
        self.a = a
        self.com_pose = com_pose


if __name__ == "__main__":
    hpp_interface = HppInterface()
    ps = hpp_interface.ps
    vf = hpp_interface.vf
    ball_init_pose = [-0.2, 0, 0.02, 0, 0, 0, 1]
    hpp_interface.get_hpp_plan(1e-2, 6)
    chc = CrocoHppConnection(
        hpp_interface.x_plan, hpp_interface.a_plan, "ur5", vf, ball_init_pose
    )
    start = time.time()
    chc.prob.set_costs(10**4, 1, 10**-3, 0, 0)
    # chc.search_best_costs(chc.prob.nb_paths - 1, False, False, True)
    chc.do_mpc(100)
    end = time.time()
    print("search duration ", end - start)
    """
    with open("datas.npy", "wb") as f:
        np.save(f, chc.prob.hpp_paths[0].x_plan)
        np.save(f, chc.prob.hpp_paths[1].x_plan)"""

"""
from hpp.gepetto import PathPlayer
v =vf.createViewer()
pp = PathPlayer (v)"""
# plot_costs_from_dic(return_cost_vectors(chc.prob.solver,weighted=True))
# plot_costs_from_dic(return_cost_vectors(self.prob.solver,weighted=True))
