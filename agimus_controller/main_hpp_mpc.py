#!/usr/bin/env python

from agimus_controller.mpc import MPC
from agimus_controller.hpp_interface import HppInterface
from agimus_controller.plots import MPCPlots
import time


if __name__ == "__main__":
    hpp_interface = HppInterface()
    ps = hpp_interface.ps
    vf = hpp_interface.vf
    ball_init_pose = [-0.2, 0, 0.02, 0, 0, 0, 1]
    hpp_interface.get_hpp_plan(1e-2, 6)
    chc = MPC(hpp_interface.x_plan, hpp_interface.a_plan, "ur5")
    start = time.time()
    chc.prob.set_costs(10**4, 1, 10**-3, 0, 0)
    # chc.search_best_costs(chc.prob.nb_paths - 1, False, False, True)
    chc.simulate_mpc(100)
    end = time.time()
    u_plan = chc.prob.get_uref(hpp_interface.x_plan, hpp_interface.a_plan)
    mpc_plots = MPCPlots(
        chc.croco_xs,
        chc.croco_us,
        hpp_interface.x_plan,
        u_plan,
        chc.robot,
        vf,
        ball_init_pose,
        chc.prob.DT,
    )
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
