#!/usr/bin/env python

from agimus_controller.ocp_croco_hpp import OCPCrocoHPP
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
    ocp = OCPCrocoHPP("ur3")
    chc = MPC(ocp, hpp_interface.x_plan, hpp_interface.a_plan, ocp.robot.model)
    start = time.time()
    chc._ocp.set_weights(10**4, 1, 10**-3, 0)
    # chc.search_best_costs(chc.prob.nb_paths - 1, False, False, True)
    chc.simulate_mpc(100, True)
    end = time.time()
    u_plan = chc._ocp.get_uref(hpp_interface.x_plan, hpp_interface.a_plan)
    mpc_plots = MPCPlots(
        chc.croco_xs,
        chc.croco_us,
        hpp_interface.x_plan,
        u_plan,
        ocp.robot.model,
        chc._ocp.DT,
        "wrist_3_joint",
        vf,
        ball_init_pose,
    )
    print("mpc duration ", end - start)
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
