import time
from agimus_controller.hpp_panda.hpp_interface import HppInterfacePanda
from agimus_controller.mpc import MPC
from agimus_controller.plots import MPCPlots

from wrapper_panda import PandaWrapper
from ocp_croco_hpp import OCPCrocoHPP

if __name__ == "__main__":
    pandawrapper = PandaWrapper()
    rmodel, cmodel, vmodel = pandawrapper.create_robot()
    ee_frame_name = pandawrapper.get_ee_frame_name()
    hpp_interface = HppInterfacePanda()
    x_plan, a_plan, whole_traj_T = hpp_interface.get_hpp_plan(1e-2, 7)
    ocp = OCPCrocoHPP("panda")
    chc = MPC(ocp, x_plan, a_plan, rmodel, cmodel)
    start = time.time()
    chc._ocp.set_weights(10**4, 1, 10**-3, 0)
    chc.simulate_mpc(100)
    end = time.time()
    u_plan = chc._ocp.get_uref(x_plan, a_plan)
    mpc_plots = MPCPlots(
        chc.croco_xs,
        chc.croco_us,
        x_plan,
        u_plan,
        rmodel,
        chc._ocp.DT,
        ee_frame_name=ee_frame_name,
    )
    mpc_plots.plot_traj()
