import time
from agimus_controller.hpp_interface import HppInterface
from agimus_controller.mpc import MPC
from agimus_controller.utils.plots import MPCPlots

from agimus_controller.utils.wrapper_panda import PandaWrapper
from agimus_controller.ocps.ocp_croco_hpp import OCPCrocoHPP

if __name__ == "__main__":
    pandawrapper = PandaWrapper(auto_col=True)
    rmodel, cmodel, vmodel = pandawrapper.create_robot()
    ee_frame_name = pandawrapper.get_ee_frame_name()
    q_init = [6.2e-01, 1.7e00, 1.5e00, -6.9e-01, -1.3e00, 1.1e00, 1.5e-01]

    hpp_interface = HppInterface()
    ps = hpp_interface.get_panda_planner(q_init)
    x_plan, a_plan, whole_traj_T = hpp_interface.get_hpp_plan(
        1e-2, 7, ps.client.problem.getPath(ps.numberPaths() - 1)
    )
    ocp = OCPCrocoHPP(rmodel, cmodel, use_constraints=False)

    mpc = MPC(ocp, x_plan, a_plan, rmodel, cmodel)
    start = time.time()
    mpc.ocp.set_weights(10**4, 1, 10**-3, 0)
    mpc.simulate_mpc(100)  # , node_idx_breakpoint=whole_traj_T - 30
    end = time.time()
    u_plan = mpc.ocp.get_u_plan(x_plan, a_plan)
    mpc_plots = MPCPlots(
        mpc.croco_xs,
        mpc.croco_us,
        x_plan,
        u_plan,
        rmodel,
        mpc.ocp.DT,
        ee_frame_name=ee_frame_name,
        v=hpp_interface.get_viewer(),
    )
    mpc_plots.plot_traj()
