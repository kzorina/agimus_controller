import time
from agimus_controller.hpp_panda.hpp_interface import HppInterfacePanda
from agimus_controller.mpc import MPC
from agimus_controller.plots import MPCPlots


def main():
    hpp_interface = HppInterfacePanda()
    x_plan, a_plan, whole_traj_T = hpp_interface.get_hpp_plan(1e-2, 7)

    chc = MPC(x_plan, a_plan, "panda")
    start = time.time()
    chc.prob.set_costs(10**4, 1, 10**-3, 0, 0)
    # chc.search_best_costs(chc.prob.nb_paths - 1, False, False, True)
    chc.simulate_mpc(100)
    end = time.time()
    u_plan = chc.prob.get_uref(x_plan, a_plan)
    mpc_plots = MPCPlots(
        chc.croco_xs, chc.croco_us, x_plan, u_plan, chc.robot, chc.prob.DT
    )


if __name__ == "__main__":
    hpp_interface = HppInterfacePanda()
    x_plan, a_plan, whole_traj_T = hpp_interface.get_hpp_plan(1e-2, 7)

    chc = MPC(x_plan, a_plan, "panda")
    start = time.time()
    chc.prob.set_costs(10**4, 1, 10**-3, 0, 0)
    # chc.search_best_costs(chc.prob.nb_paths - 1, False, False, True)
    chc.simulate_mpc(100)
    end = time.time()
    u_plan = chc.prob.get_uref(x_plan, a_plan)
    mpc_plots = MPCPlots(
        chc.croco_xs, chc.croco_us, x_plan, u_plan, chc.robot, chc.prob.DT
    )
