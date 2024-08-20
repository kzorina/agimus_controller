#!/usr/bin/env python
import time
import numpy as np
from agimus_controller.robot_model.ur5_model import UR5RobotModel

from agimus_controller.ocps.ocp_croco_hpp import OCPCrocoHPP
from agimus_controller.mpc import MPC
from agimus_controller.hpp_interface import HppInterface
from agimus_controller.visualization.plots import MPCPlots


def main():
    robot = UR5RobotModel()
    rmodel = robot.get_reduced_robot_model()
    hpp_interface = HppInterface()
    hpp_interface.start_corbaserver()
    hpp_interface.set_ur3_problem_solver(robot.get_default_configuration())
    x_plan, a_plan, _ = hpp_interface.get_hpp_x_a_planning(1e-2)
    viewer = hpp_interface.get_viewer()
    armature = np.zeros(rmodel.nq)
    ocp = OCPCrocoHPP(
        rmodel=rmodel, cmodel=None, use_constraints=False, armature=armature
    )
    mpc = MPC(ocp, x_plan, a_plan, rmodel)
    start = time.time()
    mpc.ocp.set_weights(10**4, 1, 10**-3, 0)
    mpc.simulate_mpc(T=100, save_predictions=False)
    end = time.time()
    u_plan = mpc.ocp.get_u_plan(x_plan, a_plan)
    MPCPlots(
        croco_xs=mpc.croco_xs,
        croco_us=mpc.croco_us,
        whole_x_plan=x_plan,
        whole_u_plan=u_plan,
        rmodel=rmodel,
        DT=mpc.ocp.DT,
        ee_frame_name="wrist_3_joint",
        viewer=viewer,
    )
    print("mpc duration ", end - start)
    return True


if __name__ == "__main__":
    main()
