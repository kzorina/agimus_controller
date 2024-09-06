import time
import numpy as np
from agimus_controller.hpp_interface import HppInterface
from agimus_controller.mpc import MPC
from agimus_controller.visualization.plots import MPCPlots
from agimus_controller.ocps.ocp_croco_hpp import OCPCrocoHPP
from agimus_controller.robot_model.panda_model import (
    PandaRobotModel,
    PandaRobotModelParameters,
)
from agimus_controller.main.servers import Servers


class APP(object):
    def main(self, use_gui=False, spawn_servers=False):
        if spawn_servers:
            self.servers = Servers()
            self.servers.spawn_servers(use_gui)

        panda_params = PandaRobotModelParameters()
        panda_params.collision_as_capsule = True
        panda_params.self_collision = True
        pandawrapper = PandaRobotModel.load_model(params=panda_params)

        rmodel = pandawrapper.get_reduced_robot_model()
        cmodel = pandawrapper.get_reduced_collision_model()
        ee_frame_name = panda_params.ee_frame_name

        hpp_interface = HppInterface()
        q_init, q_goal = hpp_interface.get_panda_q_init_q_goal()
        hpp_interface.set_panda_planning(q_init, q_goal, use_gepetto_gui=use_gui)
        viewer = hpp_interface.get_viewer()
        x_plan, a_plan, _ = hpp_interface.get_hpp_x_a_planning(1e-2)

        armature = np.zeros(rmodel.nq)
        ocp = OCPCrocoHPP(rmodel, cmodel, use_constraints=False, armature=armature)
        mpc = MPC(ocp, x_plan, a_plan, rmodel, cmodel)

        start = time.time()
        mpc.ocp.set_weights(10**4, 1, 10**-3, 0)
        mpc.simulate_mpc(T=100, save_predictions=False)
        end = time.time()
        print("Time of solving: ", end - start)
        u_plan = mpc.ocp.get_u_plan(x_plan, a_plan)
        self.mpc_plots = MPCPlots(
            croco_xs=mpc.croco_xs,
            croco_us=mpc.croco_us,
            whole_x_plan=x_plan,
            whole_u_plan=u_plan,
            rmodel=rmodel,
            DT=mpc.ocp.DT,
            ee_frame_name=ee_frame_name,
            viewer=viewer,
        )
        if use_gui:
            self.mpc_plots.display_path()
        return True


def main():
    return APP().main(use_gui=False, spawn_servers=False)


if __name__ == "__main__":
    app = APP()
    app.main(use_gui=True, spawn_servers=True)
