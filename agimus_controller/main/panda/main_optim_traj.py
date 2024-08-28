import time
import numpy as np
import pinocchio as pin
from pathlib import Path

from agimus_controller.visualization.wrapper_meshcat import MeshcatWrapper
from agimus_controller.robot_model.panda_model import (
    PandaRobotModel,
    PandaRobotModelParameters,
)
from agimus_controller.ocps.ocp import OCPPandaReachingColWithMultipleCol
from agimus_controller.hpp_panda.scenes import Scene


def main(display=False):
    ### PARAMETERS
    # Number of nodes of the trajectory
    T = 20
    # Time step between each node
    dt = 0.01

    # Creating the robot
    panda_params = PandaRobotModelParameters()
    panda_params.collision_as_capsule = True
    panda_params.self_collision = True
    env = Path(__file__).resolve().parent.parent.parent / "resources" / "panda_env.yaml"
    pandawrapper = PandaRobotModel.load_model(params=panda_params, env=env)
    rmodel = pandawrapper.get_reduced_robot_model()
    cmodel = pandawrapper.get_reduced_collision_model()
    vmodel = pandawrapper.get_reduced_visual_model()

    # Creating the scene
    scene = Scene(name_scene="wall", q_init=pandawrapper.get_default_configuration())
    rmodel1, cmodel1, _, TARGET2, q0 = scene.create_scene_from_urdf(rmodel, cmodel)
    # Generating the meshcat visualizer
    MeshcatVis = MeshcatWrapper()
    vis, _ = MeshcatVis.visualize(
        TARGET2,
        robot_model=rmodel1,
        robot_collision_model=cmodel1,
        robot_visual_model=vmodel,
    )

    ### INITIAL X0
    x0 = np.concatenate([q0, pin.utils.zero(rmodel.nv)])

    ### CREATING THE PROBLEM WITHOUT WARM START
    problem = OCPPandaReachingColWithMultipleCol(
        rmodel1,
        cmodel1,
        TARGET2,
        T,
        dt,
        x0,
        WEIGHT_GRIPPER_POSE=100,
        WEIGHT_xREG=1e-2,
        WEIGHT_uREG=1e-4,
        SAFETY_THRESHOLD=2.5e-3,
        callbacks=True,
    )

    ddp = problem()

    XS_init = [x0] * (T + 1)
    # US_init = [np.zeros(rmodel.nv)] * T
    US_init = ddp.problem.quasiStatic(XS_init[:-1])
    # Solving the problem
    ddp.solve(XS_init, US_init)

    print("End of the computation, press enter to display the traj if requested.")
    ### DISPLAYING THE TRAJ
    while display:
        vis.display(q0)
        input()
        for xs in ddp.xs:
            vis.display(np.array(xs[:7].tolist()))
            time.sleep(1e-1)
        input()
        print("replay")

    return True


if __name__ == "__main__":
    main(True)
