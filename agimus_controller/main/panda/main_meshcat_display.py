from agimus_controller.agimus_controller.robot_model.wrapper_panda import PandaWrapper
from agimus_controller.utils.wrapper_meshcat import MeshcatWrapper
import pinocchio as pin


def main():
    # Creating the robot
    robot_wrapper = PandaWrapper(capsule=True)
    rmodel, cmodel, vmodel = robot_wrapper()
    # rdata = rmodel.createData()
    # cdata = cmodel.createData()
    # Generating the meshcat visualizer
    MeshcatVis = MeshcatWrapper()
    vis = MeshcatVis.visualize(
        robot_model=rmodel, robot_visual_model=vmodel, robot_collision_model=cmodel
    )
    vis[0].display(pin.randomConfiguration(rmodel))
    return True


if __name__ == "__main__":
    main()
