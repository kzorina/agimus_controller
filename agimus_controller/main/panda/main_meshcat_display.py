from agimus_controller.robot_model.panda_model import PandaRobotModel
from agimus_controller.visualization.wrapper_meshcat import MeshcatWrapper
import pinocchio as pin


def main():
    # Creating the robot
    robot_wrapper = PandaRobotModel.load_model()
    rmodel = robot_wrapper.get_reduced_robot_model()
    cmodel = robot_wrapper.get_reduced_collision_model()
    vmodel = robot_wrapper.get_reduced_visual_model()
    # Generating the meshcat visualizer
    MeshcatVis = MeshcatWrapper()
    vis = MeshcatVis.visualize(
        robot_model=rmodel, robot_visual_model=vmodel, robot_collision_model=cmodel
    )
    vis[0].display(pin.randomConfiguration(rmodel))
    return True


if __name__ == "__main__":
    main()
