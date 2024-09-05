import pinocchio as pin
from agimus_controller.robot_model.panda_model import PandaRobotModel
from agimus_controller.visualization.wrapper_meshcat import MeshcatWrapper
from agimus_controller.utils.process_handler import MeshcatServer


class APP(object):
    def main(self):
        self.meshcat_server = MeshcatServer()
        self.meshcat_server.start()

        # Creating the robot
        self.panda_wrapper = PandaRobotModel.load_model()
        self.rmodel = self.panda_wrapper.get_reduced_robot_model()
        self.cmodel = self.panda_wrapper.get_reduced_collision_model()
        self.vmodel = self.panda_wrapper.get_reduced_visual_model()
        # Generating the meshcat visualizer
        self.MeshcatVis = MeshcatWrapper()
        self.vis = self.MeshcatVis.visualize(
            robot_model=self.rmodel,
            robot_visual_model=self.vmodel,
            robot_collision_model=self.cmodel,
        )
        self.display_random_configuration()
        return True

    def display_random_configuration(self):
        self.vis[0].display(pin.randomConfiguration(self.rmodel))

    def display_configuration(self, q):
        self.vis[0].display(q)


def main():
    return APP().main()


if __name__ == "__main__":
    app = APP()
    app.main()
