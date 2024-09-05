import numpy as np
from pathlib import Path
import pinocchio as pin
from agimus_controller.robot_model.panda_model import (
    PandaRobotModel,
    PandaRobotModelParameters,
)
from agimus_controller.main.servers import MeshcatServer
from agimus_controller.visualization.wrapper_meshcat import MeshcatWrapper
from agimus_controller.hpp_panda.scenes import Scene


class APP(object):
    def main(self, use_gui=False, spawn_servers=False):
        if spawn_servers and use_gui:
            self.servers = MeshcatServer()

        # Creating the robot
        robot_params = PandaRobotModelParameters()
        robot_params.self_collision = False
        robot_params.collision_as_capsule = True
        panda_wrapper = PandaRobotModel.load_model(
            params=robot_params,
            env=Path(__file__).resolve().parent.parent.parent
            / "resources"
            / "panda_env.yaml",
        )
        rmodel = panda_wrapper.get_reduced_robot_model()
        cmodel = panda_wrapper.get_reduced_collision_model()
        vmodel = panda_wrapper.get_reduced_visual_model()

        scene = Scene("wall", q_init=panda_wrapper.get_default_configuration())
        rmodel, cmodel, target, target2, q0 = scene.create_scene_from_urdf(
            rmodel, cmodel
        )
        q = np.array(
            [
                1.06747267,
                1.44892299,
                -0.10145964,
                -2.42389347,
                2.60903241,
                3.45138352,
                -2.04166928,
            ]
        )
        q = panda_wrapper.get_default_configuration()
        rdata = rmodel.createData()
        cdata = cmodel.createData()
        # Generating the meshcat visualizer
        MeshcatVis = MeshcatWrapper()
        vis = MeshcatVis.visualize(
            robot_model=rmodel,
            robot_visual_model=vmodel,
            robot_collision_model=cmodel,
            robot_data=rdata,
            robot_collision_data=cdata,
            TARGET=target,
        )
        vis[0].display(q)

        pin.computeCollisions(rmodel, rdata, cmodel, cdata, q, False)
        for k in range(len(cmodel.collisionPairs)):
            cr = cdata.collisionResults[k]
            cp = cmodel.collisionPairs[k]
            print(
                "collision pair:",
                cmodel.geometryObjects[cp.first].name,
                ",",
                cmodel.geometryObjects[cp.second].name,
                "- collision:",
                "Yes" if cr.isCollision() else "No",
            )


def main():
    return APP().main(use_gui=False, spawn_servers=False)


if __name__ == "__main__":
    app = APP()
    app.main(use_gui=True, spawn_servers=True)
