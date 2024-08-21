import unittest
from pathlib import Path
import pinocchio as pin
from agimus_controller.robot_model.robot_model import RobotModel
from agimus_controller.robot_model.robot_model import RobotModelParameters
from agimus_controller.robot_model.panda_model import PandaRobotModel
from agimus_controller.robot_model.panda_model import PandaRobotModelParameters
from agimus_controller.robot_model.ur5_model import UR5RobotModel


class TestBuildModel(unittest.TestCase):
    def test_constructor(self):
        robot_model = RobotModel()

        self.assertEqual(robot_model.get_complete_robot_model(), pin.Model())
        self.assertEqual(
            robot_model.get_complete_collision_model(), pin.GeometryModel()
        )
        self.assertEqual(robot_model.get_complete_visual_model(), pin.GeometryModel())
        self.assertEqual(robot_model.get_reduced_robot_model(), pin.Model())
        self.assertEqual(robot_model.get_reduced_collision_model(), pin.GeometryModel())
        self.assertEqual(robot_model.get_reduced_visual_model(), pin.GeometryModel())
        self.assertEqual(robot_model.get_default_configuration().size, 0)

    def test_load_panda_model(self):
        robot_params = PandaRobotModelParameters()
        robot_params.self_collision = False
        robot_params.collision_as_capsule = False
        robot_model = PandaRobotModel.load_model(params=robot_params)

        self.assertNotEqual(robot_model.get_complete_robot_model(), pin.Model())
        self.assertNotEqual(robot_model.get_complete_collision_model(), pin.Model())
        self.assertNotEqual(robot_model.get_complete_visual_model(), pin.Model())
        self.assertNotEqual(robot_model.get_reduced_robot_model(), pin.Model())
        self.assertNotEqual(robot_model.get_reduced_collision_model(), pin.Model())
        self.assertNotEqual(robot_model.get_reduced_visual_model(), pin.Model())
        self.assertNotEqual(robot_model.get_default_configuration().size, 0)

        m = robot_model.get_reduced_robot_model()
        self.assertEqual(m.nq, m.nv)
        self.assertEqual(m.nq, 7)
        self.assertEqual(m.name, "panda")
        self.assertTrue(m.existFrame("panda_hand_joint"))
        self.assertTrue(m.existFrame("panda_camera_joint"))

    def test_load_panda_self_collision(self):
        robot_params = PandaRobotModelParameters()
        robot_params.self_collision = True
        robot_params.collision_as_capsule = False
        PandaRobotModel.load_model(params=robot_params)

    def test_load_panda_collision_as_capsule(self):
        robot_params = PandaRobotModelParameters()
        robot_params.self_collision = False
        robot_params.collision_as_capsule = True
        PandaRobotModel.load_model(params=robot_params)

    def test_load_panda_collision_as_capsule_and_self_collision(self):
        robot_params = PandaRobotModelParameters()
        robot_params.collision_as_capsule = True
        robot_params.self_collision = True
        PandaRobotModel.load_model(params=robot_params)

    def test_load_panda_collisions(self):
        robot_params = PandaRobotModelParameters()
        env = Path(__file__).resolve().parent / "resources" / "col_env.yaml"
        robot_model = PandaRobotModel.load_model(env=env, params=robot_params)

        with open("robot_model.col", "w") as f:
            f.write(str(robot_model.get_reduced_collision_model()))

    def test_load_ur5_model(self):
        robot_model = UR5RobotModel.load_model()
        self.assertNotEqual(robot_model.get_complete_robot_model(), pin.Model())
        self.assertNotEqual(robot_model.get_complete_collision_model(), pin.Model())
        self.assertNotEqual(robot_model.get_complete_visual_model(), pin.Model())
        self.assertNotEqual(robot_model.get_reduced_robot_model(), pin.Model())
        self.assertNotEqual(robot_model.get_reduced_collision_model(), pin.Model())
        self.assertNotEqual(robot_model.get_reduced_visual_model(), pin.Model())
        self.assertNotEqual(robot_model.get_default_configuration().size, 0)

        m = robot_model.get_reduced_robot_model()
        self.assertEqual(m.nq, m.nv)
        self.assertEqual(m.nq, 6)
        self.assertEqual(m.name, "ur5")

    def test_vs_previous_version(self):
        from build_models import RobotModelConstructor
        import example_robot_data

        params = RobotModelParameters()
        params.locked_joint_names = ["panda_finger_joint1", "panda_finger_joint2"]
        params.urdf = (
            Path(__file__).resolve().parent / "resources" / "urdf" / "robot.urdf"
        )
        params.srdf = (
            Path(__file__).resolve().parent / "resources" / "urdf" / "demo.srdf"
        )
        env = Path(__file__).resolve().parent / "resources" / "param.yaml"

        # model1
        robot1 = RobotModel.load_model(params, env)

        # model2
        robot2 = RobotModelConstructor()
        robot2.set_robot_model(
            example_robot_data.load("panda"), str(params.urdf), str(params.srdf)
        )
        robot2.set_collision_model(str(params.urdf), str(env))

        self.assertEqual(
            robot1.get_reduced_robot_model(),
            robot2.get_robot_reduced_model(),
        )
        self.assertEqual(
            str(robot1.get_reduced_collision_model()),
            str(robot2.get_collision_reduced_model()),
        )


if __name__ == "__main__":
    unittest.main()
