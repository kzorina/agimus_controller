import unittest
from pathlib import Path
import pinocchio as pin
from agimus_controller.robot_model.robot_model import RobotModel
from agimus_controller.robot_model.panda_model import PandaRobotModel
from agimus_controller.robot_model.panda_model import PandaRobotModelParameters
from agimus_controller.robot_model.ur3_model import UR3RobotModel


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
        self.assertFalse(robot_model.get_reduced_collision_model().collisionPairs)
        self.assertEqual(robot_model.get_default_configuration().size, m.nq)

    def test_load_panda_self_collision(self):
        robot_params = PandaRobotModelParameters()
        robot_params.self_collision = True
        robot_params.collision_as_capsule = False
        robot_model = PandaRobotModel.load_model(params=robot_params)

        cmodel = robot_model.get_reduced_collision_model()
        for col in cmodel.collisionPairs:
            first = cmodel.geometryObjects[col.first].name.rsplit("_", 1)[0]
            second = cmodel.geometryObjects[col.second].name.rsplit("_", 1)[0]
            self.assertTrue("_sc" in first)
            self.assertTrue("_sc" in second)

    def test_load_panda_collision_as_capsule(self):
        robot_params = PandaRobotModelParameters()
        robot_params.self_collision = False
        robot_params.collision_as_capsule = True
        robot_model = PandaRobotModel.load_model(params=robot_params)
        cmodel = robot_model.get_reduced_collision_model()
        self.assertEqual(len(cmodel.collisionPairs), 0)

    def test_load_panda_collision_as_capsule_and_self_collision(self):
        robot_params = PandaRobotModelParameters()
        robot_params.collision_as_capsule = True
        robot_params.self_collision = True
        PandaRobotModel.load_model(params=robot_params)

    def test_load_panda_collisions(self):
        robot_params = PandaRobotModelParameters()
        env = Path(__file__).resolve().parent / "resources" / "col_env.yaml"
        PandaRobotModel.load_model(env=env, params=robot_params)

    def test_load_ur3_model(self):
        robot_model = UR3RobotModel.load_model()
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
        self.assertEqual(m.name, "ur3")
        self.assertEqual(robot_model.get_default_configuration().size, m.nq)


if __name__ == "__main__":
    unittest.main()
