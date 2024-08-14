import unittest
import pinocchio as pin
from agimus_controller.robot_model.robot_model import RobotModel
from agimus_controller.robot_model.panda_model import PandaRobotModel


class TestBuildModel(unittest.TestCase):
    def test_constructor(self):
        robot_model = RobotModel()

        self.assertEqual(robot_model.get_complete_robot_model(), pin.Model())
        self.assertEqual(robot_model.get_complete_collision_model(), pin.Model())
        self.assertEqual(robot_model.get_complete_visual_model(), pin.Model())
        self.assertEqual(robot_model.get_reduced_robot_model(), pin.Model())
        self.assertEqual(robot_model.get_reduced_collision_model(), pin.Model())
        self.assertEqual(robot_model.get_reduced_visual_model(), pin.Model())
        self.assertEqual(robot_model.get_default_configuration().size, 0)

    def test_load_panda_model(self):
        robot_model = PandaRobotModel.load_model()
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
        self.assertEqual(m.existJointName("panda_hand_joint"))
        self.assertEqual(m.existJointName("panda_camera_joint"))


if __name__ == "__main__":
    unittest.main()
