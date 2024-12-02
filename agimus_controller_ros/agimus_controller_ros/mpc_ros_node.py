#! /usr/bin/env python3

from agimus_msgs.msg import MpcInput
from geometry_msgs.msg import Pose
from std_msgs.msg import String
from rclpy.node import Node
import rclpy
import pinocchio as pin
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
import numpy as np


class AgimusControllerNode(Node):
    def __init__(self):
        super().__init__("agimus_controller_node")

        self.extract_ros_parameters()
        self.subscriber_robot_description_ = self.create_subscription(
            String,
            "/robot_description",
            self.robot_description_callback,
            qos_profile=QoSProfile(
                depth=1,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                reliability=ReliabilityPolicy.RELIABLE,
            ),
        )

        self.timer = self.create_timer(self.controller_period, self.run_mpc)
        self.get_logger().info("agimus_controller_node started.")

    def extract_ros_parameters(self):
        self.declare_parameter("moving_joint_names", [""])
        self.declare_parameter("controller_period", 0.01)

        self.moving_joint_names = (
            self.get_parameter("moving_joint_names")
            .get_parameter_value()
            .string_array_value
        )
        self.controller_period = (
            self.get_parameter("controller_period").get_parameter_value().double_value
        )

    def robot_description_callback(self, msg: String):
        """Callback to get robot description and store to object"""
        self._pin_model = pin.buildModelFromXML(msg.data)
        self._pin_data = self._pin_model.createData()
        self._ee_frame_id = self._pin_model.getFrameId(self.ee_frame_name)
        self.get_logger().warn(f"Model loaded, pin_model.nq = {self._pin_model.nq}")

    def run_mpc(self):
        """
        Main function running the mpc and publishing the trajectories at 100Hz.
        """
        if self._pin_model is None:  # wait for model to be available
            return

        # Currently not changing the last two joints - fingers
        # TODO: change once we have a finger flag
        for i in range(self._pin_model.nq - 2):
            self.q[i] = self.q0[i] + 0.2 * np.sin(2 * np.pi * self.t)

        # Extract the end-effector position and orientation
        pin.forwardKinematics(self._pin_model, self._pin_data, self.q)
        pin.updateFramePlacement(self._pin_model, self._pin_data, self._ee_frame_id)

        ee_pose = self._pin_data.oMf[self._ee_frame_id]
        xyz_quatxyzw = pin.SE3ToXYZQUAT(ee_pose)

        # Create the message
        msg = MpcInput()
        msg.q = [float(val) for val in self.q]

        msg.qdot = [0.0] * len(
            self.q
        )  # TODO: only works for robot with only revolute joints
        msg.qddot = [0.0] * len(
            self.q
        )  # TODO: only works for robot with only revolute joints
        msg.q_w = [1.0] * len(self.q)
        msg.qdot_w = [1e-3] * len(self.q)
        msg.qddot_w = [1e-3] * len(self.q)

        pose = Pose()
        pose.position.x = xyz_quatxyzw[0]
        pose.position.y = xyz_quatxyzw[1]
        pose.position.z = xyz_quatxyzw[2]
        pose.orientation.x = xyz_quatxyzw[3]
        pose.orientation.y = xyz_quatxyzw[4]
        pose.orientation.z = xyz_quatxyzw[5]
        pose.orientation.w = xyz_quatxyzw[6]
        msg.pose = pose
        msg.pose_w = [1.0] * 6

        msg.ee_frame_name = self.ee_frame_name

        self.publisher_.publish(msg)
        # self.get_logger().info(f'Published MPC Input: {msg}')
        self.t += self.dt


def main(args=None):
    rclpy.init(args=args)
    node = MpcInputDummyPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
