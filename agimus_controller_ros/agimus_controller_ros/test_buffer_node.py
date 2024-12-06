from agimus_msgs.msg import MpcInput
from geometry_msgs.msg import Pose
# from std_msgs.msg import String
from rclpy.node import Node
import rclpy
# import pinocchio as pin
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
# import numpy as np

import numpy as np
from collections import deque
from dataclasses import dataclass
from pinocchio import SE3, Force, XYZQUATToSE3

def ros_pose_to_array(pose: Pose):
    """Convert geometry_msgs.msg.Pose to a 7d numpy array"""
    return np.array([
        pose.position.x,
        pose.position.y,
        pose.position.z,
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ])

@dataclass
class TrajectoryPoint:
    """Trajectory point aiming at being a reference for the MPC."""

    time_ns: int = None
    robot_configuration: np.ndarray = None
    robot_velocity: np.ndarray = None
    robot_acceleration: np.ndarray = None
    robot_effort: np.ndarray = None
    forces: dict[Force]  = None # Dictionary of pinocchio.Force
    end_effector_poses: dict[SE3]  = None # Dictionary of pinocchio.SE3


@dataclass
class TrajectoryPointWeights:
    """Trajectory point weights aiming at being set in the MPC costs."""

    w_robot_configuration: np.ndarray = None
    w_robot_velocity: np.ndarray = None
    w_robot_acceleration: np.ndarray = None
    w_robot_effort: np.ndarray = None
    w_forces: dict[np.ndarray] = None
    w_end_effector_poses: dict[np.ndarray] = None


@dataclass
class WeightedTrajectoryPoint:
    """Trajectory point and it's corresponding weights."""

    point: TrajectoryPoint
    weights: TrajectoryPointWeights


class TrajectoryBuffer(deque):
    """List of variable size in which the HPP trajectory nodes will be."""

    def clear_past(self, current_time_ns):
        while self and self[0].point.time_ns < current_time_ns:
            self.popleft()


class TestBufferNode(Node):
    def __init__(self):
        super().__init__("mpc_input_dummy_publisher")
        qos_profile = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.subscriber_mpc_input = self.create_subscription(
            MpcInput, "/mpc_input", self.mpc_input_callback, qos_profile=qos_profile
        )
        self.traj_buffer = TrajectoryBuffer()
        self.index = 0

    def mpc_input_callback(self, msg: MpcInput):
        traj_point = TrajectoryPoint(
            robot_configuration=msg.q,
            robot_velocity=msg.qdot,
            robot_acceleration=msg.qddot,
            end_effector_poses={
                msg.ee_frame_name: XYZQUATToSE3(ros_pose_to_array(msg.pose ))
            }
        )

        traj_weights = TrajectoryPointWeights(
            w_robot_configuration=msg.q_w,
            w_robot_velocity=msg.qdot_w,
            w_robot_acceleration=msg.qddot_w,
            w_end_effector_poses=msg.pose_w
        )

        w_traj_point = WeightedTrajectoryPoint(
            point=traj_point, 
            weights=traj_weights
        )

        self.traj_buffer.append(w_traj_point)
        print(f"Current buffer len: {len(self.traj_buffer)}")

def main(args=None):
    rclpy.init(args=args)
    node = TestBufferNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()