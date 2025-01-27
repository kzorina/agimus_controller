import pinocchio as pin
import numpy as np
import numpy.typing as npt


from geometry_msgs.msg import Pose
from agimus_msgs.msg import MpcInput

from agimus_controller.trajectory import (
    TrajectoryPoint,
    TrajectoryPointWeights,
    WeightedTrajectoryPoint,
)


def ros_pose_to_array(pose: Pose) -> npt.NDArray[np.float64]:
    """Convert geometry_msgs.msg.Pose to a 7d numpy array"""
    return np.array(
        [
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ]
    )


def mpc_msg_to_weighted_traj_point(
    msg: MpcInput, time_ns: int
) -> WeightedTrajectoryPoint:
    """Build WeightedTrajectoryPoint object from MPCInput msg."""
    xyz_quat_pose = ros_pose_to_array(msg.pose)
    traj_point = TrajectoryPoint(
        time_ns=time_ns,
        robot_configuration=msg.q,
        robot_velocity=msg.qdot,
        robot_acceleration=msg.qddot,
        robot_effort=np.array(msg.robot_effort, dtype=np.float64),
        end_effector_poses={msg.ee_frame_name: pin.XYZQUATToSE3(xyz_quat_pose)},
    )

    traj_weights = TrajectoryPointWeights(
        w_robot_configuration=msg.w_q,
        w_robot_velocity=msg.w_qdot,
        w_robot_acceleration=msg.w_qddot,
        w_robot_effort=np.array(msg.w_robot_effort, dtype=np.float64),
        # w_robot_effort=np.ones(7) * 1e-5,
        w_end_effector_poses={msg.ee_frame_name: msg.w_pose},
    )

    return WeightedTrajectoryPoint(point=traj_point, weights=traj_weights)
