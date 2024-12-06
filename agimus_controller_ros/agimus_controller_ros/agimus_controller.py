#!/usr/bin/env python3
import atexit
from copy import deepcopy
import numpy as np
import time
import pinocchio as pin

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from std_msgs.msg import Header, String
from agimus_msgs.msg import MpcInput
from builtin_interfaces.msg import Duration

from linear_feedback_controller_msgs.msg import Control, Sensor
from linear_feedback_controller_msgs_py.numpy_conversions import matrix_numpy_to_msg

from agimus_controller.utils.pin_utils import get_ee_pose_from_configuration
from agimus_controller.mpc import MPC
from agimus_controller.ocps.ocp_croco_hpp import OCPCrocoHPP
from agimus_controller_ros.sim_utils import convert_float_to_ros_duration_msg
from agimus_controller_ros.parameters import AgimusControllerNodeParameters
from agimus_controller.trajectory import (
    TrajectoryBuffer,
    TrajectoryPoint,
    TrajectoryPointWeights,
    WeightedTrajectoryPoint
)


class AgimusController(Node):
    def __init__(
        self,
        params: AgimusControllerNodeParameters,
        node_name: str = "agimus_controller_node",
    ) -> None:
        super().__init__(node_name)
        self.params = params
        self.traj_buffer = TrajectoryBuffer()
        self.last_point = None

        self.initialize_ros_attributes()
        self.get_logger().info("Init done")

    def initialize_ros_attributes(self):
        self.sensor_msg = Sensor()
        self.control_msg = Control()
        qos_profile = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.state_subscriber = self.create_subscription(
            Sensor, "robot_sensors", self.sensor_callback, qos_profile=qos_profile
        )
        self.subscriber_robot_description = self.create_subscription(
            String,
            "/robot_description",
            self.robot_description_callback,
            qos_profile=qos_profile,
        )
        self.subscriber_mpc_input = self.create_subscription(
            MpcInput,
            "/mpc_input",
            self.mpc_input_callback,
            qos_profile=qos_profile,
        )

        self.control_publisher = self.create_publisher(
            Control, "motion_server_control", 10
        )

        self.ocp_solve_time_pub = self.create_publisher(Duration, "ocp_solve_time", 10)

        self.ocp_x0_pub = self.create_publisher(Sensor, "ocp_x0", 10)
        self.first_robot_sensor_msg_received = False
        self.first_pose_ref_msg_received = True

    def sensor_callback(self, sensor_msg):
        self.sensor_msg = deepcopy(sensor_msg)
        if not self.first_robot_sensor_msg_received:
            self.first_robot_sensor_msg_received = True

    def mpc_input_callback(self, msg: MpcInput):
        xyz_quat = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ])
        traj_point = TrajectoryPoint(
            robot_configuration=msg.q,
            robot_velocity=msg.qdot,
            robot_acceleration=msg.qddot,
            end_effector_poses={
                msg.ee_frame_name: pin.XYZQUATToSE3(xyz_quat)
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

    def robot_description_callback(self, msg: String):
        self.rmodel = pin.buildModelFromXML(msg.data)
        locked_joint_names = [
            name
            for name in self.rmodel.names
            if name not in self.moving_joint_names and name != "universe"
        ]
        locked_joint_ids = [self.rmodel.getJointId(name) for name in locked_joint_names]
        self.rmodel = pin.buildReducedModel(
            self.rmodel,
            list_of_geom_models=[],
            list_of_joints_to_lock=locked_joint_ids,
            reference_configuration=np.zeros(self.rmodel.nq),
        )[0]
        self.rdata = self.rmodel.createData()
        self.effector_frame_id = self.rmodel.getFrameId(
            self.params.ocp.effector_frame_name
        )
        self.nq = self.rmodel.nq
        self.nv = self.rmodel.nv
        self.nx = self.nq + self.nv
        self.get_logger().info("robot_description_callback")
        self.get_logger().info(f"pin_model.nq {self.rmodel.nq}")

    def wait_first_sensor_msg(self):
        wait_for_input = True
        while not rclpy.is_shutdown() and wait_for_input:
            wait_for_input = (
                not self.first_robot_sensor_msg_received
                or not self.first_pose_ref_msg_received
            )
            if wait_for_input:
                rclpy.get_logger().info(3, "Waiting until we receive a sensor message.")
            rclpy.get_logger().info("Start controller")
            self.rate.sleep()
        return wait_for_input

    def wait_buffer_has_twice_horizon_points(self):
        while (
            len(self.traj_buffer) < 2 * self.params.ocp.horizon_size
        ):
            pass


    def first_solve(self, x0):

        # retrieve horizon state and acc references
        x_plan = np.zeros([self.params.ocp.horizon_size, self.nx])
        a_plan = np.zeros([self.params.ocp.horizon_size, self.nv])
        for idx_point in enumerate(self.params.ocp.horizon_size):
            traj_point = self.traj_buffer.popleft()
            x_plan[idx_point, :] = np.concatenate([
                traj_point.robot_configuration,
                traj_point.robot_velocity
            ])
            a_plan[idx_point, :] = traj_point.robot_acceleration

        # First solve
        self.ocp = OCPCrocoHPP(self.rmodel, self.cmodel, self.params.ocp)
        self.ocp.set_planning_variables(x_plan, a_plan)
        self.mpc = MPC(self.ocp, x_plan, a_plan, self.rmodel, self.cmodel)
        us_init = self.mpc.ocp.u_plan[: self.params.ocp.horizon_size - 1]
        self.mpc.mpc_first_step(x_plan, us_init, x0)
        self.next_node_idx = self.params.ocp.horizon_size
        if self.params.save_predictions_and_refs:
            self.mpc.create_mpc_data()
        _, u, k = self.mpc.get_mpc_output()
        return u, k

    def solve(self, x0):
        self.target_translation_object_to_effector = None
        self.fill_buffer()
        if len(self.traj_buffer) > 0:
            traj_point = self.traj_buffer.popleft()
            self.last_point = traj_point
            new_x_ref = np.concatenate([
                traj_point.robot_configuration,
                traj_point.robot_velocity
            ])
        else:
            traj_point = self.last_point
            new_x_ref = np.concatenate([
                traj_point.robot_configuration,
                traj_point.robot_velocity
            ])
        new_a_ref = traj_point.robot_acceleration
        self.in_world_M_effector = get_ee_pose_from_configuration(
            self.rmodel,
            self.rdata,
            self.effector_frame_id,
            new_x_ref[: self.rmodel.nq],
        )
        self.mpc.mpc_step(x0, new_x_ref, new_a_ref, self.in_world_M_effector)

        if self.params.save_predictions_and_refs:
            self.mpc.fill_predictions_and_refs_arrays()
        _, u, k = self.mpc.get_mpc_output()
        return u, k

    def get_sensor_msg(self):
        sensor_msg = deepcopy(self.sensor_msg)
        return sensor_msg

    def send(self, sensor_msg, u, k):
        self.control_msg.header = Header()
        self.control_msg.header.stamp = Time.time()
        self.control_msg.feedback_gain = matrix_numpy_to_msg(k)
        self.control_msg.feedforward = matrix_numpy_to_msg(u)
        self.control_msg.initial_state = sensor_msg
        self.control_publisher.publish(self.control_msg)

    def exit_handler(self):
        print("saving data")
        np.save("mpc_params.npy", self.params.get_dict())
        if self.params.save_predictions_and_refs:
            np.save("mpc_data.npy", self.mpc.mpc_data)

    def get_x0_from_sensor_msg(self, sensor_msg):
        return np.concatenate(
            [sensor_msg.joint_state.position, sensor_msg.joint_state.velocity]
        )

    def publish_ocp_solve_time(self, ocp_solve_time):
        self.ocp_solve_time_pub.publish(
            convert_float_to_ros_duration_msg(ocp_solve_time)
        )

    def run(self):
        self.wait_first_sensor_msg()
        self.wait_buffer_has_twice_horizon_points()
        sensor_msg = self.get_sensor_msg()
        start_compute_time = time.time()
        u, k = self.first_solve(self.get_x0_from_sensor_msg(sensor_msg))
        compute_time = time.time() - start_compute_time
        self.send(sensor_msg, u, k)
        self.publish_ocp_solve_time(compute_time)
        self.ocp_x0_pub.publish(sensor_msg)
        self.rate.sleep()
        atexit.register(self.exit_handler)
        self.create_timer(self.params.ocp.dt, self.run_callback)

    def run_callback(self, *args):
        start_compute_time = time.time()
        sensor_msg = self.get_sensor_msg()
        u, k = self.solve(self.get_x0_from_sensor_msg(sensor_msg))
        self.send(sensor_msg, u, k)
        compute_time = time.time() - start_compute_time
        self.publish_ocp_solve_time(compute_time)
        self.ocp_x0_pub.publish(sensor_msg)


def main(args=None):
    rclpy.init(args=args)
    params = AgimusControllerNodeParameters()
    params.set_parameters_from_ros()
    node = AgimusController(params)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
