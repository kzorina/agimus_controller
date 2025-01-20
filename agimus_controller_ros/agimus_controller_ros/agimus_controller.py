#!/usr/bin/env python3
import numpy as np
import numpy.typing as npt
import time
import pinocchio as pin
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.duration import Duration

from std_msgs.msg import String
from agimus_msgs.msg import MpcInput
import builtin_interfaces
import os
from ament_index_python.packages import get_package_share_directory

import linear_feedback_controller_msgs_py.lfc_py_types as lfc_py_types
from linear_feedback_controller_msgs_py.numpy_conversions import (
    sensor_msg_to_numpy,
    control_numpy_to_msg,
)
from linear_feedback_controller_msgs.msg import Control, Sensor

from agimus_controller.mpc import MPC
from agimus_controller.mpc_data import OCPResults
from agimus_controller.ocp.ocp_croco_joint_state import OCPCrocoJointState
from agimus_controller.ocp_param_base import OCPParamsBaseCroco
from agimus_controller.warm_start_reference import WarmStartReference
from agimus_controller.factory.robot_model import RobotModels, RobotModelParameters


from agimus_controller_ros.ros_utils import mpc_msg_to_weighted_traj_point
# from agimus_controller.mpc import MPC
# from agimus_controller.ocps.ocp_croco_hpp import OCPCrocoHPP

from agimus_controller.trajectory import TrajectoryBuffer, TrajectoryPoint
from agimus_controller_ros.agimus_controller_parameters import agimus_controller_params


class AgimusController(Node):
    """Agimus controller's ROS 2 node class."""

    def __init__(self, node_name: str = "agimus_controller_node") -> None:
        """Get ROS parameters, initialize trajectory buffer and ros attributes."""
        super().__init__(node_name)
        self.param_listener = agimus_controller_params.ParamListener(self)
        self.params = self.param_listener.get_params()
        self.params.ocp.armature = np.array(self.params.ocp.armature)
        self.traj_buffer = TrajectoryBuffer()
        self.last_point = None
        self.first_run_done = False
        self.rmodel = None
        self.mpc = None

        self.initialize_ros_attributes()
        self.get_logger().info("Init done")

    def initialize_ros_attributes(self) -> None:
        """Initialize ROS related attributes such as Publishers, Subscribers and Timers"""
        self.sensor_msg = None
        self.control_msg = None
        self.state_subscriber = self.create_subscription(
            Sensor,
            "sensor",
            self.sensor_callback,
            qos_profile=QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            ),
        )
        self.subscriber_robot_description = self.create_subscription(
            String,
            "/robot_description",
            self.robot_description_callback,
            qos_profile=QoSProfile(
                depth=1,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                reliability=ReliabilityPolicy.RELIABLE,
            ),
        )
        self.subscriber_mpc_input = self.create_subscription(
            MpcInput,
            "mpc_input",
            self.mpc_input_callback,
            qos_profile=10,
        )
        self.control_publisher = self.create_publisher(
            Control,
            "control",
            qos_profile=QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            ),
        )
        if self.params.publish_debug_data:
            self.ocp_solve_time_pub = self.create_publisher(
                builtin_interfaces.msg.Duration, "ocp_solve_time", 10
            )
            self.ocp_x0_pub = self.create_publisher(Sensor, "ocp_x0", 10)
        self.create_timer(1.0 / self.params.rate, self.run_callback)

    def setup_mpc(self):
        """Creates mpc, ocp, warmstart"""

        ocp_params = OCPParamsBaseCroco(
            dt=self.params.ocp.dt,
            horizon_size=self.params.ocp.horizon_size,
            solver_iters=self.params.ocp.max_iter,
            callbacks=self.params.ocp.activate_callback,
            qp_iters=self.params.ocp.max_qp_iter,
        )

        ocp = OCPCrocoJointState(self.robot_models, ocp_params)
        ws = WarmStartReference()
        ws.setup(self.robot_models._robot_model)
        self.mpc = MPC()
        self.mpc.setup(ocp, ws, self.traj_buffer)
    
    def sensor_callback(self, sensor_msg: Sensor) -> None:
        """Update the sensor_msg attribute of the class."""
        self.sensor_msg = sensor_msg

    def mpc_input_callback(self, msg: MpcInput) -> None:
        """Fill the new point msg in the trajectory buffer."""
        w_traj_point = mpc_msg_to_weighted_traj_point(msg, self.get_clock().now().nanoseconds)
        self.traj_buffer.append(w_traj_point)
        self.params.ocp.effector_frame_name = msg.ee_frame_name
        self.effector_frame_name = msg.ee_frame_name

    def robot_description_callback(self, msg: String) -> None:
        """Create the models of the robot from the urdf string."""
        
        # TODO: fix, just hardcoded the thing: should exist in the demo folder?
        # add as a ros parameter in the yaml file srdf_path
        temp_srdf_path = os.path.join(
                get_package_share_directory("franka_description"),
                "robots/fer",
                "fer.srdf",
            )
        params = RobotModelParameters(
            urdf_xml=msg.data,
            srdf_path=Path(temp_srdf_path),
            q0=np.zeros(7),  # TODO: change from hardcooding
            full_q0=np.zeros(7),  # TODO: change from hardcooding
            free_flyer=self.params.free_flyer,
            locked_joint_names=[],
            collision_as_capsule=self.params.collision_as_capsule,
            self_collision=self.params.self_collision,
            armature=self.params.ocp.armature,
        )

        self.robot_models = RobotModels(params)
        self.rmodel = self.robot_models._robot_model

        self.get_logger().info("Robot Models initialized")


    def buffer_has_twice_horizon_points(self) -> bool:
        """
        Return true if buffer size has more than two times
        the horizon size and False otherwise.
        """
        return len(self.traj_buffer) >= 2 * self.params.ocp.horizon_size

    def send_control_msg(self, ocp_res: OCPResults) -> None:
        """Get OCP control output and publish it."""
        # _, tau, K_ricatti = self.mpc.get_mpc_output()
        ctrl_msg = lfc_py_types.Control(
            feedback_gain=ocp_res.ricatti_gains[0],
            feedforward=ocp_res.feed_forward_terms[0].reshape(self.rmodel.nv, 1),
            initial_state=sensor_msg_to_numpy(self.sensor_msg),
        )
        self.control_publisher.publish(control_numpy_to_msg(ctrl_msg))

    def run_callback(self, *args) -> None:
        """
        Timer callback that checks we can start solve before doing it,
        then publish messages related to the OCP.
        """
        if self.sensor_msg is None:
            self.get_logger().warn(
                "Waiting for sensor messages to arrive...",
                throttle_duration_sec=5.0,
            )
            return
        if self.rmodel is None:
            self.get_logger().warn(
                "Waiting for robot model to arrive...",
                throttle_duration_sec=5.0,
            )
            return
        if self.mpc is None:
            self.setup_mpc()
        if not self.buffer_has_twice_horizon_points():
                self.get_logger().warn(
                    f"Waiting for buffer to be filled... Current size {len(self.traj_buffer)}",
                    throttle_duration_sec=5.0,
                )
                return
        start_compute_time = time.perf_counter()
        np_sensor_msg: lfc_py_types.Sensor = sensor_msg_to_numpy(self.sensor_msg)

        x0_traj_point = TrajectoryPoint(
            time_ns=self.get_clock().now().nanoseconds,
            robot_configuration=np_sensor_msg.joint_state.position,
            robot_velocity=np_sensor_msg.joint_state.velocity,
            )
        ocp_res =self.mpc.run(initial_state=x0_traj_point, current_time_ns=self.get_clock().now().nanoseconds)
        if ocp_res is None:
            return

        self.send_control_msg(ocp_res)
        compute_time = time.perf_counter() - start_compute_time
        # if self.params.publish_debug_data:
        #     self.ocp_solve_time_pub.publish(Duration(seconds=compute_time).to_msg())
        #     self.ocp_x0_pub.publish(self.sensor_msg)


def main(args=None) -> None:
    """Creates the Agimus controller ROS node object and spins it."""
    rclpy.init(args=args)
    agimus_controller_node = AgimusController()
    try:
        rclpy.spin(agimus_controller_node)
    except KeyboardInterrupt:
        pass
    agimus_controller_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
