#!/usr/bin/env python3
import rospy
import numpy as np
from copy import deepcopy
import time
import pinocchio as pin
from enum import Enum
from threading import Lock
from std_msgs.msg import Duration, Header
from vision_msgs.msg import Detection2DArray
import tf2_ros
import tf2_geometry_msgs
from linear_feedback_controller_msgs.msg import Control, Sensor
import atexit

from agimus_controller_ros.ros_np_multiarray import to_multiarray_f64
from agimus_controller.utils.path_finder import get_package_path
from agimus_controller.trajectory_buffer import TrajectoryBuffer
from agimus_controller.trajectory_point import PointAttribute
from agimus_controller.robot_model.panda_model import (
    PandaRobotModel,
    PandaRobotModelParameters,
)
from agimus_controller.utils.pin_utils import (
    get_ee_pose_from_configuration,
)
from agimus_controller.mpc import MPC
from agimus_controller.ocps.ocp_croco_hpp import OCPCrocoHPP
from agimus_controller_ros.sim_utils import convert_float_to_ros_duration_msg


class HPPStateMachine(Enum):
    WAITING_PICK_TRAJECTORY = 1
    RECEIVING_PICK_TRAJECTORY = 2
    WAITING_PLACE_TRAJECTORY = 3
    RECEIVING_PLACE_TRAJECTORY = 4
    WAITING_GOING_INIT_POSE_TRAJECTORY = 5
    RECEIVING_GOING_INIT_POSE_TRAJECTORY = 6


class AgimusControllerNodeParameters:
    def __init__(self) -> None:
        self.save_predictions_and_refs = rospy.get_param(
            "save_predictions_and_refs", False
        )
        self.dt = rospy.get_param("dt", 0.01)
        self.rate = rospy.get_param("rate", 100)
        self.horizon_size = rospy.get_param("horizon_size", 100)
        self.armature = np.array(
            rospy.get_param("armature", [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
        )
        self.gripper_weight = rospy.get_param("gripper_weight", 10000)
        self.state_weight = rospy.get_param("state_weight", 10)
        self.control_weight = rospy.get_param("control_weight", 0.001)
        self.use_constraints = rospy.get_param("use_constraints", False)
        self.use_vision = rospy.get_param("use_vision", False)


class ControllerBase:
    def __init__(self) -> None:
        self.params = AgimusControllerNodeParameters()
        self.traj_buffer = TrajectoryBuffer()
        self.point_attributes = [PointAttribute.Q, PointAttribute.V, PointAttribute.A]
        self.elapsed_time = None
        self.last_elapsed_time = None
        self.state_machine_timeline = [
            HPPStateMachine.WAITING_PICK_TRAJECTORY,
            HPPStateMachine.RECEIVING_PICK_TRAJECTORY,
            HPPStateMachine.WAITING_PLACE_TRAJECTORY,
            HPPStateMachine.RECEIVING_PLACE_TRAJECTORY,
            HPPStateMachine.WAITING_GOING_INIT_POSE_TRAJECTORY,
            HPPStateMachine.RECEIVING_GOING_INIT_POSE_TRAJECTORY,
        ]
        self.state_machine_timeline_idx = 0
        self.state_machine = self.state_machine_timeline[
            self.state_machine_timeline_idx
        ]

        robot_params = PandaRobotModelParameters()
        robot_params.collision_as_capsule = True
        robot_params.self_collision = False
        agimus_demos_description_dir = get_package_path("agimus_demos_description")
        collision_file_path = (
            agimus_demos_description_dir / "pick_and_place" / "obstacle_params.yaml"
        )
        robot_constructor = PandaRobotModel.load_model(
            params=robot_params, env=collision_file_path
        )

        self.rmodel = robot_constructor.get_reduced_robot_model()
        self.cmodel = robot_constructor.get_reduced_collision_model()

        self.rdata = self.rmodel.createData()
        self.effector_frame_id = self.rmodel.getFrameId("panda_hand_tcp")
        self.nq = self.rmodel.nq
        self.nv = self.rmodel.nv
        self.nx = self.nq + self.nv
        self.ocp = OCPCrocoHPP(
            self.rmodel,
            self.cmodel,
            use_constraints=self.params.use_constraints,
            armature=self.params.armature,
        )
        self.ocp.set_weights(
            self.params.gripper_weight,
            self.params.state_weight,
            self.params.control_weight,
            0,
        )
        self.mpc_data = {}

        self.rate = rospy.Rate(self.params.rate, reset=True)
        self.mutex = Lock()
        self.sensor_msg = Sensor()
        self.control_msg = Control()
        self.ocp_solve_time = Duration()
        self.x0 = np.zeros(self.nq + self.nv)
        self.x_guess = np.zeros(self.nq + self.nv)
        self.u_guess = np.zeros(self.nv)
        self.state_subscriber = rospy.Subscriber(
            "robot_sensors", Sensor, self.sensor_callback
        )

        self.control_publisher = rospy.Publisher(
            "motion_server_control", Control, queue_size=1, tcp_nodelay=True
        )
        self.ocp_solve_time_pub = rospy.Publisher(
            "ocp_solve_time", Duration, queue_size=1, tcp_nodelay=True
        )

        self.init_in_world_M_object = None
        self.in_world_M_object = None
        self.target_translation_object_to_effector = None
        self.in_world_M_effector = None
        self.start_time = 0.0
        self.first_robot_sensor_msg_received = False
        self.first_pose_ref_msg_received = True
        self.last_point = None
        if self.params.use_vision:
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
            # to avoid errors when looking at transforms too early in time
            time.sleep(0.5)
            self.vision_subscriber = rospy.Subscriber(
                "/happypose/detections", Detection2DArray, self.vision_callback
            )

    def get_transform_between_frames(self, parent_frame, child_frame, timestamp):
        return self.tf_buffer.lookup_transform(
            target_frame=parent_frame, source_frame=child_frame, time=timestamp
        )

    def sensor_callback(self, sensor_msg):
        with self.mutex:
            self.sensor_msg = deepcopy(sensor_msg)
            if not self.first_robot_sensor_msg_received:
                self.first_robot_sensor_msg_received = True

    def vision_callback(self, vision_msg: Detection2DArray):
        if vision_msg.detections == []:
            return
        in_camera_pose_object = vision_msg.detections[0].results[0].pose
        image_timestamp = vision_msg.detections[0].header.stamp
        in_world_M_camera = self.get_transform_between_frames(
            "world", "camera_color_optical_frame", image_timestamp
        )
        in_world_pose_object = tf2_geometry_msgs.do_transform_pose(
            in_camera_pose_object, in_world_M_camera
        )
        pose = in_world_pose_object.pose
        trans = pose.position
        rot = pose.orientation
        pose_array = [trans.x, trans.y, trans.z, rot.w, rot.x, rot.y, rot.z]
        self.in_world_M_object = pin.XYZQUATToSE3(pose_array)
        if self.init_in_world_M_object is None:
            self.init_in_world_M_object = self.in_world_M_object
            print("Initialized object pose")

    def wait_first_sensor_msg(self):
        wait_for_input = True
        while not rospy.is_shutdown() and wait_for_input:
            wait_for_input = (
                not self.first_robot_sensor_msg_received
                or not self.first_pose_ref_msg_received
            )
            if wait_for_input:
                rospy.loginfo_throttle(3, "Waiting until we receive a sensor message.")
                with self.mutex:
                    sensor_msg = deepcopy(self.sensor_msg)
                    self.start_time = sensor_msg.header.stamp.to_sec()
            rospy.loginfo_once("Start controller")
            self.rate.sleep()
        return wait_for_input

    def wait_buffer_has_twice_horizon_points(self):
        while (
            self.traj_buffer.get_size(self.point_attributes)
            < 2 * self.params.horizon_size
        ):
            self.fill_buffer()

    def get_next_trajectory_point(self):
        raise RuntimeError("Not implemented")

    def update_state_machine(self):
        raise RuntimeError("Not implemented")

    def fill_buffer(self):
        point = self.get_next_trajectory_point()
        if point is not None:
            self.traj_buffer.add_trajectory_point(point)

    def first_solve(self):
        sensor_msg = self.get_sensor_msg()
        # retrieve horizon state and acc references
        horizon_points = self.traj_buffer.get_points(
            self.params.horizon_size, self.point_attributes
        )
        x_plan = np.zeros([self.params.horizon_size, self.nx])
        a_plan = np.zeros([self.params.horizon_size, self.nv])
        for idx_point, point in enumerate(horizon_points):
            x_plan[idx_point, :] = point.get_x_as_q_v()
            a_plan[idx_point, :] = point.a

        # First solve
        self.mpc = MPC(self.ocp, x_plan, a_plan, self.rmodel, self.cmodel)
        x0 = np.concatenate(
            [sensor_msg.joint_state.position, sensor_msg.joint_state.velocity]
        )
        self.mpc.mpc_first_step(x_plan, a_plan, x0, self.params.horizon_size)
        self.next_node_idx = self.params.horizon_size
        if self.params.save_predictions_and_refs:
            self.create_mpc_data()
        self.mpc_data["hpp_trajectory"].append(list(x_plan))
        _, u, k = self.mpc.get_mpc_output()
        return sensor_msg, u, k

    def solve(self):
        sensor_msg = self.get_sensor_msg()
        x0 = np.concatenate(
            [sensor_msg.joint_state.position, sensor_msg.joint_state.velocity]
        )
        self.target_translation_object_to_effector = None
        self.in_world_M_effector = None
        self.fill_buffer()
        if self.traj_buffer.get_size(self.point_attributes) > 0:
            point = self.traj_buffer.get_points(1, self.point_attributes)[0]
            self.last_point = point
            new_x_ref = point.get_x_as_q_v()
            self.mpc_data["hpp_trajectory"].append(new_x_ref)

        else:
            point = self.last_point
            new_x_ref = point.get_x_as_q_v()
        new_a_ref = point.a
        self.in_world_M_effector = get_ee_pose_from_configuration(
            self.rmodel,
            self.rdata,
            self.effector_frame_id,
            new_x_ref[: self.rmodel.nq],
        )
        # if last point of the pick trajectory is in horizon and we wanna use vision pose
        if (
            self.state_machine == HPPStateMachine.WAITING_PLACE_TRAJECTORY
            and self.params.use_vision
            and self.traj_buffer.get_size(self.point_attributes) == 0
        ):
            self.compute_new_placement(new_x_ref)
        self.mpc.mpc_step(x0, new_x_ref, new_a_ref, self.in_world_M_effector)

        if self.next_node_idx < self.mpc.whole_x_plan.shape[0] - 1:
            self.next_node_idx += 1

        if self.params.save_predictions_and_refs:
            self.fill_predictions_and_refs_arrays()
        _, u, k = self.mpc.get_mpc_output()

        return sensor_msg, u, k

    def compute_new_placement(self, hpp_x_goal):
        self.in_world_M_effector = get_ee_pose_from_configuration(
            self.rmodel,
            self.rdata,
            self.effector_frame_id,
            hpp_x_goal[: self.rmodel.nq],
        )
        in_object_rot_effector = (
            self.init_in_world_M_object.rotation.T * self.in_world_M_effector.rotation
        )
        self.target_translation_object_to_effector = (
            self.in_world_M_effector.translation
            - self.init_in_world_M_object.translation
        )
        self.in_world_M_effector.translation = (
            self.in_world_M_object.translation
            + self.target_translation_object_to_effector
        )
        self.in_world_M_effector.rotation = (
            self.in_world_M_object.rotation * in_object_rot_effector
        )

    def get_sensor_msg(self):
        with self.mutex:
            sensor_msg = deepcopy(self.sensor_msg)
        return sensor_msg

    def send(self, sensor_msg, u, k):
        self.control_msg.header = Header()
        self.control_msg.header.stamp = rospy.Time.now()
        self.control_msg.feedback_gain = to_multiarray_f64(k)
        self.control_msg.feedforward = to_multiarray_f64(u)
        self.control_msg.initial_state = sensor_msg
        self.control_publisher.publish(self.control_msg)

    def create_mpc_data(self):
        xs, us = self.mpc.get_predictions()
        x_ref, p_ref, u_ref = self.mpc.get_reference()

        self.mpc_data["preds_xs"] = [xs]
        self.mpc_data["preds_us"] = [us]
        self.mpc_data["state_refs"] = [x_ref]
        self.mpc_data["translation_refs"] = [p_ref]
        self.mpc_data["control_refs"] = [u_ref]
        self.mpc_data["state"] = [self.state_machine.value]
        if self.params.use_constraints:
            collision_residuals = self.mpc.get_collision_residuals()
            self.mpc_data["coll_residuals"] = collision_residuals
        self.mpc_data["hpp_trajectory"] = []
        if self.in_world_M_object is not None:
            self.mpc_data["vision_refs"] = [
                np.array(self.in_world_M_object.translation)
            ]
            self.mpc_data["obj_trans_ee"] = []

    def fill_predictions_and_refs_arrays(self):
        xs, us = self.mpc.get_predictions()
        x_ref, p_ref, u_ref = self.mpc.get_reference()

        self.mpc_data["preds_xs"].append(xs)
        self.mpc_data["preds_us"].append(us)
        self.mpc_data["state_refs"].append(x_ref)
        self.mpc_data["translation_refs"].append(p_ref)
        self.mpc_data["control_refs"].append(u_ref)
        self.mpc_data["state"].append(self.state_machine.value)
        if self.init_in_world_M_object is not None:
            self.mpc_data["init_in_world_M_object"] = self.init_in_world_M_object
        if self.params.use_constraints:
            collision_residuals = self.mpc.get_collision_residuals()
            for coll_residual_key in collision_residuals.keys():
                self.mpc_data["coll_residuals"][coll_residual_key] += (
                    collision_residuals[coll_residual_key]
                )

        if self.target_translation_object_to_effector is not None:
            self.mpc_data["obj_trans_ee"].append(
                self.target_translation_object_to_effector
            )
        # if self.in_world_M_effector is not None:
        # self.mpc_data["ee_target_pose"].append(self.in_world_M_effector)

        if "vision_refs" in self.mpc_data.keys():
            self.mpc_data["vision_refs"].append(
                np.array(self.in_world_M_object.translation)
            )

    def exit_handler(self):
        print("saving data")
        np.save("mpc_data.npy", self.mpc_data)

    def run(self):
        self.wait_first_sensor_msg()
        self.wait_buffer_has_twice_horizon_points()
        sensor_msg, u, k = self.first_solve()
        input("Press enter to continue ...")
        self.send(sensor_msg, u, k)
        self.rate.sleep()
        if self.params.save_predictions_and_refs:
            atexit.register(self.exit_handler)
        while not rospy.is_shutdown():
            start_compute_time = time.time()
            self.update_state_machine()
            sensor_msg, u, k = self.solve()
            self.send(sensor_msg, u, k)
            compute_time = time.time() - start_compute_time
            self.rate.sleep()
            self.ocp_solve_time = convert_float_to_ros_duration_msg(compute_time)
            self.ocp_solve_time_pub.publish(self.ocp_solve_time)
