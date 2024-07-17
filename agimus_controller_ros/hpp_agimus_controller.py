#!/usr/bin/env python3
import rospy
import numpy as np
from copy import deepcopy
import time
import os
from threading import Lock
from std_msgs.msg import Duration, Header
import example_robot_data
from linear_feedback_controller_msgs.msg import Control, Sensor

from agimus_controller.utils.ros_np_multiarray import to_multiarray_f64
from agimus_controller.utils.wrapper_panda import PandaWrapper
from agimus_controller.utils.build_models import get_robot_model, get_collision_model
from agimus_controller.hpp_interface import HppInterface
from agimus_controller.mpc import MPC
from agimus_controller.ocps.ocp_croco_hpp import OCPCrocoHPP


class HppAgimusController:
    def __init__(self) -> None:
        self.dt = 1e-2
        self.q_goal = [-0.8311, 0.6782, 0.3201, -1.1128, 1.2190, 1.9823, 0.7248]
        self.horizon_size = 100

        robot = example_robot_data.load("panda")

        current_dir_path = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(current_dir_path, "../urdf/robot.urdf")
        srdf_path = os.path.join(current_dir_path, "../srdf/demo.srdf")
        yaml_path = os.path.join(current_dir_path, "../config/param.yaml")
        self.rmodel = get_robot_model(robot, urdf_path, srdf_path)
        self.cmodel = get_collision_model(self.rmodel, urdf_path, yaml_path)
        self.pandawrapper = PandaWrapper(auto_col=True)
        self.ee_frame_name = self.pandawrapper.get_ee_frame_name()
        self.hpp_interface = HppInterface()
        self.armature = np.array([0.01] * self.rmodel.nq)
        self.ocp = OCPCrocoHPP(
            self.rmodel, self.cmodel, use_constraints=False, armature=self.armature
        )
        self.ocp.set_weights(10**4, 10, 10**-3, 0)
        self.mpc_iter = 0
        self.save_predictions_and_refs = False
        self.nb_mpc_iter_to_save = None
        self.mpc_data = {}

        self.rate = rospy.Rate(100, reset=True)
        self.mutex = Lock()
        self.sensor_msg = Sensor()
        self.control_msg = Control()
        self.ocp_solve_time = Duration()
        self.x0 = np.zeros(self.rmodel.nq + self.rmodel.nv)
        self.x_guess = np.zeros(self.rmodel.nq + self.rmodel.nv)
        self.u_guess = np.zeros(self.rmodel.nv)
        self.state_subscriber = rospy.Subscriber(
            "robot_sensors",
            Sensor,
            self.sensor_callback,
        )
        self.control_publisher = rospy.Publisher(
            "motion_server_control", Control, queue_size=1, tcp_nodelay=True
        )
        self.ocp_solve_time_pub = rospy.Publisher(
            "ocp_solve_time", Duration, queue_size=1, tcp_nodelay=True
        )
        self.start_time = 0.0
        self.first_solve = False
        self.first_robot_sensor_msg_received = False
        self.first_pose_ref_msg_received = True

    def sensor_callback(self, sensor_msg):
        with self.mutex:
            self.sensor_msg = deepcopy(sensor_msg)
            if not self.first_robot_sensor_msg_received:
                self.first_robot_sensor_msg_received = True

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

    def plan_and_first_solve(self):
        sensor_msg = self.get_sensor_msg()

        # Plan
        q_init = [*sensor_msg.joint_state.position]
        self.hpp_interface.set_panda_planning(q_init, self.q_goal)
        ps = self.hpp_interface.ps
        whole_x_plan, whole_a_plan, _ = self.hpp_interface.get_hpp_x_a_planning(
            self.dt,
            self.rmodel.nq,
            ps.client.problem.getPath(ps.numberPaths() - 1),
        )

        # First solve
        self.mpc = MPC(self.ocp, whole_x_plan, whole_a_plan, self.rmodel, self.cmodel)
        self.x_plan = self.mpc.whole_x_plan[: self.horizon_size, :]
        self.a_plan = self.mpc.whole_a_plan[: self.horizon_size, :]
        x0 = np.concatenate(
            [sensor_msg.joint_state.position, sensor_msg.joint_state.velocity]
        )
        self.mpc.mpc_first_step(self.x_plan, self.a_plan, x0, self.horizon_size)
        self.next_node_idx = self.horizon_size
        whole_traj_T = whole_x_plan.shape[0]
        if self.save_predictions_and_refs:
            self.nb_mpc_iter_to_save = whole_traj_T
            self.create_mpc_data()
            self.update_predictions_and_refs_arrays()
        self.mpc_iter += 1

        _, u, k = self.mpc.get_mpc_output()
        return sensor_msg, u, k

    def solve(self):
        sensor_msg = self.get_sensor_msg()
        x0 = np.concatenate(
            [sensor_msg.joint_state.position, sensor_msg.joint_state.velocity]
        )

        new_x_ref = self.mpc.whole_x_plan[self.next_node_idx, :]
        new_a_ref = self.mpc.whole_a_plan[self.next_node_idx, :]

        mpc_start_time = time.time()
        self.mpc.mpc_step(x0, new_x_ref, new_a_ref)
        mpc_duration = time.time() - mpc_start_time
        rospy.loginfo_throttle(1, "mpc_duration = %s", str(mpc_duration))
        if self.next_node_idx < self.mpc.whole_x_plan.shape[0] - 1:
            self.next_node_idx += 1
        if self.save_predictions_and_refs:
            self.update_predictions_and_refs_arrays()
        self.mpc_iter += 1
        _, u, k = self.mpc.get_mpc_output()

        return sensor_msg, u, k

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

    def update_predictions_and_refs_arrays(self):
        if self.mpc_iter < self.nb_mpc_iter_to_save:
            xs = self.mpc.ocp.solver.xs
            us = self.mpc.ocp.solver.us
            x_ref, p_ref, u_ref = self.mpc.get_reference()
            self.fill_predictions_and_refs_arrays(
                self.mpc_iter, xs, us, x_ref, p_ref, u_ref
            )
        if self.mpc_iter == self.nb_mpc_iter_to_save:
            np.save("mpc_data.npy", self.mpc_data)

    def create_mpc_data(self):
        self.mpc_data["preds_xs"] = np.zeros(
            [self.nb_mpc_iter_to_save, self.horizon_size, 2 * self.rmodel.nq]
        )
        self.mpc_data["preds_us"] = np.zeros(
            [self.nb_mpc_iter_to_save, self.horizon_size - 1, self.rmodel.nq]
        )
        self.mpc_data["state_refs"] = np.zeros(
            [self.nb_mpc_iter_to_save, 2 * self.rmodel.nq]
        )
        self.mpc_data["translation_refs"] = np.zeros([self.nb_mpc_iter_to_save, 3])
        self.mpc_data["control_refs"] = np.zeros(
            [self.nb_mpc_iter_to_save, self.rmodel.nq]
        )

    def fill_predictions_and_refs_arrays(self, idx, xs, us, x_ref, p_ref, u_ref):
        self.mpc_data["preds_xs"][idx, :, :] = xs
        self.mpc_data["preds_us"][idx, :, :] = us
        self.mpc_data["state_refs"][idx, :] = x_ref
        self.mpc_data["translation_refs"][idx, :] = p_ref
        self.mpc_data["control_refs"][idx, :] = u_ref

    def run(self):
        self.wait_first_sensor_msg()
        sensor_msg, u, k = self.plan_and_first_solve()
        input("Press enter to continue ...")
        self.send(sensor_msg, u, k)
        self.rate.sleep()
        while not rospy.is_shutdown():
            start_compute_time = rospy.Time.now()
            sensor_msg, u, k = self.solve()
            self.send(sensor_msg, u, k)
            self.rate.sleep()
            self.ocp_solve_time.data = rospy.Time.now() - start_compute_time
            self.ocp_solve_time_pub.publish(self.ocp_solve_time)


def crocco_motion_server_node():
    rospy.init_node("croccodyl_motion_server_node_py", anonymous=True)
    node = HppAgimusController()
    node.run()


if __name__ == "__main__":
    try:
        crocco_motion_server_node()
    except rospy.ROSInterruptException:
        pass
