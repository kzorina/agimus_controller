#!/usr/bin/env python3
import rospy
import numpy as np
import pinocchio as pin
from copy import deepcopy
from threading import Lock
from std_msgs.msg import Duration, Header
from linear_feedback_controller_msgs.msg import Control, Sensor

from agimus_controller.utils.ros_np_multiarray import to_multiarray_f64
from agimus_controller.utils.wrapper_panda import PandaWrapper
from agimus_controller.hpp_interface import HppInterface
from agimus_controller.mpc import MPC
from agimus_controller.ocps.ocp_croco_hpp import OCPCrocoHPP


class HppAgimusController:
    def __init__(self) -> None:
        self.dt = 1e-2
        self.pandawrapper = PandaWrapper()
        self.rmodel, self.cmodel, self.vmodel = self.pandawrapper.create_robot()
        self.ee_frame_name = self.pandawrapper.get_ee_frame_name()
        self.hpp_interface = HppInterface()
        self.ps = self.hpp_interface.get_panda_planner()
        self.ocp = OCPCrocoHPP("panda")
        self.ocp.set_weights(10**4, 1, 10**-3, 0)

        self.rate = rospy.Rate(100)
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
            "motion_server_control", Control, queue_size=1
        )
        self.ocp_solve_time_pub = rospy.Publisher("ocp_solve_time", Duration, 1)
        self.start_time = 0.0
        self.first_solve = False
        self.first_robot_sensor_msg_received = False
        self.first_pose_ref_msg_received = True

        self.horizon_size = 100
        self.nb_iteration = 0

    def sensor_callback(self, sensor_msg):
        with self.mutex:
            self.sensor_msg = deepcopy(sensor_msg)
            if not self.first_robot_sensor_msg_received:
                self.first_robot_sensor_msg_received = True

    def warm_start(self, sensor_msg):
        if self.first_solve:
            self.x_guess[:] = np.concatenate(
                sensor_msg.joint_state.position,
                np.zeros(len(sensor_msg.joint_state.velocity)),
            )
            self.u_guess[:] = pin.computeGeneralizedGravity(
                self.rmodel, self.robot_data, sensor_msg.joint_state.position
            )
            xs = [np.array(x) for x in self.x_guess]
            us = [np.array(u) for u in self.u_guess]
            nb_iteration = 500
            self.first_solve = False
        else:
            xs = [np.array(x) for x in self.croco_reaching.solver.xs]
            us = [np.array(x) for x in self.croco_reaching.solver.us]
            nb_iteration = 1

        return xs, us, nb_iteration

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
        obj_pose = [-0.2, 0, 0.02, 0, 0, 0, 1]
        q_init = [*sensor_msg.joint_state.position, *obj_pose]
        self.hpp_interface.set_panda_problem_solver(q_init)
        whole_x_plan, whole_a_plan, whole_traj_T = self.hpp_interface.get_hpp_plan(
            self.dt,
            self.rmodel.nq,
            self.ps.client.problem.getPath(self.ps.numberPaths() - 1),
        )

        # First solve
        self.mpc = MPC(self.ocp, whole_x_plan, whole_a_plan, self.rmodel, self.cmodel)
        x0 = self.mpc.whole_x_plan[0, :]
        self.x_plan = self.mpc.whole_x_plan[: self.horizon_size, :]
        self.a_plan = self.mpc.whole_a_plan[: self.horizon_size, :]
        self.mpc.mpc_first_step(self.x_plan, self.a_plan, x0, self.horizon_size)

        self.next_node_idx = self.horizon_size
        self.nb_iteration = 1

    def solve_and_send(self):
        sensor_msg = self.get_sensor_msg()
        x0 = np.concatenate(
            [sensor_msg.joint_state.position, sensor_msg.joint_state.velocity]
        )

        self.x_plan = self.mpc.update_planning(
            self.x_plan, self.mpc.whole_x_plan[self.next_node_idx, :]
        )
        self.a_plan = self.mpc.update_planning(
            self.a_plan, self.mpc.whole_a_plan[self.next_node_idx, :]
        )
        self.mpc.mpc_step(x0, self.x_plan, self.a_plan)
        if self.next_node_idx < self.mpc.whole_x_plan.shape[0] - 1:
            self.next_node_idx += 1

        _, u, k = self.mpc.get_mpc_output()

        self.control_msg.header = Header()
        self.control_msg.header.stamp = rospy.Time.now()
        self.control_msg.feedback_gain = to_multiarray_f64(k)
        self.control_msg.feedforward = to_multiarray_f64(u)
        self.control_msg.initial_state = sensor_msg
        self.control_publisher.publish(self.control_msg)
        self.nb_iteration += 1

    def get_sensor_msg(self):
        with self.mutex:
            sensor_msg = deepcopy(self.sensor_msg)
        return sensor_msg

    def run(self):
        self.wait_first_sensor_msg()
        self.plan_and_first_solve()
        input("Press Enter to continue...")
        while not rospy.is_shutdown():
            start_compute_time = rospy.Time.now()
            self.solve_and_send()
            self.ocp_solve_time.data = rospy.Time.now() - start_compute_time
            self.ocp_solve_time_pub.publish(self.ocp_solve_time)
            self.rate.sleep()


def crocco_motion_server_node():
    rospy.init_node("croccodyl_motion_server_node_py", anonymous=True)
    node = HppAgimusController()
    node.run()


if __name__ == "__main__":
    try:
        crocco_motion_server_node()
    except rospy.ROSInterruptException:
        pass
