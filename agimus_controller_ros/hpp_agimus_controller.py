#!/usr/bin/env python3
import rospy
import numpy as np
import pinocchio as pin
from copy import deepcopy
from threading import Lock
from std_msgs.msg import Duration
from linear_feedback_controller_msgs.msg import Control, Sensor
from agimus_controller.hpp_interface import HppInterface
from agimus_controller.mpc import MPC


class HppAgimusController:
    def __init__(self) -> None:
        self.hpp_interface = HppInterface()
        self.robot_model = self.hpp_interface.rmodel
        self.robot_data = pin.Data(self.robot_model)

        self.rate = rospy.Rate(self.params["freq_solve"])  # 10hz
        self.mutex = Lock()
        self.sensor_msg = Sensor()
        self.control_msg = Control()
        self.ocp_solve_time = Duration()
        self.x0 = np.zeros(self.robot_model.nq + self.robot_model.nv)
        self.x_guess = np.zeros(self.robot_model.nq + self.robot_model.nv)
        self.u_guess = np.zeros(self.robot_model.nv)
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
        self.iteration = 0
        self.dt = 1.0 / self.params["publish_rate"]
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
                self.robot_model, self.robot_data, sensor_msg.joint_state.position
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

    def plan_and_first_iteration(self):
        sensor_msg = self.get_sensor_msg()
        self.x_plan, self.a_plan, self.whole_traj_T = self.hpp_interface.get_hpp_plan(
            1e-2, 7, sensor_msg.joint_state.position
        )
        self.mpc = MPC(self.x_plan, self.a_plan, "panda")
        self.mpc.prob.set_costs(10**4, 1, 10**-3, 0, 0)

        self.mpc_xs = np.zeros([self.whole_traj_T, 2 * self.nq])
        self.mpc_us = np.zeros([self.whole_traj_T - 1, self.nq])
        x0 = self.mpc.whole_x_plan[0, :]
        self.mpc_xs[0, :] = x0

        x_plan = self.mpc.whole_x_plan[: self.horizon_size, :]
        a_plan = self.mpc.whole_a_plan[: self.horizon_size, :]
        x, u0 = self.mpc.mpc_first_step(x_plan, a_plan, x0, self.horizon_size)
        self.mpc_xs[1, :] = x
        self.mpc_us[0, :] = u0

        self.next_node_idx = self.horizon_size
        self.nb_iteration = 1

    def solve_and_send(self):
        pass
        sensor_msg = self.get_sensor_msg()
        x0 = np.concatenate(
            [sensor_msg.joint_state.position, sensor_msg.joint_state.velocity]
        )
        self.croco_reaching.set_posture_ref(x0)
        x_guess, u_guess, nb_iteration_max = self.warm_start(sensor_msg)
        self.croco_reaching.solve(x_guess, u_guess, nb_iteration_max)
        
        self.mpc.simulate_mpc(100)
        u_plan = self.mpc.prob.get_uref(self.x_plan, self.a_plan)
        
        x_plan = self.mpc.update_planning(
            x_plan, self.mpc.whole_x_plan[self.next_node_idx, :]
        )
        a_plan = self.mpc.update_planning(
            a_plan, self.mpc.whole_a_plan[self.next_node_idx, :]
        )
        x, u = self.mpc.mpc_step(x, x_plan, a_plan)
        if self.next_node_idx < self.mpc.whole_x_plan.shape[0] - 1:
            self.next_node_idx += 1
            self.mpc_xs[idx + 1, :] = x
            self.mpc_us[idx, :] = u
           if idx == node_idx_breakpoint:
                breakpoint()
        self.croco_xs = mpc_xs
        self.croco_us = mpc_us
        self.mpc.simulate_mpc()
        self.control_msg.header = Header()
        self.control_msg.header.stamp = rospy.Time.now()
        self.control_msg.feedback_gain = to_multiarray_f64(ricatti_mat)
        self.control_msg.feedforward = to_multiarray_f64(tau_ff)
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
