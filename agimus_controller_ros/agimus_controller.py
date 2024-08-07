import rospy

from agimus_controller_ros.controller_base import ControllerBase

from agimus_controller_ros.hpp_subscriber import HPPSubscriber


class AgimusControllerNode(ControllerBase):
    def __init__(self) -> None:
        super().__init__()
        self.hpp_subscriber = HPPSubscriber()

    def get_next_trajectory_point(self):
        return self.hpp_subscriber.get_trajectory_point()

    def run(self):
        self.wait_first_sensor_msg()
        self.wait_buffer_has_twice_horizon_points()
        sensor_msg, u, k = self.first_solve()
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
