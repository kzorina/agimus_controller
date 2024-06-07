import rospy
from functools import partial
from geometry_msgs.msg import Vector3, Transform


class AgimusControllerNodeParameters:
    def __init__(self) -> None:
        self.op_frames = rospy.get_param("~op_frames", [])
        self.name = rospy.get_param("~name", "robot")
        self.prefix = rospy.get_param("~prefix", "agimus")
        self.rate = rospy.get_param("~rate", 100)


class TrajectoryPoint:
    configuration = None
    velocity = None
    acceleration = None
    effort = None
    com = None
    com_vel = None
    com_acc = None
    op_frame = []
    op_frame_vel = []


class AgimusControllerNode:
    def __init__(self) -> None:
        rospy.loginfo("Load parameters")
        self.params = AgimusControllerNodeParameters()

        rospy.loginfo("Create the rate object.")
        self.rate = rospy.Rate(self.params.rate)  # 10hz

        rospy.loginfo("Parse the prefix/name params.")
        if not self.params.prefix.endswith("/"):
            self.params.prefix = self.params.prefix + "/"
        if not self.params.prefix.startswith("/"):
            self.params.prefix = "/" + self.params.prefix
        self.params.name.replace("/", "")

        rospy.loginfo("Spawn the subscribers.")
        self.subscribers = []
        rospy.loginfo("\t- CoM pos subscriber.")
        self.subscribers += [
            rospy.Subscriber(
                self.params.prefix + "com/" + self.params.name,
                Vector3,
                self.com_pose_callback,
            )
        ]
        rospy.loginfo("\t- CoM vel subscriber.")
        self.subscribers += [
            rospy.Subscriber(
                self.params.prefix + "velocity/com/" + self.params.name,
                Vector3,
                self.com_velocity_callback,
            )
        ]
        for op_frame in self.params.op_frames:
            rospy.loginfo("\t- OP frame pos subscriber.")
            self.subscribers += [
                rospy.Subscriber(
                    "agimus/" + op_frame + self.params.name,
                    Transform,
                    partial(self.params.op_frames_callback, op_frame),
                )
            ]
            rospy.loginfo("\t- OP frame pos velocity.")
            self.subscribers += [
                rospy.Subscriber(
                    "agimus/" + op_frame + "/velocity",
                    Transform,
                    partial(self.params.op_frames_callback, op_frame),
                )
            ]

    def run(self):
        while not rospy.is_shutdown():
            hello_str = "hello world %s" % rospy.get_time()
            rospy.loginfo(hello_str)
            self.rate.sleep()

    def com_pose_callback(self, msg):
        rospy.logdebug("CoM pos msg = ", msg)

    def com_velocity_callback(self, msg):
        rospy.logdebug("CoM vel msg = ", msg)

    def op_frames_callback(self, op_frame, msg):
        rospy.logdebug("Op frame = ", msg)
        rospy.logdebug(op_frame)
