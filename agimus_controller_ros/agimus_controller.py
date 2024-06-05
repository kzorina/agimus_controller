import rospy
from geometry_msgs.msg import Vector3


class AgimusControllerNode:
    def __init__(self) -> None:
        self.op_frames = rospy.get_param("op_frames", [])

        self.subscribers = []
        self.subscribers += [
            rospy.Subscriber("com/position", Vector3, self.com_position_callback)
        ]
        self.subscribers += [
            rospy.Subscriber("com/velocity", Vector3, self.com_velocity_callback)
        ]
        for op_frame in self.op_frames:
            self.subscribers += [
                rospy.Subscriber("com/velocity", Vector3, self.com_velocity_callback)
            ]

    def run(self):
        pass

    def com_position_callback(self, msg):
        print("CoM pos msg = ", msg)

    def com_velocity_callback(self, msg):
        print("CoM vel msg = ", msg)

    def op_frames_callback(self, op_frame, msg):
        print("Op frame = ", msg)
        print(op_frame)
