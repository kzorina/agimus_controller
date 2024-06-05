import rospy
from geometry_msgs.msg import Vector3, Transform
from dynamic_graph_bridge_msgs.msg import Vector

class AgimusControllerNode:

    def __init__(self) -> None:
        
        self.subscribers = []
        self.subscribers += [rospy.Subscriber("com/position", Vector3, self.com_position_callback)]
        self.subscribers += [rospy.Subscriber("com/velocity", Vector3, self.com_velocity_callback)]

    def run(self):
        pass

    def com_position_callback(self, msg):
        pass

    def com_velocity_callback(self, msg):
        print("CoM vel msg = ", msg)
