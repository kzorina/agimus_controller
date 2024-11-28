from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='agimus_controller_ros',
            namespace='/ctrl_mpc_linearized',
            executable='agimus_controller_node',
            name='agimus_controller_node'
        )
    ])
