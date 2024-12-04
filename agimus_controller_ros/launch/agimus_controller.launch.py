import os
import xacro

from ament_index_python.packages import get_package_share_directory

from launch import LaunchContext, LaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, LaunchConfiguration

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    mpc_params = PathJoinSubstitution(
        [
            FindPackageShare("agimus_demo_01_lfc_alone"),
            "config",
            "linear_feedback_controller.yaml",
        ]
    )
    print("parameters of the mpc",mpc_params)
    return LaunchDescription([
        Node(
            package='agimus_controller_ros',
            namespace='/ctrl_mpc_linearized',
            executable='agimus_controller_node',
            name='agimus_controller_node',
            parameters=[
            mpc_params
        ]
        )
    ])
