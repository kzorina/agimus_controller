from rclpy.serialization import deserialize_message
import rosbag2_py
import pickle

from agimus_controller_ros.ros_utils import mpc_msg_to_weighted_traj_point
from agimus_msgs.msg import MpcInput


def convert_bytes_to_message(serialized_bytes, msg_type):
    # Deserialize the bytes into the correct message type
    message = deserialize_message(serialized_bytes, msg_type)
    return message


# Define function to convert ROS 2 messages to a pickle file
def save_rosbag_to_pickle(bag_file_path, pickle_file_path):
    # Open the rosbag
    storage_options = rosbag2_py.StorageOptions(uri=bag_file_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    # Open the pickle file
    with open(pickle_file_path, "wb") as pickle_file:
        data = []

        # Read each message in the rosbag and store it
        while reader.has_next():
            topic, msg, timestamp = reader.read_next()
            assert topic == "/mpc_input"
            print(timestamp)
            # print(msg)
            # Save topic, message, and timestamp as a tuple in the list
            data.append(
                mpc_msg_to_weighted_traj_point(
                    convert_bytes_to_message(msg, MpcInput), timestamp
                )
            )

        # Serialize data to pickle file
        pickle.dump(data, pickle_file)


# Example usage
bag_file = "/home/gepetto/ros2_ws/src/agimus_controller/bag_files/slow_sin"  # Path to the bag file without the .db3 extension
pickle_file = "slow_sim_weighted_trajectory_data.pkl"  # Path to the pickle file

save_rosbag_to_pickle(bag_file, pickle_file)
print(f"Saved ROS 2 bag data to {pickle_file}")
