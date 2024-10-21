#!/usr/bin/env python3
import rospy
import numpy as np
from agimus_controller.ocps.parameters import OCPParameters


class OCPParametersROS(OCPParameters):
    def __init__(self) -> None:
        super().__init__()

    def set_parameters_from_ros(self):
        self.dt = rospy.get_param("ocp/dt", 0.01)
        self.horizon_size = rospy.get_param("ocp/horizon_size", 100)
        self.armature = np.array(
            rospy.get_param("ocp/armature", [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
        )
        self.gripper_weight = rospy.get_param("ocp/gripper_weight", 10000)
        self.state_weight = rospy.get_param("ocp/state_weight", 10)
        self.control_weight = rospy.get_param("ocp/control_weight", 0.001)
        self.max_iter = rospy.get_param("ocp/max_iter", 1)
        self.max_qp_iter = rospy.get_param("ocp/max_qp_iter", 100)
        self.use_constraints = rospy.get_param("ocp/use_constraints", False)
        self.effector_frame_name = rospy.get_param(
            "ocp/effector_frame_name", "panda_hand_tcp"
        )
        self.activate_callback = rospy.get_param("ocp/activate_callback", False)


class AgimusControllerNodeParameters:
    def __init__(self) -> None:
        self.save_predictions_and_refs = None
        self.rate = None
        self.use_vision = None
        self.use_vision_simulated = None
        self.start_visual_servoing_dist = None
        self.increasing_weights = None
        self.ocp = OCPParametersROS()
        self.use_ros_params = None

    def set_parameters_from_ros(self):
        self.save_predictions_and_refs = rospy.get_param(
            "save_predictions_and_refs", False
        )
        self.rate = rospy.get_param("rate", 100)
        self.use_vision = rospy.get_param("use_vision", False)
        self.use_vision_simulated = rospy.get_param("use_vision_simulated", False)
        self.start_visual_servoing_dist = rospy.get_param(
            "start_visual_servoing_dist", 0.03
        )
        self.increasing_weights = rospy.get_param("increasing_weights", [])
        self.ocp.set_parameters_from_ros()
        self.use_ros_params = True

    def set_parameters_from_dict(self, params_dict):
        self.save_predictions_and_refs = params_dict["save_predictions_and_refs"]
        self.rate = params_dict["rate"]
        self.use_vision = params_dict["use_vision"]
        self.use_vision_simulated = params_dict["use_vision_simulated"]
        self.start_visual_servoing_dist = params_dict["start_visual_servoing_dist"]
        self.increasing_weights = params_dict["increasing_weights"]
        self.ocp.set_parameters_from_dict(params_dict["ocp"])
        self.use_ros_params = False

    def get_dict(self):
        params = {}
        params["save_predictions_and_refs"] = self.save_predictions_and_refs
        params["rate"] = self.rate
        params["use_vision"] = self.use_vision
        params["use_vision_simulated"] = self.use_vision_simulated
        params["start_visual_servoing_dist"] = self.start_visual_servoing_dist
        params["increasing_weights"] = self.increasing_weights
        params["ocp"] = self.ocp.get_dict()
        return params
