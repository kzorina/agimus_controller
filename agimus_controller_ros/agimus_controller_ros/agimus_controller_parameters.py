# flake8: noqa

# auto-generated DO NOT EDIT

from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import SetParametersResult
from rcl_interfaces.msg import FloatingPointRange, IntegerRange
from rclpy.clock import Clock
from rclpy.exceptions import InvalidParameterValueException
from rclpy.time import Time
import copy
import rclpy
import rclpy.parameter
from generate_parameter_library_py.python_validators import ParameterValidators



class agimus_controller_params:

    class Params:
        # for detecting if the parameter struct has been updated
        stamp_ = Time()

        publish_debug_data = True
        rate = 100.0
        free_flyer = False
        collision_as_capsule = True
        self_collision = True
        class __Ocp:
            dt = 0.01
            horizon_size = 20
            armature = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
            gripper_weight = 500.0
            state_weight = 5.0
            control_weight = 0.001
            max_qp_iter = 100
            max_iter = 10
            use_constraints = True
            activate_callback = False
        ocp = __Ocp()



    class ParamListener:
        def __init__(self, node, prefix=""):
            self.prefix_ = prefix
            self.params_ = agimus_controller_params.Params()
            self.node_ = node
            self.logger_ = rclpy.logging.get_logger("agimus_controller_params." + prefix)

            self.declare_params()

            self.node_.add_on_set_parameters_callback(self.update)
            self.clock_ = Clock()

        def get_params(self):
            tmp = self.params_.stamp_
            self.params_.stamp_ = None
            paramCopy = copy.deepcopy(self.params_)
            paramCopy.stamp_ = tmp
            self.params_.stamp_ = tmp
            return paramCopy

        def is_old(self, other_param):
            return self.params_.stamp_ != other_param.stamp_

        def unpack_parameter_dict(self, namespace: str, parameter_dict: dict):
            """
            Flatten a parameter dictionary recursively.

            :param namespace: The namespace to prepend to the parameter names.
            :param parameter_dict: A dictionary of parameters keyed by the parameter names
            :return: A list of rclpy Parameter objects
            """
            parameters = []
            for param_name, param_value in parameter_dict.items():
                full_param_name = namespace + param_name
                # Unroll nested parameters
                if isinstance(param_value, dict):
                    nested_params = self.unpack_parameter_dict(
                            namespace=full_param_name + rclpy.parameter.PARAMETER_SEPARATOR_STRING,
                            parameter_dict=param_value)
                    parameters.extend(nested_params)
                else:
                    parameters.append(rclpy.parameter.Parameter(full_param_name, value=param_value))
            return parameters

        def set_params_from_dict(self, param_dict):
            params_to_set = self.unpack_parameter_dict('', param_dict)
            self.update(params_to_set)

        def refresh_dynamic_parameters(self):
            updated_params = self.get_params()
            # TODO remove any destroyed dynamic parameters

            # declare any new dynamic parameters


        def update(self, parameters):
            updated_params = self.get_params()

            for param in parameters:
                if param.name == self.prefix_ + "ocp.dt":
                    updated_params.ocp.dt = param.value
                    self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))

                if param.name == self.prefix_ + "ocp.horizon_size":
                    updated_params.ocp.horizon_size = param.value
                    self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))

                if param.name == self.prefix_ + "ocp.armature":
                    updated_params.ocp.armature = param.value
                    self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))

                if param.name == self.prefix_ + "ocp.gripper_weight":
                    updated_params.ocp.gripper_weight = param.value
                    self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))

                if param.name == self.prefix_ + "ocp.state_weight":
                    updated_params.ocp.state_weight = param.value
                    self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))

                if param.name == self.prefix_ + "ocp.control_weight":
                    updated_params.ocp.control_weight = param.value
                    self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))

                if param.name == self.prefix_ + "ocp.max_qp_iter":
                    updated_params.ocp.max_qp_iter = param.value
                    self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))

                if param.name == self.prefix_ + "ocp.max_iter":
                    updated_params.ocp.max_iter = param.value
                    self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))

                if param.name == self.prefix_ + "ocp.use_constraints":
                    updated_params.ocp.use_constraints = param.value
                    self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))

                if param.name == self.prefix_ + "ocp.activate_callback":
                    updated_params.ocp.activate_callback = param.value
                    self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))

                if param.name == self.prefix_ + "publish_debug_data":
                    updated_params.publish_debug_data = param.value
                    self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))

                if param.name == self.prefix_ + "rate":
                    updated_params.rate = param.value
                    self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))

                if param.name == self.prefix_ + "free_flyer":
                    updated_params.free_flyer = param.value
                    self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))

                if param.name == self.prefix_ + "collision_as_capsule":
                    updated_params.collision_as_capsule = param.value
                    self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))

                if param.name == self.prefix_ + "self_collision":
                    updated_params.self_collision = param.value
                    self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))



            updated_params.stamp_ = self.clock_.now()
            self.update_internal_params(updated_params)
            return SetParametersResult(successful=True)

        def update_internal_params(self, updated_params):
            self.params_ = updated_params

        def declare_params(self):
            updated_params = self.get_params()
            # declare all parameters and give default values to non-required ones
            if not self.node_.has_parameter(self.prefix_ + "ocp.dt"):
                descriptor = ParameterDescriptor(description="Step time of the OCP in seconds", read_only = False)
                parameter = updated_params.ocp.dt
                self.node_.declare_parameter(self.prefix_ + "ocp.dt", parameter, descriptor)

            if not self.node_.has_parameter(self.prefix_ + "ocp.horizon_size"):
                descriptor = ParameterDescriptor(description="Number of nodes of the OCP", read_only = False)
                parameter = updated_params.ocp.horizon_size
                self.node_.declare_parameter(self.prefix_ + "ocp.horizon_size", parameter, descriptor)

            if not self.node_.has_parameter(self.prefix_ + "ocp.armature"):
                descriptor = ParameterDescriptor(description="Armature for each OCP node", read_only = False)
                parameter = updated_params.ocp.armature
                self.node_.declare_parameter(self.prefix_ + "ocp.armature", parameter, descriptor)

            if not self.node_.has_parameter(self.prefix_ + "ocp.gripper_weight"):
                descriptor = ParameterDescriptor(description="Scalar weight for the end-effector pose cost", read_only = False)
                parameter = updated_params.ocp.gripper_weight
                self.node_.declare_parameter(self.prefix_ + "ocp.gripper_weight", parameter, descriptor)

            if not self.node_.has_parameter(self.prefix_ + "ocp.state_weight"):
                descriptor = ParameterDescriptor(description="Scalar weight for the state cost", read_only = False)
                parameter = updated_params.ocp.state_weight
                self.node_.declare_parameter(self.prefix_ + "ocp.state_weight", parameter, descriptor)

            if not self.node_.has_parameter(self.prefix_ + "ocp.control_weight"):
                descriptor = ParameterDescriptor(description="Scalar weight for the control cost", read_only = False)
                parameter = updated_params.ocp.control_weight
                self.node_.declare_parameter(self.prefix_ + "ocp.control_weight", parameter, descriptor)

            if not self.node_.has_parameter(self.prefix_ + "ocp.max_qp_iter"):
                descriptor = ParameterDescriptor(description="Maximum number of qp iterations of the solver", read_only = False)
                parameter = updated_params.ocp.max_qp_iter
                self.node_.declare_parameter(self.prefix_ + "ocp.max_qp_iter", parameter, descriptor)

            if not self.node_.has_parameter(self.prefix_ + "ocp.max_iter"):
                descriptor = ParameterDescriptor(description="Maximum number of iterations of the solver", read_only = False)
                parameter = updated_params.ocp.max_iter
                self.node_.declare_parameter(self.prefix_ + "ocp.max_iter", parameter, descriptor)

            if not self.node_.has_parameter(self.prefix_ + "ocp.use_constraints"):
                descriptor = ParameterDescriptor(description="if True we have Collisions constraints and we use CSQP solver, if False we use FDDP solver", read_only = False)
                parameter = updated_params.ocp.use_constraints
                self.node_.declare_parameter(self.prefix_ + "ocp.use_constraints", parameter, descriptor)

            if not self.node_.has_parameter(self.prefix_ + "ocp.activate_callback"):
                descriptor = ParameterDescriptor(description="Whether or not we want to use solver callbacks", read_only = False)
                parameter = updated_params.ocp.activate_callback
                self.node_.declare_parameter(self.prefix_ + "ocp.activate_callback", parameter, descriptor)

            if not self.node_.has_parameter(self.prefix_ + "publish_debug_data"):
                descriptor = ParameterDescriptor(description="Wether we publish debug data or not", read_only = False)
                parameter = updated_params.publish_debug_data
                self.node_.declare_parameter(self.prefix_ + "publish_debug_data", parameter, descriptor)

            if not self.node_.has_parameter(self.prefix_ + "rate"):
                descriptor = ParameterDescriptor(description="Rate of mpc node in hertz", read_only = False)
                parameter = updated_params.rate
                self.node_.declare_parameter(self.prefix_ + "rate", parameter, descriptor)

            if not self.node_.has_parameter(self.prefix_ + "free_flyer"):
                descriptor = ParameterDescriptor(description="Wether we add a free flyer to the robot base", read_only = False)
                parameter = updated_params.free_flyer
                self.node_.declare_parameter(self.prefix_ + "free_flyer", parameter, descriptor)

            if not self.node_.has_parameter(self.prefix_ + "collision_as_capsule"):
                descriptor = ParameterDescriptor(description="Wether we convert collision objects to capsules", read_only = False)
                parameter = updated_params.collision_as_capsule
                self.node_.declare_parameter(self.prefix_ + "collision_as_capsule", parameter, descriptor)

            if not self.node_.has_parameter(self.prefix_ + "self_collision"):
                descriptor = ParameterDescriptor(description="Wether we account for self-collision", read_only = False)
                parameter = updated_params.self_collision
                self.node_.declare_parameter(self.prefix_ + "self_collision", parameter, descriptor)

            # TODO: need validation
            # get parameters and fill struct fields
            param = self.node_.get_parameter(self.prefix_ + "ocp.dt")
            self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))
            updated_params.ocp.dt = param.value
            param = self.node_.get_parameter(self.prefix_ + "ocp.horizon_size")
            self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))
            updated_params.ocp.horizon_size = param.value
            param = self.node_.get_parameter(self.prefix_ + "ocp.armature")
            self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))
            updated_params.ocp.armature = param.value
            param = self.node_.get_parameter(self.prefix_ + "ocp.gripper_weight")
            self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))
            updated_params.ocp.gripper_weight = param.value
            param = self.node_.get_parameter(self.prefix_ + "ocp.state_weight")
            self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))
            updated_params.ocp.state_weight = param.value
            param = self.node_.get_parameter(self.prefix_ + "ocp.control_weight")
            self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))
            updated_params.ocp.control_weight = param.value
            param = self.node_.get_parameter(self.prefix_ + "ocp.max_qp_iter")
            self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))
            updated_params.ocp.max_qp_iter = param.value
            param = self.node_.get_parameter(self.prefix_ + "ocp.max_iter")
            self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))
            updated_params.ocp.max_iter = param.value
            param = self.node_.get_parameter(self.prefix_ + "ocp.use_constraints")
            self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))
            updated_params.ocp.use_constraints = param.value
            param = self.node_.get_parameter(self.prefix_ + "ocp.activate_callback")
            self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))
            updated_params.ocp.activate_callback = param.value
            param = self.node_.get_parameter(self.prefix_ + "publish_debug_data")
            self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))
            updated_params.publish_debug_data = param.value
            param = self.node_.get_parameter(self.prefix_ + "rate")
            self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))
            updated_params.rate = param.value
            param = self.node_.get_parameter(self.prefix_ + "free_flyer")
            self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))
            updated_params.free_flyer = param.value
            param = self.node_.get_parameter(self.prefix_ + "collision_as_capsule")
            self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))
            updated_params.collision_as_capsule = param.value
            param = self.node_.get_parameter(self.prefix_ + "self_collision")
            self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))
            updated_params.self_collision = param.value


            self.update_internal_params(updated_params)
