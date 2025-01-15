from pathlib import Path
import crocoddyl
import example_robot_data as robex
import numpy as np
import pinocchio as pin
from agimus_controller.ocp_base_croco import OCPBaseCroco
from agimus_controller.ocp_param_base import OCPParamsBaseCroco
from agimus_controller.factory.robot_model import RobotModels, RobotModelParameters
from agimus_controller.ocp.ocp_exp3_main import OCPCrocoExp3

### Loading the robot
robot = robex.load("panda")
urdf_path = Path(robot.urdf)
srdf_path = Path(robot.urdf.replace("urdf", "srdf"))
urdf_meshes_dir = urdf_path.parent.parent.parent.parent.parent
free_flyer = False
locked_joint_names = ["panda_finger_joint1", "panda_finger_joint2"]
reduced_nq = robot.model.nq - len(locked_joint_names)
full_q0 = np.zeros(robot.model.nq)
q0 = np.zeros(reduced_nq)
armature = np.full(reduced_nq, 0.1)
# Store shared initial parameters
params = RobotModelParameters(
    q0=q0,
    full_q0=full_q0,
    free_flyer=free_flyer,
    locked_joint_names=locked_joint_names,
    urdf_path=urdf_path,
    srdf_path=srdf_path,
    urdf_meshes_dir=urdf_meshes_dir,
    collision_as_capsule=True,
    self_collision=True,
    armature=armature,
)

robot_models = RobotModels(params)
robot_model = robot_models.robot_model
collision_model = robot_models.collision_model

# Set mock parameters
ocp_params = OCPCrocoExp3(dt=0.1, horizon_size=10, solver_iters=100, callbacks=True)
