import pinocchio as pin
import numpy as np
from pathlib import Path
from agimus_controller.robot_model.obstacle_params_parser import ObstacleParamsParser


class RobotModelParameters:
    q0_name = "default"
    free_flyer = False
    locked_joint_names = []
    urdf = Path()
    srdf = Path()
    collision_as_capsule = True
    self_collision = True


class RobotModel:
    """Complete model of the robot."""

    _model = pin.Model()
    """ Complete model of the robot with visualisation meshes. """
    _cmodel = pin.GeometryModel()
    """ Complete model of the robot with collision meshes. """
    _vmodel = pin.GeometryModel()
    """ Reduced model of the robot. """
    _rmodel = pin.Model()
    """ Reduced model of the robot with visualisation meshes. """
    _rcmodel = pin.GeometryModel()
    """ Reduced model of the robot with collision meshes. """
    _rvmodel = pin.GeometryModel()
    """ Default configuration q0. """
    _q0 = np.array([])
    """ Obstacle and collision meshes handler. """
    _collision_parser = ObstacleParamsParser()
    """ Paramters of the model. """
    _params = RobotModelParameters()
    """ Path to the collisions environment. """
    _env = Path()

    def __init__(self):
        pass

    @classmethod
    def load_model(cls, param: RobotModelParameters, env: Path):
        model = cls()
        model._params = param
        model._env = env
        model._load_pinocchio_models(param.urdf, param.free_flyer)
        model._load_default_configuration(param.srdf, param.q0_name)
        model._load_reduced_model(param.locked_joint_names)
        model._update_collision_model(
            env, param.collision_as_capsule, param.self_collision, param.srdf
        )
        return model

    def _load_pinocchio_models(self, urdf: Path, free_flyer: bool):
        verbose = False
        assert urdf.exists() and urdf.is_file()
        if free_flyer:
            self._model, self._cmodel, self._vmodel = pin.buildModelsFromUrdf(
                filename=str(urdf),
                root_joint=pin.JointModelFreeFlyer(),
                verbose=verbose,
            )
        else:
            self._model, self._cmodel, self._vmodel = pin.buildModelsFromUrdf(
                filename=str(urdf), root_joint=None, verbose=verbose
            )

    def _load_default_configuration(self, srdf_path: Path, q0_name: str):
        if not srdf_path.is_file():
            return
        pin.loadReferenceConfigurations(self._model, str(srdf_path), False)
        if q0_name in self._model.referenceConfigurations:
            self._q0 = self._model.referenceConfigurations[q0_name]
        else:
            self._q0 = pin.neutral(self._model)

    def _load_reduced_model(self, locked_joint_names):
        locked_joint_ids = [self._model.getJointId(name) for name in locked_joint_names]
        self._rmodel, geometric_models_reduced = pin.buildReducedModel(
            self._model,
            list_of_geom_models=[self._cmodel, self._vmodel],
            list_of_joints_to_lock=locked_joint_ids,
            reference_configuration=self._q0,
        )
        self._rcmodel, self._rvmodel = geometric_models_reduced

    def _update_collision_model(
        self, env: Path, collision_as_capsule: bool, self_collision: bool, srdf: Path
    ):
        if collision_as_capsule:
            self._rcmodel = self._collision_parser.transform_model_into_capsules(
                self._rcmodel
            )
        if self_collision and srdf.exists():
            self._rcmodel = self._collision_parser.add_self_collision(
                self._rmodel, self._rcmodel, srdf
            )
        if env is not None:
            self._rcmodel = self._collision_parser.add_collisions(self._rcmodel, env)

    def get_complete_robot_model(self):
        return self._model.copy()

    def get_complete_collision_model(self):
        return self._cmodel.copy()

    def get_complete_visual_model(self):
        return self._vmodel.copy()

    def get_reduced_robot_model(self):
        return self._rmodel.copy()

    def get_reduced_collision_model(self):
        return self._rcmodel.copy()

    def get_reduced_visual_model(self):
        return self._rvmodel.copy()

    def get_default_configuration(self):
        return self._q0.copy()
