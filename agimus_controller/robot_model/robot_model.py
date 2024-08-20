import pinocchio as pin
import numpy as np
from pathlib import Path
from agimus_controller.robot_model.obstacle_params_parser import ObstacleParamsParser


class RobotModelParameters:
    model_name = ""
    q0_name = ""
    free_flyer = False
    locked_joint_names = []
    urdf = str()
    srdf = str()
    meshes = Path()
    collision_as_capsule = False
    self_collision = False


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

    def __init__(self):
        pass

    @classmethod
    def load_model(cls, param: RobotModelParameters, env: Path):
        model = cls()
        model._load_pinocchio_models(param.urdf, param.free_flyer)
        model._load_default_configuration(param.srdf, param.q0_name)
        model._load_reduced_model(param.locked_joint_names)
        model._update_collision_model(
            env, param.collision_as_capsule, param.self_collision, param.srdf
        )
        return model

    def _load_pinocchio_models(self, urdf: str, free_flyer: bool):
        # Reset models
        self._model = pin.Model()
        self._vmodel = pin.Model()
        self._cmodel = pin.Model()
        try:
            is_valid_file = Path(urdf).exists() and Path(urdf).is_file()
        except OSError:
            is_valid_file = False
        except ...:
            is_valid_file = False
        if is_valid_file:
            pin_build_model = pin.buildModelFromUrdf
            pin_build_geom = pin.buildGeomFromUrdf
        else:
            pin_build_model = pin.buildModelFromXML
            pin_build_geom = pin.buildGeomFromUrdfString

        if free_flyer:
            pin_build_model(urdf, self._model, pin.JointModelFreeFlyer())
        else:
            pin_build_model(urdf, self._model)
        self._cmodel = pin_build_geom(self._model, urdf, pin.COLLISION)
        self._vmodel = pin_build_geom(self._model, urdf, pin.VISUAL)

    def _load_default_configuration(self, srdf_path: Path, q0_name: np.array):
        pin.loadReferenceConfigurations(self._model, str(srdf_path), False)
        self._q0 = self._model.referenceConfigurations[q0_name]

    def _load_reduced_model(self, locked_joint_names):
        locked_joint_ids = [self._model.getJointId(name) for name in locked_joint_names]
        self._rmodel, geometric_models_reduced = pin.buildReducedModel(
            self._model,
            list_of_geom_models=[self._cmodel, self._vmodel],
            list_of_joints_to_lock=locked_joint_ids,
            reference_configuration=self._q0,
        )
        self._rvmodel, self._rcmodel = geometric_models_reduced

    def _update_collision_model(
        self, env: Path, collision_as_capsule: bool, self_collision: bool, srdf: Path
    ):
        rcmodel0 = self._rcmodel.copy()
        if collision_as_capsule:
            self._rcmodel = self._collision_parser.transform_model_into_capsules(
                self._rcmodel
            )
        rcmodel1 = self._rcmodel.copy()
        if self_collision and srdf.exists():
            self._rcmodel = self._collision_parser.add_self_collision(
                self._rmodel, self._rcmodel, srdf
            )
        rcmodel2 = self._rcmodel.copy()
        if env is not None:
            self._rcmodel = self._collision_parser.add_collisions(self._rcmodel, env)
        rcmodel3 = self._rcmodel.copy()
        assert rcmodel0 != rcmodel1
        assert rcmodel1 != rcmodel2
        assert rcmodel2 != rcmodel3

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
