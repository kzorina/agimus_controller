import yaml
import numpy as np
import pinocchio as pin
from pathlib import Path
from hppfcl import Sphere, Box, Cylinder, Capsule


class ObstacleParamsParser:
    def add_collisions(
        self, cmodel: pin.GeometryModel, yaml_file: Path
    ) -> pin.GeometryModel:
        new_cmodel = cmodel.copy()
        return new_cmodel

    def add_collision_pair(
        self, cmodel: pin.GeometryModel, name_object1: str, name_object2: str
    ) -> pin.GeometryModel:
        return cmodel

    def transform_model_into_capsules(
        self, model: pin.GeometryModel
    ) -> pin.GeometryModel:
        """Modifying the collision model to transform the spheres/cylinders into capsules which makes it easier to have a fully constrained robot."""
        model_copy = model.copy()
        return model_copy

    def add_self_collision(
        self, rmodel: pin.Model, rcmodel: pin.GeometryModel, srdf: Path = Path()
    ) -> pin.GeometryModel:
        return rcmodel
