import unittest

import example_robot_data as robex

from agimus_controller.agimus_controller.ocp_base_croco import OCPCrocoBase
from agimus_controller.agimus_controller.ocp_param_base import OCPParamsCrocoBase


class TestOCPBase(unittest.TestCase):
    
    def setUp(self):
        
        # Load the robot
        self.robot = robex.load('example_robot_description')
        self.rmodel = self.robot.model
        self.cmodel = self.robot.collision_model
        
        # Generate some fixed params for unit testing
        self.OCPParams = OCPParamsCrocoBase(
            dt = 0.01,
            T = 100,
            qp_iters=200,
            solver_iters=200,
            callbacks=True,
            
        )
        return super().setUp()
    
    def test_abstract_class_instantiation(self):
        with self.assertRaises(TypeError):
            
            OCPCrocoBase()