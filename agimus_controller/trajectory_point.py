import numpy as np
from enum import Enum
from __future__ import annotations


class PointAttribute(Enum):
    Q = 0
    V = 1
    A = 2
    TAU = 3
    COM_POS = 4
    COM_VEL = 5


class TrajectoryPoint:
    def __init__(self, time=0, nq=0, nv=0):
        self.q = np.zeros(nq)
        self.v = np.zeros(nv)
        self.a = np.zeros(nv)
        self.tau = np.zeros(nv)
        self.com_pos = np.zeros(3)
        self.com_vel = np.zeros(3)
        self.op_pos = {}
        self.op_vel = {}
        self.nq = nq
        self.nv = nv
        self.time = time
        self.attribute_validation_dict = self.get_attribute_validation_dict()

    def get_attribute_validation_dict(self):
        attribute_validation_dict = {}
        attribute_validation_dict[PointAttribute.Q] = self.q_is_valid
        attribute_validation_dict[PointAttribute.V] = self.v_is_valid
        attribute_validation_dict[PointAttribute.A] = self.a_is_valid
        attribute_validation_dict[PointAttribute.TAU] = self.tau_is_valid
        attribute_validation_dict[PointAttribute.COM_POS] = self.com_pos_is_valid
        attribute_validation_dict[PointAttribute.COM_VEL] = self.com_vel_is_valid
        return attribute_validation_dict

    def resize(self, nq, nv):
        self.q = np.zeros(nq)
        self.v = np.zeros(nv)
        self.a = np.zeros(nv)
        self.tau = np.zeros(nv)
        self.nq = nq
        self.nv = nv

    def get_x_as_q_v(self):
        return np.concatenate([self.q, self.v])

    def attribute_is_valid(self, attribute: PointAttribute):
        return self.attribute_validation_dict[attribute]()

    def q_is_valid(self):
        if self.q == np.zeros(self.nq):
            return False
        else:
            return True

    def v_is_valid(self):
        if self.v == np.zeros(self.nv):
            return False
        else:
            return True

    def a_is_valid(self):
        if self.a == np.zeros(self.nv):
            return False
        else:
            return True

    def tau_is_valid(self):
        if self.tau == np.zeros(self.nv):
            return False
        else:
            return True

    def com_pos_is_valid(self):
        if self.com_pos == np.zeros(3):
            return False
        else:
            return True

    def com_vel_is_valid(self):
        if self.com_vel == np.zeros(3):
            return False
        else:
            return True
