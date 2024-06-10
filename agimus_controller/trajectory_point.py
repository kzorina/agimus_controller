import numpy as np


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
        self.time = time

    def resize(self, nq, nv):
        self.q = np.zeros(nq)
        self.v = np.zeros(nv)
        self.a = np.zeros(nv)
        self.tau = np.zeros(nv)

    def get_x_as_q_v(self):
        return np.concatenate([self.q, self.v])
