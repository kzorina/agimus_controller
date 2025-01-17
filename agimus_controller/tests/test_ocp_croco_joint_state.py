import unittest
import numpy as np
from pathlib import Path
import example_robot_data as robex
import pinocchio as pin

from agimus_controller.ocp.ocp_croco_joint_state import OCPCrocoJointState
from agimus_controller.ocp_param_base import OCPParamsBaseCroco
from agimus_controller.factory.robot_model import RobotModels, RobotModelParameters
from agimus_controller.trajectory import (
    TrajectoryPoint,
    TrajectoryPointWeights,
    WeightedTrajectoryPoint,
)


class TestOCPWarmstart(unittest.TestCase):
    def setUp(self):
        # Initialize robot
        robot = robex.load("panda")
        urdf_path = Path(robot.urdf)
        srdf_path = Path(robot.urdf.replace("urdf", "srdf"))
        urdf_meshes_dir = urdf_path.parent.parent.parent.parent.parent
        free_flyer = False
        locked_joint_names = []
        reduced_nq = robot.model.nq - len(locked_joint_names)
        full_q0 = np.zeros(robot.model.nq)
        q0 = np.zeros(reduced_nq)
        armature = np.full(reduced_nq, 0.1)

        # Store shared initial parameters
        self.params = RobotModelParameters(
            q0=q0,
            full_q0=full_q0,
            free_flyer=free_flyer,
            locked_joint_names=locked_joint_names,
            urdf_path=urdf_path,
            srdf_path=srdf_path,
            urdf_meshes_dir=urdf_meshes_dir,
            collision_as_capsule=True,
            self_collision=False,
            armature=armature,
        )

        self.robot_models = RobotModels(self.params)
        self.robot_model = self.robot_models.robot_model
        self.robot_data = self.robot_model.createData()

        # OCP parameters
        dt = 0.05
        self._horizon_size = 30
        solver_iters = 100
        callbacks = False

        self._ocp_params = OCPParamsBaseCroco(
            dt=dt,
            horizon_size=self._horizon_size,
            solver_iters=solver_iters,
            callbacks=callbacks,
            qp_iters=500,
        )

    def test_ocp_solution(self):
        # Set initial state
        q0 = pin.neutral(self.robot_model)

        # Generate warmstart trajectories
        amplitude = np.deg2rad(20)
        frequency = 2.0
        self._horizon_size = self._ocp_params.horizon_size
        self._state_warmstart = []
        control_warmstart = []
        trajectory_points = []

        for t in range(1, self._horizon_size):
            fraction = t / self._horizon_size
            oscillation = amplitude * np.sin(2.0 * np.pi * frequency * fraction)
            q_t = q0.copy()
            q_t[3] += oscillation
            self._state_warmstart.append(
                np.concatenate((q_t, np.zeros(self.robot_model.nv)))
            )
            qdot_t = (
                np.zeros(self.robot_model.nv)
                if t == 1
                else self._state_warmstart[-1][: self.robot_model.nq]
                - self._state_warmstart[t - 2][: self.robot_model.nq]
            )
            control_warmstart.append(
                pin.rnea(
                    self.robot_model,
                    self.robot_data,
                    q_t,
                    qdot_t,
                    np.zeros(self.robot_model.nv),
                )
            )

            trajectory_points.append(
                WeightedTrajectoryPoint(
                    TrajectoryPoint(
                        robot_configuration=self._state_warmstart[-1][
                            : self.robot_model.nq
                        ],
                        robot_velocity=self._state_warmstart[-1][
                            self.robot_model.nq : self.robot_model.nq
                            + self.robot_model.nv
                        ],
                    ),
                    TrajectoryPointWeights(
                        w_robot_configuration=1e6,
                    ),
                )
            )

        # Solve OCP
        self._state_reg = np.concatenate((q0, np.zeros(self.robot_model.nv)))
        self._ocp = OCPCrocoJointState(self.robot_models, self._ocp_params)
        self._ocp.set_reference_weighted_trajectory(trajectory_points)
        self._ocp.solve(self._state_reg, self._state_warmstart, control_warmstart)

        # Test solution consistency
        for t in range(self._horizon_size - 1):
            for joint_idx in range(self.robot_model.nq):
                with self.subTest(t=t, joint=joint_idx):
                    error = abs(
                        self._state_warmstart[t][joint_idx]
                        - self._ocp.ocp_results.states[t + 1][joint_idx]
                    )
                    self.assertLess(
                        error,
                        1e-2,
                        f"Error at time {t}, joint {joint_idx}: {error:.5f}",
                    )

        # Test trajectories match the reference
        reference_trajectory = [
            self._state_warmstart[t][3] for t in range(self._horizon_size - 1)
        ]
        solution_trajectory = [
            self._ocp.ocp_results.states[t + 1][3]
            for t in range(self._horizon_size - 1)
        ]
        np.testing.assert_allclose(reference_trajectory, solution_trajectory, atol=1e-2)

    def plot(self):
        import matplotlib.pyplot as plt

        plt.plot(
            [self._state_reg[3]]
            + [self._state_warmstart[t][3] for t in range(self._horizon_size - 1)],
            "o-",
            label="Warmstart",
        )
        plt.plot(
            [self._ocp.ocp_results.states[t][3] for t in range(self._horizon_size)],
            "o-",
            label="Solution",
        )
        plt.xlabel("Time")
        plt.ylabel("Joint 3")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    unittest.main()
