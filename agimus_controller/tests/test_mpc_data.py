import numpy as np
import random
import unittest


from agimus_controller.mpc_data import MPCDebugData, OCPDebugData, OCPResults


class TestOCPParamsCrocoBase(unittest.TestCase):
    """
    TestOCPParamsCrocoBase unittests parameters settters and getters of OCPParamsBaseCroco class.
    """

    def __init__(self, methodName="runTest"):
        super().__init__(methodName)

    def test_ocp_results(self):
        """
        Test the initialization of the MPCDebugData class.
        """
        N = random.randint(10, 100)
        rows = random.randint(10, 20)
        cols = 2 * rows
        params = {
            "states": [np.random.rand(cols) for _ in range(N)],
            "ricatti_gains": [np.random.rand(rows, cols) for _ in range(N)],
            "feed_forward_terms": [np.random.rand(rows) for _ in range(N)],
        }
        obj = OCPResults(**deepcopy(params))
        for key, val in params.items():
            res = getattr(obj, key)
            self.assertEqual(
                res,
                val,
                f"Value missmatch for parameter '{key}'. Expected: '{val}', got: '{res}'",
            )

    def test_ocp_debug_data(self):
        """
        Test the initialization of the MPCDebugData class.
        """
        params = {
            "result": list(),
            "references": list(),
            "kkt_norms": list(),
            "collision_distance_residuals": list(),
            "problem_solved": False,
        }
        obj = OCPDebugData(**params)
        for key, val in params.items():
            res = getattr(obj, key)
            self.assertEqual(
                res,
                val,
                f"Value missmatch for parameter '{key}'. Expected: '{val}', got: '{res}'",
            )

    def test_ocp_debug_data(self):
        """
        Test the initialization of the MPCDebugData class.
        """
        params = {
            "ocp": OCPDebugData(
                **{
                    "result": list(),
                    "references": list(),
                    "kkt_norms": list(),
                    "collision_distance_residuals": list(),
                    "problem_solved": False,
                }
            ),
            "duration_iteration_ns": random.randint(0, 10000),
            "duration_horizon_update_ns": random.randint(0, 10000),
            "duration_generate_warm_start_ns": random.randint(0, 10000),
            "duration_ocp_solve_ns": random.randint(0, 10000),
        }
        obj = MPCDebugData(**params)
        for key, val in params.items():
            res = getattr(obj, key)
            self.assertEqual(
                res,
                val,
                f"Value missmatch for parameter '{key}'. Expected: '{val}', got: '{res}'",
            )


if __name__ == "__main__":
    unittest.main()
