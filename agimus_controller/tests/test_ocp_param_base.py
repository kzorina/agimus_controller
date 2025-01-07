import unittest
import numpy as np

from agimus_controller.ocp_param_base import OCPParamsBaseCroco


class TestOCPParamsCrocoBase(unittest.TestCase):
    """
    TestOCPParamsCrocoBase unittests parameters settters and getters of OCPParamsBaseCroco class.

    Methods:
        __init__(methodName="runTest"): Initializes the test case instance.
        test_initialization(): Tests the initialization of the OCPParamsBaseCroco class with various parameters.
    """

    """Test the OCPParamsBaseCroco class."""

    def __init__(self, methodName="runTest"):
        super().__init__(methodName)

    def test_initialization(self):
        """
        Test the initialization of the OCPParamsBaseCroco class.
        """
        params = {
            "dt": np.float64(0.01),
            "horizon_size": 100,
            "solver_iters": 50,
            "qp_iters": 200,
            "termination_tolerance": 1e-3,
            "eps_abs": 1e-6,
            "eps_rel": 0,
            "callbacks": False,
        }
        params = OCPParamsBaseCroco(**params)
        for key, val in params:
            res = getattr(params, key)
            self.assertEqual(
                res,
                val,
                f"Value missmatch for parameter '{key}'. Expected: '{val}', got: '{res}'",
            )


if __name__ == "__main__":
    unittest.main()
