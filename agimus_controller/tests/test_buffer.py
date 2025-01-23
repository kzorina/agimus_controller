from copy import deepcopy
import numpy as np
import numpy.typing as npt
import random
import unittest


from agimus_controller.trajectory import (
    TrajectoryBuffer,
    WeightedTrajectoryPoint,
)


class TestTrajectoryBuffer(unittest.TestCase):
    """
    TestOCPParamsCrocoBase unittests parameters settters and getters of OCPParamsBaseCroco class.
    """

    def __init__(self, methodName="runTest"):
        super().__init__(methodName)

    def generate_random_weighted_states(self):
        """
        Generate random data for the TrajectoryPointWeights.
        """
        pass

    def test_ocp_results(self):
        """
        Test the initialization of the OCPResults class.
        """
        pass


if __name__ == "__main__":
    unittest.main()
