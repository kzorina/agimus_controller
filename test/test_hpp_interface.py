import unittest
from agimus_controller.hpp_interface import HppInterface


class TestMains(unittest.TestCase):
    def test_spwan_hppcorbaserver(self):
        obj = HppInterface()
        self.assertFalse(HppInterface.is_hppcorbaserver_running())
        obj.start_corbaserver()
        self.assertTrue(HppInterface.is_hppcorbaserver_running())
        obj.stop_corbaserver()
        self.assertFalse(HppInterface.is_hppcorbaserver_running())


if __name__ == "__main__":
    unittest.main()
