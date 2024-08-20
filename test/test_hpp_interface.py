import unittest
from agimus_controller.hpp_interface import HppCorbaServer
from agimus_controller.hpp_interface import GepettoGuiServer
# from agimus_controller.hpp_interface import HppInterface


class TestMains(unittest.TestCase):
    def test_spawn_hppcorbaserver(self):
        obj = HppCorbaServer()
        self.assertTrue(obj.is_running())
        obj.stop()
        self.assertFalse(obj.is_running())
        obj.start()
        self.assertTrue(obj.is_running())
        obj.stop()
        self.assertFalse(obj.is_running())

    def test_spawn_gepetto_gui(self):
        obj = GepettoGuiServer()
        self.assertTrue(obj.is_running())
        obj.stop()
        self.assertFalse(obj.is_running())
        obj.start()
        self.assertTrue(obj.is_running())
        obj.stop()
        self.assertFalse(obj.is_running())


if __name__ == "__main__":
    unittest.main()
