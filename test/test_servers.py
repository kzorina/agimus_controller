import unittest
from agimus_controller.main.servers import HppCorbaServer, RosCore, Servers


class TestMains(unittest.TestCase):
    def test_hppcorbaserver(self):
        obj = HppCorbaServer()
        self.assertTrue(obj.is_running())
        obj.stop()
        self.assertFalse(obj.is_running())
        obj.start()
        self.assertTrue(obj.is_running())
        obj.stop()
        self.assertFalse(obj.is_running())

    def test_roscore(self):
        obj = RosCore()
        self.assertTrue(obj.is_running())
        obj.stop()
        self.assertFalse(obj.is_running())
        obj.start()
        self.assertTrue(obj.is_running())
        obj.stop()
        self.assertFalse(obj.is_running())

    def test_servers(self):
        obj = Servers()
        obj.spawn_servers(use_gui=False)
        self.assertTrue(obj.is_running())
        obj.stop()
        self.assertFalse(obj.is_running())
        obj.start()
        self.assertTrue(obj.is_running())
        obj.stop()
        self.assertFalse(obj.is_running())


if __name__ == "__main__":
    unittest.main()
