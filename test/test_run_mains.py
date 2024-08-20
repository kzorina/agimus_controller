import unittest

from agimus_controller.main.panda.main_hpp_mpc_buffer import (
    main as main_panda_hpp_mpc_buffer,
)
from agimus_controller.main.panda.main_hpp_mpc import main as main_panda_hpp_mpc
from agimus_controller.main.panda.main_meshcat_display import (
    main as main_panda_meshcat_display,
)
from agimus_controller.main.panda.main_mpc import main as main_panda_mpc
from agimus_controller.main.panda.main_optim_traj import main as main_panda_optim_traj

from agimus_controller.main.ur3.main_hpp_mpc import main as main_ur3_hpp_mpc


class TestMains(unittest.TestCase):
    def test_main_panda_hpp_mpc_buffer(self):
        self.assertTrue(main_panda_hpp_mpc_buffer())

    def test_main_panda_hpp_mpc(self):
        self.assertTrue(main_panda_hpp_mpc())

    def test_main_panda_meshcat_display(self):
        self.assertTrue(main_panda_meshcat_display())

    def test_main_panda_mpc(self):
        self.assertTrue(main_panda_mpc())

    def test_main_panda_optim_traj(self):
        self.assertTrue(main_panda_optim_traj())

    def test_main_ur3_hpp_mpc(self):
        self.assertTrue(main_ur3_hpp_mpc())


if __name__ == "__main__":
    unittest.main()
