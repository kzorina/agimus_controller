import unittest
from parameterized import parameterized

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
    @parameterized.expand(
        [
            (f.__name__, f)
            for f in [
                main_panda_hpp_mpc_buffer,
                main_panda_hpp_mpc,
                main_panda_meshcat_display,
                main_panda_mpc,
                main_panda_optim_traj,
                main_ur3_hpp_mpc,
            ]
        ]
    )
    def test_mains(self, name, main):
        self.assertTrue(main())


if __name__ == "__main__":
    unittest.main()
