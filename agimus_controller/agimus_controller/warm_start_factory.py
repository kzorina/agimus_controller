from .warm_start_base import WarmStartBase


class WarmStartFactory(object):
    def __init__(self) -> None:
        pass

    def create_warm_start(self, name: str) -> WarmStartBase:
        if name == "from_previous_solution":
            return None

        if name == "from_diffusion_model":
            return None
