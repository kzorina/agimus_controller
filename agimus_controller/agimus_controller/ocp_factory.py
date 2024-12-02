from agimus_controller.ocp_base import OCPBase

class OCPFactory(object):
    def __init__(self) -> None:
        pass

    def create_ocp(self, name: str) -> OCPBase:

        if name == "hpp_crocco":
            return None
        
        if name == "collision_avoidance":
            return None
        
        if name == "single_ee_ref":
            return None