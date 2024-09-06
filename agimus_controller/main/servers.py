from agimus_controller.utils.process_handler import (
    HppCorbaServer,
    GepettoGuiServer,
    RosCore,
    MeshcatServer,
)


class Servers(object):
    def __init__(self) -> None:
        self._servers = []

    def is_running(self):
        is_running = True
        for server in self._servers:
            is_running = is_running and server.is_running()
        return is_running

    def start(self):
        for server in self._servers:
            server.start()

    def stop(self):
        for server in self._servers:
            server.stop()

    def spawn_servers(self, use_gui):
        self._servers.append(HppCorbaServer())
        if use_gui:
            self._servers.append(GepettoGuiServer())
        self._servers.append(RosCore())

    def spawn_servers_meshcat(self, use_gui):
        self._servers.append(HppCorbaServer())
        if use_gui:
            self._servers.append(MeshcatServer())
        self._servers.append(RosCore())
