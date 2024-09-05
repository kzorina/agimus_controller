import time
import subprocess
import psutil
import os


class ProcessHandler(object):
    def __init__(self, name: str) -> None:
        self.process = None
        self.name = name
        self.start()

    def __del__(self) -> None:
        print("Killing the ", self.name, " process.")
        self.stop()

    def is_running(self) -> bool:
        return bool(
            [
                p
                for p in psutil.process_iter()
                if psutil.Process(p.pid).name() == self.name
            ]
        )

    def start(self) -> None:
        if self.process is not None and self.is_running():
            return
        self.process = subprocess.Popen([self.name], env=os.environ)
        time.sleep(0.2)

    def stop(self) -> None:
        if self.process is None:
            return
        trials = 0
        while self.is_running() and trials < 5:
            self.process.terminate()
            self.process.wait()
            trials += 1
        if trials >= 5:
            trials = 0
            while self.is_running() and trials < 5:
                self.process.kill()
                self.process.wait()
                trials += 1
            if trials >= 5:
                raise RuntimeError("Tried 5 times to shutdown the process...")
        self.process = None


class GepettoGuiServer(ProcessHandler):
    def __init__(self) -> None:
        super().__init__("gepetto-gui")

    def stop(self) -> None:
        if self.process is None:
            return
        self.process.kill()
        self.process.wait()
        self.process = None


class HppCorbaServer(ProcessHandler):
    def __init__(self) -> None:
        super().__init__("hppcorbaserver")


class RosCore(ProcessHandler):
    def __init__(self) -> None:
        super().__init__("roscore")


class MeshcatServer(ProcessHandler):
    def __init__(self) -> None:
        super().__init__("meshcat-server")
