import time
import subprocess
import psutil


class ProcessHandler(object):
    def __init__(self, name: str):
        self.process = None
        self.name = name
        self.start()

    def __del__(self):
        self.stop()

    def is_running(self):
        return bool(
            [
                p
                for p in psutil.process_iter()
                if psutil.Process(p.pid).name() == self.name
            ]
        )

    def start(self):
        if self.process is not None and not self.is_running():
            return
        self.process = subprocess.Popen([self.name])
        time.sleep(0.2)

    def stop(self):
        if self.process is None:
            return
        self.process.terminate()
        self.process.kill()
        self.process.wait()
        self.process = None
