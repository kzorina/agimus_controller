import rospkg
from pathlib import Path


def get_package_path(package_name) -> Path:
    rospack = rospkg.RosPack()
    return Path(rospack.get_path(package_name))
