from setuptools import find_packages, setup

package_name = "agimus_controller_ros"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="gepetto",
    maintainer_email="kateryna.zorina@cvut.cz",
    description="TODO: Package description",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "mpc_input_dummy_publisher = agimus_controller_ros.mpc_input_dummy_publisher:main",
            "test_buffer = agimus_controller_ros.test_buffer_node:main",
        ],
    },
)
