from setuptools import setup

setup(
    name='agimus_controller_ros',
    version='0.0.0',
    packages=["agimus_controller_ros"],
    package_dir={
        
    },
    install_requires=['setuptools'],
    zip_safe=True,
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + 'agimus_controller_ros']),
        ('share/' + 'agimus_controller_ros', ['package.xml']),
    ],
    maintainer='gepetto',
    maintainer_email='tmartinezf@laas.fr',
    description='The ROS agimus_controller package',
    license='BSD',
    entry_points={
        "console_scripts": [
            "mpc_input_dummy_publisher = agimus_controller_ros.mpc_input_dummy_publisher:main",
            "agimus_controller_node = agimus_controller_ros.agimus_controller:main"
        ],
    },
)
