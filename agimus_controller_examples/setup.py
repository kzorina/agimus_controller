from setuptools import find_packages, setup

package_name = "agimus_controller_examples"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name] + [package_name + '.' + pkg for pkg in ["hpp_panda", "main","main/panda","main/ur3","resources","utils"]],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="gepetto",
    maintainer_email="theo.martinez-fouche@laas.fr",
    description="TODO: Package description",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [],
    },
)
