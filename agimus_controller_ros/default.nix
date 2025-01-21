{
  agimus-controller,
  buildPythonPackage,
  lib,
  linear-feedback-controller-msgs,
  numpy,
  pinocchio,
  pytestCheckHook,
  rosPackages,
  setuptools,
}:
buildPythonPackage {
  pname = "agimus-controller-ros";
  version = "0-unstable-2025-01-15";

  src = lib.fileset.toSource {
    root = ./.;
    fileset = lib.fileset.unions [
      ./agimus_controller_ros
      ./package.xml
      ./resource
      ./setup.py
      ./test
    ];
  };

  build-system = [ setuptools ];

  dependencies = [
    agimus-controller
    linear-feedback-controller-msgs
    numpy
    pinocchio
    rosPackages.humble.ament-copyright
    rosPackages.humble.ament-flake8
    rosPackages.humble.ament-pep257
    rosPackages.humble.generate-parameter-library-py
    rosPackages.humble.geometry-msgs
    rosPackages.humble.rclpy
    rosPackages.humble.std-msgs
  ];

  dontUseCmakeConfigure = true; # Something is propagating cmake…

  doCheck = true;
  nativeCheckInputs = [ pytestCheckHook ];
  disabledTests = [
    # ref. https://github.com/agimus-project/agimus_controller/issues/127
    "test_pep257"
  ];
  pythonImportsCheck = [ "agimus_controller_ros" ];

  meta = {
    description = "The agimus_controller package";
    homepage = "https://github.com/agimus-project/agimus_controller";
    license = lib.licenses.bsd3;
    maintainers = [ lib.maintainers.nim65s ];
    platforms = lib.platforms.linux;
  };
}
