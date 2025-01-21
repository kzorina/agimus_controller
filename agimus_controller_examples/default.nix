{
  agimus-controller,
  buildPythonPackage,
  hpp-corbaserver,
  hpp-gepetto-viewer,
  hpp-manipulation-corba,
  lib,
  matplotlib,
  numpy,
  pinocchio,
  #pytestCheckHook,
  setuptools,
}:
buildPythonPackage {
  pname = "agimus-controller-examples";
  version = "0-unstable-2025-01-15";

  src = lib.fileset.toSource {
    root = ./.;
    fileset = lib.fileset.unions [
      ./agimus_controller_examples
      ./setup.py
    ];
  };

  build-system = [ setuptools ];

  dependencies = [
    agimus-controller
    hpp-corbaserver
    hpp-manipulation-corba
    hpp-gepetto-viewer
    matplotlib
    numpy
    pinocchio
  ];

  #nativeCheckInputs = [ pytestCheckHook ];
  doCheck = true;
  pythonImportsCheck = [ "agimus_controller_examples" ];

  meta = {
    description = "The agimus_controller package";
    homepage = "https://github.com/agimus-project/agimus_controller";
    license = lib.licenses.bsd3;
    maintainers = [ lib.maintainers.nim65s ];
    platforms = lib.platforms.linux;
  };
}
