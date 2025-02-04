{
  buildPythonPackage,
  colmpc,
  crocoddyl,
  coal,
  example-robot-data,
  franka-description,
  lib,
  mim-solvers,
  numpy,
  pinocchio,
  pip,
  pytestCheckHook,
  rosPackages,
  rospkg,
  setuptools,
}:
buildPythonPackage {
  pname = "agimus-controller";
  version = "0-unstable-2025-01-15";

  src = lib.fileset.toSource {
    root = ./.;
    fileset = lib.fileset.unions [
      ./agimus_controller
      ./package.xml
      ./resource
      ./setup.py
      ./tests
    ];
  };

  build-system = [ setuptools pip ];

  dependencies = [
    colmpc
    crocoddyl
    coal
    example-robot-data
    mim-solvers
    numpy
    pinocchio
    franka-description
    rosPackages.humble.xacro
    rosPackages.humble.ament-index-python
    rospkg
  ];

  nativeCheckInputs = [ pytestCheckHook ];
  doCheck = true;
  pythonImportsCheck = [ "agimus_controller" ];
  dontWrapQtApps = true;

  pytestCheckPhase = ''
    AMENT_PREFIX_PATH=${franka-description.out}:$AMENT_PREFIX_PATH pytest -v -rs
  '';
  # Override the configure phase to prevent CMake from running
  configurePhase = ''
    runHook preConfigure
    echo "Skipping CMake and using Python setup.py for configuration."
    runHook postConfigure
  '';

  buildPhase = ''
    runHook preBuild
    python setup.py sdist bdist_wheel
    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall
    pip install --no-deps --prefix=$out dist/*.whl
    runHook postInstall
  '';

  meta = {
    description = "The agimus_controller package";
    homepage = "https://github.com/agimus-project/agimus_controller";
    license = lib.licenses.bsd3;
    maintainers = [ lib.maintainers.nim65s ];
    platforms = lib.platforms.linux;
  };
}
