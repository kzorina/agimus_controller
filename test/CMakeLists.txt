if(BUILD_ROS_INTERFACE)

  find_package(rostest REQUIRED)
  add_rostest(ros/subscribers.launch.test)

  add_python_unit_test(test_build_model test/test_build_model.py)
  add_python_unit_test(test_run_mains test/test_run_mains.py)

endif()
