if(BUILD_ROS_INTERFACE)

  find_package(rostest REQUIRED)
  add_rostest(subscribers.launch.test)

endif()
