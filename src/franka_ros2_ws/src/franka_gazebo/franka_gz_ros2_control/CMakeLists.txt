cmake_minimum_required(VERSION 3.5)

# Suppress CMP0002 warning
option(CHECK_TIDY "Enable clang-tidy checks" OFF)

project(franka_gz_ros2_control)

# Default to C11
if(NOT CMAKE_C_STANDARD)
    set(CMAKE_C_STANDARD 11)
endif()
# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif()

# Compiler options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(controller_manager REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(yaml_cpp_vendor REQUIRED)
find_package(franka_hardware REQUIRED)
find_package(urdf REQUIRED)
find_package(kdl_parser REQUIRED)
find_package(Franka 0.19.0 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)

find_package(gz_sim_vendor REQUIRED)
find_package(gz-sim REQUIRED)
find_package(gz_plugin_vendor REQUIRED)
find_package(gz-plugin REQUIRED)

include_directories(include)

add_library(${PROJECT_NAME}-system SHARED
  src/gz_ros2_control_plugin.cpp
  src/model_kdl.cpp
)
target_link_libraries(${PROJECT_NAME}-system
  gz-plugin::gz-plugin
  gz-plugin::core
  gz-sim::gz-sim
)
ament_target_dependencies(${PROJECT_NAME}-system
  ament_index_cpp
  controller_manager
  hardware_interface
  pluginlib
  rclcpp
  yaml_cpp_vendor
  rclcpp_lifecycle
  franka_hardware
  Franka
  urdf
  kdl_parser
  tf2_geometry_msgs
  tf2_eigen
)

#########

add_library(franka_gz_hardware_plugins SHARED
  src/gz_system.cpp
  src/model_kdl.cpp
)
target_link_libraries(franka_gz_hardware_plugins
  gz-plugin::gz-plugin
  gz-plugin::core
  gz-sim::gz-sim
)
ament_target_dependencies(franka_gz_hardware_plugins
  rclcpp_lifecycle
  hardware_interface
  rclcpp
  franka_hardware
  Franka
  urdf
  kdl_parser
  tf2_geometry_msgs
  tf2_eigen
  pluginlib
)


## Install
install(TARGETS
  franka_gz_hardware_plugins
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY
  include/
  DESTINATION include
)

# Testing and linting
if(BUILD_TESTING)
  find_package(ament_cmake_clang_format REQUIRED)
  find_package(ament_cmake_copyright REQUIRED)
  find_package(ament_cmake_cppcheck REQUIRED)
  find_package(ament_cmake_flake8 REQUIRED)
  find_package(ament_cmake_lint_cmake REQUIRED)
  find_package(ament_cmake_pep257 REQUIRED)
  find_package(ament_cmake_xmllint REQUIRED)
  find_package(ament_cmake_gmock REQUIRED)

  set(CPP_DIRECTORIES src include)
  ament_clang_format(CONFIG_FILE ../../.clang-format ${CPP_DIRECTORIES})
  ament_copyright(src ${CPP_DIRECTORIES} package.xml)
  ament_cppcheck(${CPP_DIRECTORIES})
  ament_lint_cmake(CMakeLists.txt)
  ament_flake8()
  ament_pep257()
  ament_xmllint()
  if(CHECK_TIDY)
    find_package(ament_cmake_clang_tidy REQUIRED)
    set(ament_cmake_clang_tidy_CONFIG_FILE ../../.clang-tidy)
    ament_clang_tidy(${CMAKE_BINARY_DIR} --jobs 6)
  endif()
endif()

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME}-system franka_gz_hardware_plugins)

# Install directories
install(TARGETS ${PROJECT_NAME}-system
  DESTINATION lib
)

pluginlib_export_plugin_description_file(${PROJECT_NAME} franka_gz_hardware_plugins.xml)

# Setup the project
ament_package()
