// Copyright 2021 Open Source Robotics Foundation, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef GZ_ROS2_CONTROL__GZ_ROS2_CONTROL_PLUGIN_HPP_
#define GZ_ROS2_CONTROL__GZ_ROS2_CONTROL_PLUGIN_HPP_

#include <gz/sim/System.hh>
#include <memory>
#include <string>
#include "gz_ros2_control/model_kdl.h"

namespace gz_ros2_control {
// Forward declarations.
class GZROS2ControlPluginPrivate;

class GZROS2ControlPlugin : public gz::sim::System,
                            public gz::sim::ISystemConfigure,
                            public gz::sim::ISystemPreUpdate,
                            public gz::sim::ISystemPostUpdate {
 public:
  /// \brief Constructor
  GZROS2ControlPlugin();

  /// \brief Destructor
  ~GZROS2ControlPlugin() override;

  // Documentation inherited
  void Configure(const gz::sim::Entity& _entity,
                 const std::shared_ptr<const sdf::Element>& _sdf,
                 gz::sim::EntityComponentManager& _ecm,
                 gz::sim::EventManager& _eventMgr) override;

  // Documentation inherited
  void PreUpdate(const gz::sim::UpdateInfo& _info, gz::sim::EntityComponentManager& _ecm) override;

  void PostUpdate(const gz::sim::UpdateInfo& _info,
                  const gz::sim::EntityComponentManager& _ecm) override;

 private:
  /// @brief gets the root link of the urdf model
  /// @param model urdf model of the robot
  /// @return root link of the robot
  urdf::LinkConstSharedPtr getRootLink(const urdf::Model& model);

  /// @brief gets the tip link of the urdf model
  /// @param model urdf model of the robot
  /// @return tip link of the robot
  std::string findTipLink(const urdf::Model& model);

  /// \brief Private data pointer.
  std::unique_ptr<GZROS2ControlPluginPrivate> dataPtr;

  /// \brief KDL model built from the URDF model
  ModelKDL kdl_model_;
};
}  // namespace gz_ros2_control

#endif  // GZ_ROS2_CONTROL__GZ_ROS2_CONTROL_PLUGIN_HPP_
