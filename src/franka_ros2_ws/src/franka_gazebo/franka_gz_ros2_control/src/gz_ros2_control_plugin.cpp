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
#include "gz_ros2_control/gz_ros2_control_plugin.hpp"

#include <unistd.h>

#include <chrono>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gz/sim/Model.hh>
#include <gz/sim/components/Joint.hh>
#include <gz/sim/components/JointType.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/ParentEntity.hh>
#include <gz/sim/components/World.hh>

#include <gz/plugin/Register.hh>

#include <controller_manager/controller_manager.hpp>

#include <hardware_interface/component_parser.hpp>
#include <hardware_interface/resource_manager.hpp>
#include <hardware_interface/types/hardware_interface_type_values.hpp>

#include <pluginlib/class_loader.hpp>

#include <rclcpp/logger.hpp>
#include <rclcpp/rclcpp.hpp>

#include "gz_ros2_control/gz_system.hpp"
#include "gz_ros2_control/model_kdl.h"

namespace gz_ros2_control {
class GZResourceManager : public hardware_interface::ResourceManager {
 public:
  GZResourceManager(ModelKDL& kdl_model,
                    rclcpp::Node::SharedPtr& node,
                    gz::sim::EntityComponentManager& ecm,
                    std::map<std::string, gz::sim::Entity> enabledJoints)
      : hardware_interface::ResourceManager(node->get_node_clock_interface(),
                                            node->get_node_logging_interface()),
        ecm_(ecm),
        gz_system_loader_("franka_gz_ros2_control", "gz_ros2_control::GZSystemInterface"),
        logger_(node->get_logger().get_child("GZResourceManager")) {
    node_ = node;
    enabledJoints_ = enabledJoints;
    kdl_model_ = kdl_model;
  }

  GZResourceManager(const GZResourceManager&) = delete;

  bool load_and_initialize_components(const std::string& urdf,
                                      const unsigned int update_rate) override {
    components_are_loaded_and_initialized_ = true;
    const auto hardware_info = hardware_interface::parse_control_resources_from_urdf(urdf);

    for (const auto& individual_hardware_info : hardware_info) {
      std::string robot_hw_sim_type_str_ = individual_hardware_info.hardware_plugin_name;
      RCLCPP_DEBUG(logger_, "Load hardware interface %s ...", robot_hw_sim_type_str_.c_str());

      std::unique_ptr<gz_ros2_control::GZSystemInterface> gzSimSystem;
      std::scoped_lock guard(resource_interfaces_lock_, claimed_command_interfaces_lock_);
      try {
        gzSimSystem = std::unique_ptr<gz_ros2_control::GZSystemInterface>(
            gz_system_loader_.createUnmanagedInstance(robot_hw_sim_type_str_));
      } catch (pluginlib::PluginlibException& ex) {
        RCLCPP_ERROR(logger_, "The plugin failed to load for some reason. Error: %s\n", ex.what());
        continue;
      }
      if (!gzSimSystem->initSim(kdl_model_, node_, enabledJoints_, individual_hardware_info, ecm_,
                                update_rate)) {
        RCLCPP_FATAL(logger_, "Could not initialize robot simulation interface");
        components_are_loaded_and_initialized_ = false;
        break;
      }
      RCLCPP_DEBUG(logger_, "Initialized robot simulation interface %s!",
                   robot_hw_sim_type_str_.c_str());

      import_component(std::move(gzSimSystem), individual_hardware_info);
    }

    return components_are_loaded_and_initialized_;
  }

 private:
  std::shared_ptr<rclcpp::Node> node_;
  gz::sim::EntityComponentManager& ecm_;
  std::map<std::string, gz::sim::Entity> enabledJoints_;
  ModelKDL kdl_model_;

  pluginlib::ClassLoader<gz_ros2_control::GZSystemInterface> gz_system_loader_;

  rclcpp::Logger logger_;
};
//////////////////////////////////////////////////
class GZROS2ControlPluginPrivate {
 public:
  /// \brief Get the URDF XML from the parameter server
  [[nodiscard]] std::string getURDF() const;

  /// \brief Get a list of enabled, unique, 1-axis joints of the model. If no
  /// joint names are specified in the plugin configuration, all valid 1-axis
  /// joints are returned
  /// \param[in] _entity Entity of the model that the plugin is being
  /// configured for
  /// \param[in] _ecm Gazebo Entity Component Manager
  /// \return List of entities containing all enabled joints
  std::map<std::string, gz::sim::Entity> GetEnabledJoints(
      const gz::sim::Entity& _entity,
      gz::sim::EntityComponentManager& _ecm) const;

  // NOLINTBEGIN(misc-non-private-member-variables-in-classes)

  /// \brief Entity ID for sensor within Gazebo.
  gz::sim::Entity entity_;

  /// \brief Node Handles
  std::shared_ptr<rclcpp::Node> node_{nullptr};

  /// \brief Thread where the executor will spin
  std::thread thread_executor_spin_;

  /// \brief Executor to spin the controller
  rclcpp::executors::MultiThreadedExecutor::SharedPtr executor_;

  /// \brief Timing
  rclcpp::Duration control_period_ = rclcpp::Duration(1, 0);

  /// \brief Interface loader
  std::shared_ptr<pluginlib::ClassLoader<gz_ros2_control::GZSystemInterface>> robot_hw_sim_loader_{
      nullptr};

  /// \brief Controller manager
  std::shared_ptr<controller_manager::ControllerManager> controller_manager_{nullptr};

  /// \brief String with the robot description param_name
  std::string robot_description_ = "robot_description";

  /// \brief String with the name of the node that contains the
  /// robot_description
  std::string robot_description_node_ = "robot_state_publisher";

  /// \brief Last time the update method was called
  rclcpp::Time last_update_sim_time_ros_ = rclcpp::Time(static_cast<int64_t>(0), RCL_ROS_TIME);

  /// \brief ECM pointer
  gz::sim::EntityComponentManager* ecm{nullptr};

  /// \brief controller update rate
  int update_rate;
  // NOLINTEND(misc-non-private-member-variables-in-classes)
};

//////////////////////////////////////////////////
std::map<std::string, gz::sim::Entity> GZROS2ControlPluginPrivate::GetEnabledJoints(
    const gz::sim::Entity& _entity,
    gz::sim::EntityComponentManager& _ecm) const {
  std::map<std::string, gz::sim::Entity> output;

  std::vector<std::string> enabledJoints;

  // Get all available joints
  auto jointEntities = _ecm.ChildrenByComponents(_entity, gz::sim::components::Joint());

  // Iterate over all joints and verify whether they can be enabled or not
  for (const auto& jointEntity : jointEntities) {
    const auto jointName = _ecm.Component<gz::sim::components::Name>(jointEntity)->Data();

    // Make sure the joint type is supported, i.e. it has exactly one
    // actuated axis
    const auto* jointType = _ecm.Component<gz::sim::components::JointType>(jointEntity);
    switch (jointType->Data()) {
      case sdf::JointType::PRISMATIC:
      case sdf::JointType::REVOLUTE:
      case sdf::JointType::CONTINUOUS:
      case sdf::JointType::GEARBOX: {
        // Supported joint type
        break;
      }
      case sdf::JointType::FIXED: {
        RCLCPP_INFO(node_->get_logger(),
                    "[gz_ros2_control] Fixed joint [%s] (Entity=%lu)] is skipped",
                    jointName.c_str(), jointEntity);
        continue;
      }
      case sdf::JointType::REVOLUTE2:
      case sdf::JointType::SCREW:
      case sdf::JointType::BALL:
      case sdf::JointType::UNIVERSAL: {
        RCLCPP_WARN(node_->get_logger(),
                    "[gz_ros2_control] Joint [%s] (Entity=%lu)] is of unsupported type."
                    " Only joints with a single axis are supported.",
                    jointName.c_str(), jointEntity);
        continue;
      }
      default: {
        RCLCPP_WARN(node_->get_logger(),
                    "[gz_ros2_control] Joint [%s] (Entity=%lu)] is of unknown type",
                    jointName.c_str(), jointEntity);
        continue;
      }
    }
    output[jointName] = jointEntity;
  }

  return output;
}

//////////////////////////////////////////////////
std::string GZROS2ControlPluginPrivate::getURDF() const {
  std::string urdf_string;

  using namespace std::chrono_literals;
  auto parameters_client =
      std::make_shared<rclcpp::AsyncParametersClient>(node_, robot_description_node_);
  while (!parameters_client->wait_for_service(0.5s)) {
    if (!rclcpp::ok()) {
      RCLCPP_ERROR(node_->get_logger(), "Interrupted while waiting for %s service. Exiting.",
                   robot_description_node_.c_str());
      return {};
    }
    RCLCPP_ERROR(node_->get_logger(), "%s service not available, waiting again...",
                 robot_description_node_.c_str());
  }

  RCLCPP_INFO(node_->get_logger(), "connected to service!! %s asking for %s",
              robot_description_node_.c_str(), this->robot_description_.c_str());

  // search and wait for robot_description on param server
  while (urdf_string.empty()) {
    RCLCPP_DEBUG(node_->get_logger(), "param_name %s", this->robot_description_.c_str());

    try {
      auto f = parameters_client->get_parameters({this->robot_description_});
      f.wait();
      std::vector<rclcpp::Parameter> values = f.get();
      urdf_string = values[0].as_string();
    } catch (const std::exception& e) {
      RCLCPP_ERROR(node_->get_logger(), "%s", e.what());
    }

    if (!urdf_string.empty()) {
      break;
    } else {
      RCLCPP_ERROR(node_->get_logger(),
                   "gz_ros2_control plugin is waiting for model"
                   " URDF in parameter [%s] on the ROS param server.",
                   this->robot_description_.c_str());
    }
    std::this_thread::sleep_for(std::chrono::microseconds(100000));
  }
  RCLCPP_INFO(node_->get_logger(), "Received URDF from param server");

  return urdf_string;
}

//////////////////////////////////////////////////
GZROS2ControlPlugin::GZROS2ControlPlugin()
    : dataPtr(std::make_unique<GZROS2ControlPluginPrivate>()) {}

//////////////////////////////////////////////////
GZROS2ControlPlugin::~GZROS2ControlPlugin() {
  // Stop controller manager thread
  if (!this->dataPtr->controller_manager_) {
    return;
  }
  this->dataPtr->executor_->remove_node(this->dataPtr->controller_manager_);
  this->dataPtr->executor_->cancel();
  this->dataPtr->thread_executor_spin_.join();
}

// Function to get the root link from the URDF model
urdf::LinkConstSharedPtr GZROS2ControlPlugin::getRootLink(const urdf::Model& model) {
  for (const auto& link_pair : model.links_) {
    if (!link_pair.second->parent_joint) {
      return link_pair.second;
    }
  }
  return nullptr;
}

// Function to iteratively find the tip link in the URDF model
std::string GZROS2ControlPlugin::findTipLink(const urdf::Model& model) {
  urdf::LinkConstSharedPtr root_link = getRootLink(model);
  if (!root_link) {
    std::cerr << "No root link found in the URDF model." << std::endl;
    return "";
  }

  std::queue<urdf::LinkConstSharedPtr> link_queue;
  link_queue.push(root_link);
  std::string tip_link;

  while (!link_queue.empty()) {
    urdf::LinkConstSharedPtr current_link = link_queue.front();
    link_queue.pop();

    // If the link has no child links, it is a tip link
    if (current_link->child_links.empty()) {
      tip_link = current_link->name;
    } else {
      // Add all child links to the queue
      for (const auto& child : current_link->child_links) {
        link_queue.push(child);
      }
    }
  }

  return tip_link;
}

//////////////////////////////////////////////////
void GZROS2ControlPlugin::Configure(const gz::sim::Entity& _entity,
                                    const std::shared_ptr<const sdf::Element>& _sdf,
                                    gz::sim::EntityComponentManager& _ecm,
                                    gz::sim::EventManager&) {
  rclcpp::Logger logger = rclcpp::get_logger("GazeboSimROS2ControlPlugin");
  // Make sure the controller is attached to a valid model
  const auto model = gz::sim::Model(_entity);
  if (!model.Valid(_ecm)) {
    RCLCPP_ERROR(logger,
                 "[Gazebo ROS 2 Control] Failed to initialize because [%s] "
                 "(Entity=%lu)] is not a model."
                 "Please make sure that Gazebo ROS 2 Control is attached to "
                 "a valid model.",
                 model.Name(_ecm).c_str(), _entity);
    return;
  }

  // Get params from SDF
  auto paramFileName = _sdf->Get<std::string>("parameters");

  if (paramFileName.empty()) {
    RCLCPP_ERROR(logger,
                 "Gazebo ros2 control found an empty parameters "
                 "file. Failed to initialize.");
    return;
  }

  // Get params from SDF
  auto robot_param_node = _sdf->Get<std::string>("robot_param_node");
  if (!robot_param_node.empty()) {
    this->dataPtr->robot_description_node_ = robot_param_node;
  }
  RCLCPP_INFO(logger, "robot_param_node is %s", this->dataPtr->robot_description_node_.c_str());

  auto robot_description = _sdf->Get<std::string>("robot_param");
  if (!robot_description.empty()) {
    this->dataPtr->robot_description_ = robot_description;
  }
  RCLCPP_INFO(logger, "robot_param_node is %s", this->dataPtr->robot_description_.c_str());

  std::vector<std::string> arguments = {"--ros-args"};

  auto sdfPtr = const_cast<sdf::Element*>(_sdf.get());

  sdf::ElementPtr argument_sdf = sdfPtr->GetElement("parameters");
  while (argument_sdf) {
    auto argument = argument_sdf->Get<std::string>();
    arguments.push_back(RCL_PARAM_FILE_FLAG);
    arguments.push_back(argument);
    argument_sdf = argument_sdf->GetNextElement("parameters");
  }

  // Get controller manager node name
  std::string controllerManagerNodeName{"controller_manager"};

  if (sdfPtr->HasElement("controller_manager_name")) {
    controllerManagerNodeName = sdfPtr->GetElement("controller_manager_name")->Get<std::string>();
  }

  std::string ns = "/";
  if (sdfPtr->HasElement("ros")) {
    sdf::ElementPtr sdfRos = sdfPtr->GetElement("ros");

    // Set namespace if tag is present
    if (sdfRos->HasElement("namespace")) {
      ns = sdfRos->GetElement("namespace")->Get<std::string>();
      // prevent exception: namespace must be absolute, it must lead with a '/'
      if (ns.empty() || ns[0] != '/') {
        ns = '/' + ns;
      }
      if (ns.length() > 1) {
        this->dataPtr->robot_description_node_ = ns + "/" + this->dataPtr->robot_description_node_;
      }
    }

    // Get list of remapping rules from SDF
    if (sdfRos->HasElement("remapping")) {
      sdf::ElementPtr argument_sdf = sdfRos->GetElement("remapping");

      arguments.push_back(RCL_ROS_ARGS_FLAG);
      while (argument_sdf) {
        auto argument = argument_sdf->Get<std::string>();
        arguments.push_back(RCL_REMAP_FLAG);
        arguments.push_back(argument);
        argument_sdf = argument_sdf->GetNextElement("remapping");
      }
    }
  }

  std::vector<const char*> argv;
  for (const auto& arg : arguments) {
    argv.push_back(reinterpret_cast<const char*>(arg.data()));
  }

  // Create a default context, if not already
  if (!rclcpp::ok()) {
    rclcpp::init(static_cast<int>(argv.size()), argv.data());
  }

  std::string node_name = "gz_ros2_control";

  this->dataPtr->node_ = rclcpp::Node::make_shared(node_name, ns);
  this->dataPtr->executor_ = std::make_shared<rclcpp::executors::MultiThreadedExecutor>();
  this->dataPtr->executor_->add_node(this->dataPtr->node_);
  auto spin = [this]() { this->dataPtr->executor_->spin(); };
  this->dataPtr->thread_executor_spin_ = std::thread(spin);

  RCLCPP_DEBUG_STREAM(this->dataPtr->node_->get_logger(),
                      "[Gazebo ROS 2 Control] Setting up controller for ["
                          << model.Name(_ecm) << "] (Entity=" << _entity << ")].");

  // Get list of enabled joints
  auto enabledJoints = this->dataPtr->GetEnabledJoints(_entity, _ecm);

  if (enabledJoints.size() == 0) {
    RCLCPP_DEBUG_STREAM(this->dataPtr->node_->get_logger(),
                        "[Gazebo ROS 2 Control] There are no available Joints.");
    return;
  }

  // Read urdf from ros parameter server then
  // setup actuators and mechanism control node.
  // This call will block if ROS is not properly initialized.
  std::string urdf_string;
  try {
    urdf_string = this->dataPtr->getURDF();
    // convert urdf_string to urdf::Model
    urdf::Model model;
    model.initString(urdf_string);

    std::string root_link = model.getRoot()->name;
    std::string tip_link = findTipLink(model);

    kdl_model_ = ModelKDL(model, root_link, tip_link);
  } catch (const std::runtime_error& ex) {
    RCLCPP_ERROR_STREAM(
        this->dataPtr->node_->get_logger(),
        "Error parsing URDF in gz_ros2_control plugin, plugin not active : " << ex.what());
    return;
  }

  std::unique_ptr<hardware_interface::ResourceManager> resource_manager_ =
      std::make_unique<gz_ros2_control::GZResourceManager>(kdl_model_, this->dataPtr->node_, _ecm,
                                                           enabledJoints);

  // Create the controller manager
  RCLCPP_INFO(this->dataPtr->node_->get_logger(), "Loading controller_manager");
  rclcpp::NodeOptions cm_options = controller_manager::get_cm_node_options();
  // cm_options.arguments({"--ros-args", "-r", "robot_description:=/robot_description"});
  arguments.push_back("-r");
  arguments.push_back("__node:=" + controllerManagerNodeName);
  arguments.push_back("-r");
  arguments.push_back("__ns:=" + ns);
  arguments.push_back("-p");
  arguments.push_back("use_sim_time:=true");
  cm_options.arguments(arguments);
  this->dataPtr->controller_manager_ = std::make_shared<controller_manager::ControllerManager>(
      std::move(resource_manager_), this->dataPtr->executor_, controllerManagerNodeName,
      this->dataPtr->node_->get_namespace(), cm_options);
  this->dataPtr->executor_->add_node(this->dataPtr->controller_manager_);

  if (!this->dataPtr->controller_manager_->has_parameter("update_rate")) {
    RCLCPP_ERROR_STREAM(this->dataPtr->node_->get_logger(),
                        "controller manager doesn't have an update_rate parameter");
    return;
  }

  this->dataPtr->update_rate = this->dataPtr->controller_manager_->get_update_rate();
  this->dataPtr->control_period_ =
      rclcpp::Duration(std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::duration<double>(1.0 / static_cast<double>(this->dataPtr->update_rate))));

  // Wait for CM to receive robot description from the topic and then initialize Resource Manager
  while (!this->dataPtr->controller_manager_->is_resource_manager_initialized()) {
    RCLCPP_WARN(this->dataPtr->node_->get_logger(),
                "Waiting RM to load and initialize hardware...");
    std::this_thread::sleep_for(std::chrono::microseconds(2000000));
  }

  this->dataPtr->entity_ = _entity;
}

//////////////////////////////////////////////////
void GZROS2ControlPlugin::PreUpdate(const gz::sim::UpdateInfo& _info,
                                    gz::sim::EntityComponentManager& /*_ecm*/) {
  if (!this->dataPtr->controller_manager_) {
    return;
  }
  static bool warned{false};
  if (!warned) {
    rclcpp::Duration gazebo_period(_info.dt);

    // Check the period against the simulation period
    if (this->dataPtr->control_period_ < _info.dt) {
      RCLCPP_ERROR_STREAM(this->dataPtr->node_->get_logger(),
                          "Desired controller update period ("
                              << this->dataPtr->control_period_.seconds()
                              << " s) is faster than the gazebo simulation period ("
                              << gazebo_period.seconds() << " s).");
    } else if (this->dataPtr->control_period_ > gazebo_period) {
      RCLCPP_WARN_STREAM(this->dataPtr->node_->get_logger(),
                         " Desired controller update period ("
                             << this->dataPtr->control_period_.seconds()
                             << " s) is slower than the gazebo simulation period ("
                             << gazebo_period.seconds() << " s).");
    }
    warned = true;
  }

  rclcpp::Time sim_time_ros(
      std::chrono::duration_cast<std::chrono::nanoseconds>(_info.simTime).count(), RCL_ROS_TIME);
  rclcpp::Duration sim_period = sim_time_ros - this->dataPtr->last_update_sim_time_ros_;
  // Always set commands on joints, otherwise at low control frequencies the
  // joints tremble as they are updated at a fraction of gazebo sim time
  this->dataPtr->controller_manager_->write(sim_time_ros, sim_period);
}

//////////////////////////////////////////////////
void GZROS2ControlPlugin::PostUpdate(const gz::sim::UpdateInfo& _info,
                                     const gz::sim::EntityComponentManager& /*_ecm*/) {
  if (!this->dataPtr->controller_manager_) {
    return;
  }
  // Get the simulation time and period
  rclcpp::Time sim_time_ros(
      std::chrono::duration_cast<std::chrono::nanoseconds>(_info.simTime).count(), RCL_ROS_TIME);
  rclcpp::Duration sim_period = sim_time_ros - this->dataPtr->last_update_sim_time_ros_;

  if (sim_period >= this->dataPtr->control_period_) {
    this->dataPtr->last_update_sim_time_ros_ = sim_time_ros;
    auto gz_controller_manager = std::dynamic_pointer_cast<gz_ros2_control::GZSystemInterface>(
        this->dataPtr->controller_manager_);
    this->dataPtr->controller_manager_->read(sim_time_ros, sim_period);
    this->dataPtr->controller_manager_->update(sim_time_ros, sim_period);
  }
}
}  // namespace gz_ros2_control

GZ_ADD_PLUGIN(gz_ros2_control::GZROS2ControlPlugin,
              gz::sim::System,
              gz_ros2_control::GZROS2ControlPlugin::ISystemConfigure,
              gz_ros2_control::GZROS2ControlPlugin::ISystemPreUpdate,
              gz_ros2_control::GZROS2ControlPlugin::ISystemPostUpdate)
