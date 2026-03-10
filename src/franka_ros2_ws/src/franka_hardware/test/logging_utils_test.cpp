// Copyright (c) 2025 Franka Robotics GmbH
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

#include <gmock/gmock.h>
#include <cstdarg>

#include <rclcpp/rclcpp.hpp>

#include <franka_hardware/logging_utils.hpp>

// Helper to call formatVariadicMessage with variadic args for testing
namespace {
std::string testFormatMessage(const char* text, ...) {
  va_list args;
  va_start(args, text);
  std::string result = franka_hardware::formatVariadicMessage(text, args);
  va_end(args);
  return result;
}
}  // namespace

class LoggingUtilsTest : public ::testing::Test {
 protected:
  void SetUp() override { rclcpp::init(0, nullptr); }

  void TearDown() override { rclcpp::shutdown(); }
};

TEST_F(LoggingUtilsTest, givenVariadicIntArgs_whenFormatted_expectCorrectOutput) {
  // This test verifies the fix for the variadic formatting bug in logRclcppFatalRed
  // The original bug passed va_list directly to RCLCPP_FATAL which doesn't accept va_list
  int expected_major = 1;
  int actual_major = 0;

  std::string result = testFormatMessage(
      "Unsupported major version of the Franka ros2_control interface. Expected "
      "major version %d, got %d. Please update your URDF (aka franka_description).",
      expected_major, actual_major);

  ASSERT_EQ(result,
            "Unsupported major version of the Franka ros2_control interface. Expected "
            "major version 1, got 0. Please update your URDF (aka franka_description).");
}

TEST_F(LoggingUtilsTest, givenVariadicStringArg_whenFormatted_expectCorrectOutput) {
  std::string param_name = "robot_ip";

  std::string result = testFormatMessage("Parameter '%s' is not set", param_name.c_str());

  ASSERT_EQ(result, "Parameter 'robot_ip' is not set");
}

TEST_F(LoggingUtilsTest, givenMixedVariadicArgs_whenFormatted_expectCorrectOutput) {
  const char* param_name = "version";
  int major = 2;
  int minor = 1;
  int patch = 0;

  std::string result =
      testFormatMessage("Parameter '%s' version: %d.%d.%d", param_name, major, minor, patch);

  ASSERT_EQ(result, "Parameter 'version' version: 2.1.0");
}

TEST_F(LoggingUtilsTest, givenNoVariadicArgs_whenFormatted_expectCorrectOutput) {
  std::string result = testFormatMessage("Could not connect to robot");

  ASSERT_EQ(result, "Could not connect to robot");
}

TEST_F(LoggingUtilsTest, givenLogRclcppFatalRed_whenCalled_expectNoThrow) {
  // Test that logRclcppFatalRed doesn't crash with various argument types
  auto logger = rclcpp::get_logger("test_logger");

  EXPECT_NO_THROW(franka_hardware::logRclcppFatalRed(logger, "Simple message"));
  EXPECT_NO_THROW(franka_hardware::logRclcppFatalRed(logger, "Int value: %d", 42));
  EXPECT_NO_THROW(franka_hardware::logRclcppFatalRed(logger, "String value: %s", "test"));
  EXPECT_NO_THROW(franka_hardware::logRclcppFatalRed(logger, "Mixed: %s %d %d", "version", 1, 2));
}







