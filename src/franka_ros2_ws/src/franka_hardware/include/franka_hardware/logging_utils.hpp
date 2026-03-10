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

#pragma once

#include <array>
#include <cstdarg>
#include <cstdio>
#include <string>

#include <rclcpp/rclcpp.hpp>

namespace franka_hardware {

/**
 * @brief Formats a message using variadic arguments and vsnprintf.
 *
 * This function is used internally by logRclcppFatalRed to properly format
 * variadic arguments before passing to RCLCPP logging macros.
 *
 * @param text Format string (printf-style)
 * @param args va_list of arguments
 * @return Formatted string
 */
inline std::string formatVariadicMessage(const char* text, va_list args) {
  std::array<char, 1024> buffer;
  vsnprintf(buffer.data(), buffer.size(), text, args);
  return {buffer.data()};
}

/**
 * @brief Logs a fatal message in red color to ROS 2 logging system.
 *
 * @param logger The rclcpp::Logger to use
 * @param text Format string (printf-style)
 * @param ... Variadic arguments for the format string
 */
inline void logRclcppFatalRed(const rclcpp::Logger& logger, const char* text, ...) {
  va_list args;
  va_start(args, text);
  std::string formatted = formatVariadicMessage(text, args);
  va_end(args);
  RCLCPP_FATAL(logger, "\033[1;31m%s\033[0m", formatted.c_str());
}

}  // namespace franka_hardware
