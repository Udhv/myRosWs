import os
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import cv2
import rclpy
import time

from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint
from shape_msgs.msg import SolidPrimitive

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class FR3HandMove(Node):

    def __init__(self):
        super().__init__("fr3_hand_move")

        self.get_logger().info("Node started")

        # Execution lock
        self.executing = False

        # MoveIt Action Client
        self.move_client = ActionClient(
            self,
            MoveGroup,
            "move_action"
        )

        # Camera
        self.cap = cv2.VideoCapture(2)

        # MediaPipe setup
        BaseOptions = python.BaseOptions
        VisionRunningMode = vision.RunningMode

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path="hand_landmarker.task"
            ),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=1,
        )

        self.hand_landmarker = vision.HandLandmarker.create_from_options(options)

        self.last_move_time = 0

        self.timer = self.create_timer(0.5, self.process_frame)

    # ---------------------------------------------------

    def process_frame(self):

        if self.executing:
            return

        success, frame = self.cap.read()
        if not success:
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = int(time.time() * 1000)
        result = self.hand_landmarker.detect_for_video(mp_image, timestamp)

        if result.hand_landmarks:

            # Slow down planning rate
            if time.time() - self.last_move_time < 3.0:
                return

            self.last_move_time = time.time()

            wrist = result.hand_landmarks[0][0]

            target_pose = PoseStamped()
            target_pose.header.frame_id = "fr3_link0"

            target_pose.pose.position.x = (wrist.x - 0.5) * 0.6
            target_pose.pose.position.y = (wrist.y - 0.5) * -0.6
            target_pose.pose.position.z = 0.4

            target_pose.pose.orientation.w = 1.0

            self.send_goal(target_pose)

            print("Planning robot motion...")

        cv2.imshow("FR3 Hand Control", frame)
        cv2.waitKey(1)

    # ---------------------------------------------------

    def send_goal(self, pose):

        self.get_logger().info("Waiting for MoveGroup action server...")

        if not self.move_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("MoveGroup action server not available")
            return

        self.get_logger().info("MoveGroup action server connected")

        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = "fr3_arm"
        goal_msg.request.num_planning_attempts = 1
        goal_msg.request.allowed_planning_time = 3.0

        # Position constraint
        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = "fr3_link0"
        pos_constraint.link_name = "fr3_link8"

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.SPHERE
        primitive.dimensions = [0.01]

        pos_constraint.constraint_region.primitives.append(primitive)
        pos_constraint.constraint_region.primitive_poses.append(pose.pose)
        pos_constraint.weight = 1.0

        # Orientation constraint
        orient_constraint = OrientationConstraint()
        orient_constraint.header.frame_id = "fr3_link0"
        orient_constraint.link_name = "fr3_link8"
        orient_constraint.orientation = pose.pose.orientation
        orient_constraint.absolute_x_axis_tolerance = 0.2
        orient_constraint.absolute_y_axis_tolerance = 0.2
        orient_constraint.absolute_z_axis_tolerance = 0.2
        orient_constraint.weight = 1.0

        constraints = Constraints()
        constraints.position_constraints.append(pos_constraint)
        constraints.orientation_constraints.append(orient_constraint)

        goal_msg.request.goal_constraints.append(constraints)

        self.executing = True

        future = self.move_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)

    # ---------------------------------------------------

    def goal_response_callback(self, future):

        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected")
            self.executing = False
            return

        self.get_logger().info("Goal accepted")

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    # ---------------------------------------------------

    def result_callback(self, future):

        result = future.result().result
        self.get_logger().info("Execution finished")

        self.executing = False


# -------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = FR3HandMove()
    rclpy.spin(node)

    node.cap.release()
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()