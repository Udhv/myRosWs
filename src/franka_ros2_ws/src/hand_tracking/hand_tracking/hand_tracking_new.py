import os
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import cv2
import time
import math
import mediapipe as mp
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer
from tf2_ros import TransformListener
from geometry_msgs.msg import Pose
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    PositionConstraint,
    OrientationConstraint,
)
+++++++
from shape_msgs.msg import SolidPrimitive
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from franka_msgs.action import Grasp
from franka_msgs.action import Move
from copy import deepcopy


from dataclasses import dataclass


@dataclass
class Point:
    """Simple mutable x/y container used for the filtered control
    point, so we never overwrite the real MediaPipe landmark objects."""
    x: float = 0.0
    y: float = 0.0


# ---------- IMPROVEMENT 6 ----------
# Landmark indices used for the palm-center average (wrist + the four
# MCP knuckle joints), pulled out as a named constant instead of being
# repeated inline. Tuple, since this set of indices never changes.
PALM_POINTS = (0, 5, 9, 13, 17)


class FR3HandMove(Node):

    def __init__(self):

        super().__init__("fr3_hand_move")

        self.get_logger().info("FR3 Hand Teleoperation Started")

        self.last_gripper_time = 0.0
        self.gripper_delay = 0.5
        # =====================================================
        # Robot State Machine
        # =====================================================

        self.control_enabled = False
        self.executing = False

        self.gripper_closed = False

        # =====================================================
        # Reference Positions
        # =====================================================

        self.hand_origin = None
        self.robot_origin = None

        self.latest_pose = None
        self.current_robot_pose = None

        self.last_target = None

        # ==========================================
        # TF
        # ==========================================

        self.tf_buffer = Buffer()

        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

        # =====================================================
        # Motion Parameters
        # =====================================================

        self.scale_xy = 0.50
        self.scale_z = 0.35

        # ---------- CHANGED: dead zone increased (was 0.015) ----------
        self.dead_zone = 0.025

        self.lowpass_alpha = 0.60

        self.filtered_x = None
        self.filtered_y = None
        self.filtered_z = None

        # =====================================================
        # Gesture Thresholds
        # =====================================================

        # ---------- CHANGED: Adaptive Fist Detection ----------
        # (replaces the old single self.fist_threshold = 0.08)
        self.fist_enable_threshold = 0.42
        self.fist_disable_threshold = 0.50

        self.gesture_buffer = []
        self.buffer_size = 5

        self.fist_state = False

        # ---------- NEW: Hand Position Filter ----------
        self.filtered_wrist_x = None
        self.filtered_wrist_y = None

        self.hand_alpha = 0.55

        # ---------- Hand Velocity Tracking ----------
        self.prev_hand_x = None
        self.prev_hand_y = None
        self.prev_hand_time = time.time()

        # ---------- IMPROVEMENT ----------
        # vx/vy were previously computed only for a debug print and
        # otherwise unused. max_hand_velocity lets us actually act on
        # them: a spike (units/sec in normalized image coords) usually
        # means a bad/glitched detection rather than real hand motion,
        # so that frame's motion command is skipped.
        self.max_hand_velocity = 3.0
        self.velocity_spike = False

        self.pinch_close = 0.040
        self.pinch_open = 0.060

        # =====================================================
        # Workspace Limits
        # =====================================================

        self.x_min = -0.25
        self.x_max = 0.25

        self.y_min = -0.25
        self.y_max = 0.25

        self.z_min = 0.25
        self.z_max = 0.55

        # =====================================================
        # Timing
        # =====================================================

        self.last_move_time = 0

        self.prev_time = time.time()

        self.frame_counter = 0
        # ---------- IMPROVEMENT ----------
        # DEBUG is now a ROS parameter instead of a hardcoded constant,
        # so it can be toggled without editing code, e.g.:
        #   ros2 run <pkg> <exe> --ros-args -p debug:=false
        self.declare_parameter("debug", True)
        self.DEBUG = self.get_parameter("debug").value

        # ---------- IMPROVEMENT 4 ----------
        self.camera_fail_count = 0

        # =====================================================
        # MoveIt Action Client
        # =====================================================

        self.move_client = ActionClient(
            self,
            MoveGroup,
            "move_action"
        )

        self.get_logger().info(
            "Waiting for MoveGroup..."
        )

        self.move_client.wait_for_server()

        self.get_logger().info(
            "MoveGroup Connected"
        )

        # =====================================================
        # Camera
        # =====================================================

        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():

            raise RuntimeError(
                "Camera could not be opened."
            )

        self.cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            780
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            720
        )

        self.cap.set(
            cv2.CAP_PROP_FPS,
            30
        )
        # ==========================================
        # Gripper Action Clients
        # ==========================================

        #self.grasp_client = ActionClient(
         #   self,
          #  Grasp,
          #  "/franka_gripper/grasp"
        #)

        #self.move_gripper_client = ActionClient(
         #   self,
          #  Move,
           # "/franka_gripper/move"
        #)

        self.get_logger().info(
            "Waiting for Gripper..."
        )

        #self.grasp_client.wait_for_server()

        #self.move_gripper_client.wait_for_server()

        self.get_logger().info(
            "Gripper Connected"
        )
        # =====================================================
        # MediaPipe
        # =====================================================

        BaseOptions = python.BaseOptions

        VisionRunningMode = vision.RunningMode

        options = vision.HandLandmarkerOptions(

            base_options=BaseOptions(
                model_asset_path="hand_landmarker.task"
            ),

            running_mode=VisionRunningMode.VIDEO,

            num_hands=1,

            # ---------- CHANGED: confidence thresholds raised (were 0.70) ----------
            min_hand_detection_confidence=0.80,

            min_hand_presence_confidence=0.80,

            min_tracking_confidence=0.80,
        )

        self.hand_landmarker = (
            vision.HandLandmarker.create_from_options(
                options
            )
        )

        # =====================================================
        # Timer
        # =====================================================

        self.timer = self.create_timer(
            0.03,
            self.process_frame
        )
        self.pose_timer = self.create_timer(
            0.05,
            self.update_robot_pose
        )

        self.get_logger().info(
            "Initialization Complete"
        )
        # =====================================================
    # Helper Function
    # =====================================================

    def distance(self, p1, p2):

        return math.sqrt(
            (p1.x - p2.x) ** 2 +
            (p1.y - p2.y) ** 2
        )

    # ---------- NEW: small averaging helper ----------
    def average(self, values):
        return sum(values) / len(values)

    # ---------- FIX (Issue 3): hand_alpha was set but never used ----------
    # This is a separate low-pass filter (its own alpha) dedicated to the
    # hand/control-point position, distinct from self.lowpass() which is
    # used for smoothing the outgoing robot pose in compute_target_pose.
    def hand_lowpass(self, value, previous):

        if previous is None:
            return value

        return (
            self.hand_alpha * value
            +
            (1.0 - self.hand_alpha) * previous
        )

    # =====================================================
    # Detect Closed Fist (CHANGED: adaptive, palm-normalized,
    # hysteresis + temporal smoothing via gesture_buffer)
    # =====================================================

    def detect_fist(self, landmarks):

        wrist = landmarks[0]

        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]

        middle_mcp = landmarks[9]

        # Normalize finger-to-wrist distances by palm size so the
        # threshold doesn't depend on how close the hand is to the camera.
        # ---------- FIX (Issue 2): guard against divide-by-zero ----------
        # If MediaPipe momentarily returns a near-zero palm size (noisy
        # frame), dividing by it would spike/crash the ratios below.
        palm_size = max(
            self.distance(wrist, middle_mcp),
            0.001
        )

        d1 = self.distance(index_tip, wrist) / palm_size
        d2 = self.distance(middle_tip, wrist) / palm_size
        d3 = self.distance(ring_tip, wrist) / palm_size
        d4 = self.distance(pinky_tip, wrist) / palm_size

        folded = 0

        if d1 < self.fist_enable_threshold:
            folded += 1

        if d2 < self.fist_enable_threshold:
            folded += 1

        if d3 < self.fist_enable_threshold:
            folded += 1

        if d4 < self.fist_enable_threshold:
            folded += 1

        current = folded >= 3

        self.gesture_buffer.append(current)

        if len(self.gesture_buffer) > self.buffer_size:
            self.gesture_buffer.pop(0)

        stable = sum(self.gesture_buffer) >= 4

        # Hysteresis: easier to enter fist state than to leave it,
        # so borderline hand poses don't flicker control on/off.
        if not self.fist_state:

            if stable:
                self.fist_state = True

        else:

            unfolded = 0

            if d1 > self.fist_disable_threshold:
                unfolded += 1

            if d2 > self.fist_disable_threshold:
                unfolded += 1

            if d3 > self.fist_disable_threshold:
                unfolded += 1

            if d4 > self.fist_disable_threshold:
                unfolded += 1

            if unfolded >= 3:
                self.fist_state = False

        if self.DEBUG:

            # ---------- IMPROVEMENT ----------
            # get_logger() (vs print()) gets ROS timestamps, respects
            # log levels, and works with ros2 bag / launch / remote.
            self.get_logger().debug(
                "Gesture Debug | "
                f"Palm: {palm_size:.3f} | "
                f"d1-d4: {d1:.2f} {d2:.2f} {d3:.2f} {d4:.2f} | "
                f"Folded Fingers: {folded}/4 | "
                f"Buffer: {self.gesture_buffer} | "
                f"Stable: {self.fist_state}"
            )

        return self.fist_state


    # =====================================================
    # Detect Pinch
    # =====================================================

    def detect_pinch(self, landmarks):

        thumb_tip = landmarks[4]
        index_tip = landmarks[8]

        pinch_distance = self.distance(
            thumb_tip,
            index_tip
        )

        # Close gesture
        if (
            not self.gripper_closed and
            pinch_distance < self.pinch_close
        ):
            return "close"

        # Open gesture
        if (
            self.gripper_closed and
            pinch_distance > self.pinch_open
        ):
            return "open"

        return None
    # =====================================================
    # Enable Robot Control
    # =====================================================

    def enable_control(self, wrist):

        self.control_enabled = True

        self.hand_origin = (
            wrist.x,
            wrist.y
        )

        
        self.filtered_x = None
        self.filtered_y = None
        self.filtered_z = None

        self.get_logger().info(
            "CONTROL ENABLED"
        )


    # =====================================================
    # Disable Robot Control
    # =====================================================

    def disable_control(self):

        self.control_enabled = False

        self.hand_origin = None

        self.filtered_x = None
        self.filtered_y = None
        self.filtered_z = None

        self.latest_pose = None

        # ---------- REAL BUG 1 FIX ----------
        # filtered_wrist_x/y and prev_hand_x/y are read+updated every
        # frame in process_frame() regardless of control_enabled, BEFORE
        # enable_control() would ever run. Resetting them there (as the
        # previous revision did) wiped the filter memory the frame right
        # after it had just been used, causing a one-frame jump in the
        # control point. Resetting here instead means: by the time the
        # user next closes their fist, the filter has already been
        # running warm on live data with no discontinuity.
        self.filtered_wrist_x = None
        self.filtered_wrist_y = None

        self.prev_hand_x = None
        self.prev_hand_y = None

        # ---------- ISSUE 1 FIX ----------
        # Without this, a brief open-hand moment that disables control
        # left old "fist" samples sitting in the buffer, which could
        # let the very next re-close re-trigger control instantly
        # using stale history instead of a fresh, stable reading.
        self.gesture_buffer.clear()
        self.fist_state = False

        self.get_logger().info(
            "CONTROL DISABLED"
        )


    # =====================================================
    # Workspace Limiter
    # =====================================================

    def clamp_workspace(self, x, y, z):

        x = max(
            self.x_min,
            min(self.x_max, x)
        )

        y = max(
            self.y_min,
            min(self.y_max, y)
        )

        z = max(
            self.z_min,
            min(self.z_max, z)
        )

        return x, y, z


    # =====================================================
    # Dead Zone Filter
    # =====================================================

    def apply_deadzone(self, dx, dy):

        if abs(dx) < self.dead_zone:
            dx = 0.0

        if abs(dy) < self.dead_zone:
            dy = 0.0

        return dx, dy


    # =====================================================
    # Low Pass Filter
    # =====================================================

    def lowpass(self, value, previous):

        if previous is None:
            return value

        return (
            self.lowpass_alpha * value
            +
            (1.0 - self.lowpass_alpha) * previous
        )


    # =====================================================
    # FPS Display
    # =====================================================

    def update_fps(self):

        self.frame_counter += 1

        elapsed = time.time() - self.prev_time

        if elapsed >= 1.0:

            self.get_logger().debug(
                f"FPS : {self.frame_counter}"
            )

            self.prev_time = time.time()

            self.frame_counter = 0
        # =====================================================
    # Compute Target Pose
    # =====================================================
    # =====================================================
    # Debug State
    # =====================================================

    def debug_state(
        self,
        wrist,
        dx,
        dy,
        robot_x,
        robot_y,
        target_pose
    ):

        if not self.DEBUG:
            return

        origin_str = "n/a"

        if self.hand_origin is not None:
            origin_str = f"({self.hand_origin[0]:.3f}, {self.hand_origin[1]:.3f})"

        current_robot_str = "n/a"

        if self.current_robot_pose is not None:
            current_robot_str = (
                f"({self.current_robot_pose.pose.position.x:.4f}, "
                f"{self.current_robot_pose.pose.position.y:.4f}, "
                f"{self.current_robot_pose.pose.position.z:.4f})"
            )

        # ---------- IMPROVEMENT ----------
        # get_logger() (vs print()) gets ROS timestamps, respects log
        # levels, and works with ros2 bag / launch / remote.
        self.get_logger().debug(
            "TELEOP DEBUG | "
            f"Wrist: ({wrist.x:.3f}, {wrist.y:.3f}) | "
            f"Hand Origin: {origin_str} | "
            f"dx,dy: ({dx:.4f}, {dy:.4f}) | "
            f"Robot Offset: ({robot_x:.4f}, {robot_y:.4f}) | "
            f"Filtered: ({self.filtered_x:.4f}, {self.filtered_y:.4f}) | "
            f"Target Pose: ({target_pose.pose.position.x:.4f}, "
            f"{target_pose.pose.position.y:.4f}, "
            f"{target_pose.pose.position.z:.4f}) | "
            f"Current Robot: {current_robot_str} | "
            f"Control: {self.control_enabled} | "
            f"Executing: {self.executing}"
        )
    def compute_target_pose(self, wrist):

        if self.hand_origin is None:
            return None

        # ------------------------------------------
        # Relative Hand Motion
        # ------------------------------------------

        dx = wrist.x - self.hand_origin[0]
        dy = wrist.y - self.hand_origin[1]

        # Ignore small motion
        dx, dy = self.apply_deadzone(dx, dy)

        # ------------------------------------------
        # Convert Camera Motion to Robot Motion
        # ------------------------------------------

        robot_x = dx * self.scale_xy
        robot_y = -dy * self.scale_xy
        # ---------- IMPROVEMENT (noted, not yet wired up) ----------
        # Z motion is currently disabled by design (see process_frame,
        # where target_pose.pose.position.z is pinned to robot_origin).
        # If/when real Z control is added, replace this constant with
        # something derived from the hand, e.g.:
        #   robot_z = self.robot_origin.pose.position.z + dz
        robot_z = 0.40

        # ------------------------------------------
        # Low-pass Filter
        # ------------------------------------------

        self.filtered_x = self.lowpass(
            robot_x,
            self.filtered_x
        )

        self.filtered_y = self.lowpass(
            robot_y,
            self.filtered_y
        )

        self.filtered_z = self.lowpass(
            robot_z,
            self.filtered_z
        )

        # ------------------------------------------
        # Workspace Limits
        # ------------------------------------------

        x, y, z = self.clamp_workspace(
            self.filtered_x,
            self.filtered_y,
            self.filtered_z
        )

        # ------------------------------------------
        # Create Target Pose
        # ------------------------------------------

        target_pose = PoseStamped()

        target_pose.header.frame_id = "fr3_link0"

        target_pose.pose.position.x = x
        target_pose.pose.position.y = y
        target_pose.pose.position.z = z

        # Keep orientation fixed
        target_pose.pose.orientation.x = 0.0
        target_pose.pose.orientation.y = 0.0
        target_pose.pose.orientation.z = 0.0
        target_pose.pose.orientation.w = 1.0
        self.debug_state(
            wrist,
            dx,
            dy,
            robot_x,
            robot_y,
            target_pose
        )
        return target_pose


    # =====================================================
    # Update Latest Target
    # =====================================================

    def update_target(self, pose):

        if pose is None:
            return

        self.latest_pose = pose


    # =====================================================
    # Try Sending Robot Goal
    # =====================================================

    def try_send_goal(self):

        if not self.control_enabled:
            return

        if self.executing:
            return

        if self.latest_pose is None:
            return

        # ---------- CHANGED: planning rate slowed (was 0.10) ----------
        if time.time() - self.last_move_time < 0.15:
            return

        self.last_move_time = time.time()

        self.get_logger().info("Planning Robot Motion...")

        self.send_goal(self.latest_pose)


    # =====================================================
    # Draw Status on Screen
    # =====================================================

    def draw_status(self, frame):

        if self.control_enabled:
            mode = "CONTROL : ON"
            color = (0,255,0)
        else:
            mode = "CONTROL : OFF"
            color = (0,0,255)

        cv2.putText(
            frame,
            mode,
            (10,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        if self.gripper_closed:
            grip = "GRIPPER : CLOSED"
        else:
            grip = "GRIPPER : OPEN"

        cv2.putText(
            frame,
            grip,
            (10,60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,0),
            2
        )

        if self.executing:
            state = "ROBOT : MOVING"
        else:
            state = "ROBOT : READY"

        cv2.putText(
            frame,
            state,
            (10,90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,255),
            2
        )
        # =====================================================
    # Main Camera Loop
    # =====================================================

    def process_frame(self):

        
        success, frame = self.cap.read()

        if not success:

            # ---------- IMPROVEMENT 4 ----------
            # Don't fail completely silently: log occasionally so a
            # dropped/disconnected camera is noticeable, without
            # spamming the log on every single frame.
            self.camera_fail_count += 1

            if self.camera_fail_count % 30 == 1:
                self.get_logger().warn(
                    f"Camera read failed ({self.camera_fail_count} times so far)"
                )

            return

        self.camera_fail_count = 0

        frame = cv2.flip(frame, 1)

        self.update_fps()

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = int(time.time() * 1000)

        mp_start = time.perf_counter()

        result = self.hand_landmarker.detect_for_video(
            mp_image,
            timestamp
        )

        mediapipe_ms = round(
            (time.perf_counter() - mp_start) * 1000,
            2
        )

        self.get_logger().debug(
            f"MediaPipe : {mediapipe_ms} ms"
        )

        # ------------------------------------------------
        # No Hand Detected
        # ------------------------------------------------

        if not result.hand_landmarks:

            # ---------- IMPROVEMENT 1 ----------
            # Clear the gesture buffer so stale "fist" samples from
            # before the hand disappeared don't influence detection
            # once it reappears.
            self.gesture_buffer.clear()
            self.fist_state = False

            # ---------- IMPROVEMENT 2 ----------
            # Don't leave the robot in control mode with no hand to
            # track: drop control immediately if it was enabled.
            if self.control_enabled:
                self.disable_control()

            self.draw_status(frame)

            cv2.imshow(
                "FR3 Hand Control",
                frame
            )

            cv2.waitKey(1)

            return

        # ------------------------------------------------
        # Hand Detected
        # ------------------------------------------------

        landmarks = result.hand_landmarks[0]

        # NOTE (Issue 1 fix): `landmarks` / `landmarks[0]` (the real
        # wrist) are left completely untouched from here on. They are
        # only ever read from, never written to, so detect_fist() and
        # detect_pinch() keep seeing MediaPipe's real, unmodified data.
        wrist = landmarks[0]

        # ------------------------------------------------
        # Use palm center instead of raw wrist point for the CONTROL
        # signal only. Built as a separate Point, not written back
        # into the landmark. Uses average() + PALM_POINTS (Issue 2 /
        # Improvement 6 fix) instead of dead code / inline magic indices.
        # ------------------------------------------------

        cx = self.average(
            [landmarks[i].x for i in PALM_POINTS]
        )

        cy = self.average(
            [landmarks[i].y for i in PALM_POINTS]
        )

        control_point = Point(cx, cy)

        # ------------------------------------------------
        # Low-pass filter the (palm-center) control point before use.
        # Removes jitter. Uses hand_lowpass()/hand_alpha (Issue 3 fix)
        # -- a separate filter from self.lowpass()/lowpass_alpha, which
        # is reserved for smoothing the outgoing robot pose.
        # ------------------------------------------------

        fx = self.hand_lowpass(
            control_point.x,
            self.filtered_wrist_x
        )

        fy = self.hand_lowpass(
            control_point.y,
            self.filtered_wrist_y
        )

        self.filtered_wrist_x = fx
        self.filtered_wrist_y = fy

        control_point.x = fx
        control_point.y = fy

        # ------------------------------------------------
        # Hand velocity (debug / future gating use)
        # ------------------------------------------------

        now = time.time()

        self.velocity_spike = False

        if self.prev_hand_x is not None:

            # ---------- IMPROVEMENT 3 ----------
            # Clamp dt so an unusually small (or zero) timestep between
            # frames can't blow up into an unrealistic velocity value.
            dt = max(now - self.prev_hand_time, 0.001)

            vx = (control_point.x - self.prev_hand_x) / dt
            vy = (control_point.y - self.prev_hand_y) / dt

            if abs(vx) > self.max_hand_velocity or abs(vy) > self.max_hand_velocity:
                self.velocity_spike = True

            if self.DEBUG:
                self.get_logger().debug(f"Velocity : {vx:.3f} {vy:.3f}")

        self.prev_hand_x = control_point.x
        self.prev_hand_y = control_point.y
        self.prev_hand_time = now

        # ------------------------------------------------
        # Draw Hand
        # ------------------------------------------------

        for lm in landmarks:

            h, w, _ = frame.shape

            cx_px = int(lm.x * w)
            cy_px = int(lm.y * h)

            cv2.circle(
                frame,
                (cx_px, cy_px),
                4,
                (0,255,0),
                -1
            )

        # ------------------------------------------------
        # Gesture Detection
        # ------------------------------------------------

        fist = self.detect_fist(
            landmarks
        )

        pinch = self.detect_pinch(
            landmarks
        )

        # ------------------------------------------------
        # Enable Control
        # ------------------------------------------------

        if fist:

            if not self.control_enabled:

                self.enable_control(
                    control_point
                )

                # Save current robot pose as reference
                if self.current_robot_pose is not None:

                    self.robot_origin = PoseStamped()

                    self.robot_origin.header.frame_id = "fr3_link0"

                    

                    self.robot_origin.pose = deepcopy(
                        self.current_robot_pose.pose
                    )

        else:

            if self.control_enabled:

                self.disable_control()

        # ------------------------------------------------
        # Gripper
        # ------------------------------------------------

        if pinch == "close":

            self.gripper_closed = True
            #self.close_gripper()

        elif pinch == "open":

            self.gripper_closed = False
            #self.open_gripper()

        # ------------------------------------------------
        # Robot Motion
        # ------------------------------------------------

        if self.control_enabled and not self.velocity_spike:

            target_pose = self.compute_target_pose(
                control_point
            )

            if (
                target_pose is not None
                and
                self.robot_origin is not None
            ):

                # Relative motion
                target_pose.pose.position.x += \
                    self.robot_origin.pose.position.x

                target_pose.pose.position.y += \
                    self.robot_origin.pose.position.y

                target_pose.pose.position.z = \
                    self.robot_origin.pose.position.z

                target_pose.pose.orientation = \
                    self.robot_origin.pose.orientation

                self.update_target(
                    target_pose
                )

                self.try_send_goal()

        # ------------------------------------------------
        # Display
        # ------------------------------------------------

        self.draw_status(frame)

        cv2.imshow(
            "FR3 Hand Control",
            frame
        )

        cv2.waitKey(1)
       # =====================================================
        # Close Gripper
        # =====================================================

    def close_gripper(self):

        goal = Grasp.Goal()
        
        goal.width = 0.0

        goal.speed = 0.05

        goal.force = 30.0

        goal.epsilon.inner = 0.005

        goal.epsilon.outer = 0.005

        if time.time() - self.last_gripper_time < self.gripper_delay:
            return

        self.last_gripper_time = time.time()

        self.get_logger().info(
            "Closing Gripper..."
        )

        self.grasp_client.send_goal_async(
            goal
        )

        # TODO:
        # Replace with Franka Gripper Action
        #
        # Example:
        #
        # grasp_goal.width = 0.0
        # grasp_goal.speed = 0.05
        # grasp_goal.force = 30.0
        #
        # self.gripper_client.send_goal_async(grasp_goal)

        self.get_logger().info("GRIPPER CLOSE")


    # =====================================================
    # Open Gripper
    # =====================================================

    def open_gripper(self):

        goal = Move.Goal()

        goal.width = 0.08

        goal.speed = 0.05

        if time.time() - self.last_gripper_time < self.gripper_delay:
            return

        self.last_gripper_time = time.time()

        self.get_logger().info(
            "Opening Gripper..."
        )

        self.move_gripper_client.send_goal_async(
            goal
        )

        # TODO:
        # Replace with Franka Gripper Action
        #
        # Example:
        #
        # move_goal.width = 0.08
        # move_goal.speed = 0.05
        #
        # self.gripper_client.send_goal_async(move_goal)

        self.get_logger().info("GRIPPER OPEN")


    # =====================================================
    # Send MoveIt Goal
    # =====================================================

    def send_goal(self, pose):

        if pose is None:
            return

        goal_msg = MoveGroup.Goal()

        goal_msg.request.group_name = "fr3_arm"

        goal_msg.request.num_planning_attempts = 1

        goal_msg.request.allowed_planning_time = 0.25

        goal_msg.request.max_velocity_scaling_factor = 0.50

        goal_msg.request.max_acceleration_scaling_factor = 0.50

        # --------------------------------------------------
        # Position Constraint
        # --------------------------------------------------

        position_constraint = PositionConstraint()

        position_constraint.header.frame_id = "fr3_link0"

        position_constraint.link_name = "fr3_link8"

        sphere = SolidPrimitive()

        sphere.type = SolidPrimitive.SPHERE

        sphere.dimensions = [0.05]

        position_constraint.constraint_region.primitives.append(
            sphere
        )

        position_constraint.constraint_region.primitive_poses.append(
            pose.pose
        )

        position_constraint.weight = 1.0

        # --------------------------------------------------
        # Orientation Constraint
        # --------------------------------------------------

        orientation_constraint = OrientationConstraint()

        orientation_constraint.header.frame_id = "fr3_link0"

        orientation_constraint.link_name = "fr3_link8"

        orientation_constraint.orientation = pose.pose.orientation

        orientation_constraint.absolute_x_axis_tolerance = 1.0

        orientation_constraint.absolute_y_axis_tolerance = 1.0

        orientation_constraint.absolute_z_axis_tolerance = 1.0

        orientation_constraint.weight = 1.0

        # --------------------------------------------------

        constraints = Constraints()

        constraints.position_constraints.append(
            position_constraint
        )

        constraints.orientation_constraints.append(
            orientation_constraint
        )

        goal_msg.request.goal_constraints.append(
            constraints
        )

        self.executing = True

        self.get_logger().info(
            "Planning Robot Motion..."
        )

        future = self.move_client.send_goal_async(
            goal_msg
        )

        future.add_done_callback(
            self.goal_response_callback
        )
        

    def update_robot_pose(self):

        try:

            transform = self.tf_buffer.lookup_transform(

                "fr3_link0",

                "fr3_link8",

                rclpy.time.Time()

            )

            pose = PoseStamped()

            pose.header.frame_id = "fr3_link0"

            pose.pose.position.x = \
                transform.transform.translation.x

            pose.pose.position.y = \
                transform.transform.translation.y

            pose.pose.position.z = \
                transform.transform.translation.z

            pose.pose.orientation = \
                transform.transform.rotation

            self.current_robot_pose = pose

        except Exception:

            return
    def result_callback(self, future):

        self.executing = False

        # ---------- RUNTIME ISSUE 3 FIX ----------
        # future.result() (and .result on the wrapped result) can raise
        # if the goal was aborted/cancelled or the server dropped the
        # connection. Previously this would propagate as an unhandled
        # exception inside a callback and the failure went unnoticed.
        try:
            result = future.result().result
        except Exception as e:
            self.get_logger().error(
                f"MoveGroup result failed: {e}"
            )
            return

        self.get_logger().info("Motion Complete")

        if self.latest_pose is not None:
            self.try_send_goal()
# =====================================================
# Main
# =====================================================
    def goal_response_callback(self, future):

            # ---------- RUNTIME ISSUE 2 FIX ----------
            # future.result() can raise if the action server disconnects
            # or the goal request itself failed to send.
            try:
                goal_handle = future.result()
            except Exception as e:
                self.executing = False
                self.get_logger().error(
                    f"MoveGroup goal request failed: {e}"
                )
                return

            if not goal_handle.accepted:
                self.executing = False
                return

            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(self.result_callback)
def main(args=None):

    rclpy.init(args=args)

    node = None

    try:

        node = FR3HandMove()

        node.get_logger().info(
            "===================================="
        )

        node.get_logger().info(
            " FR3 Hand Teleoperation Started "
        )

        node.get_logger().info(
            "===================================="
        )

        node.get_logger().info(
            "Open Hand  : Robot Locked"
        )

        node.get_logger().info(
            "Closed Fist : Robot Enabled"
        )

        node.get_logger().info(
            "Pinch : Gripper"
        )

        rclpy.spin(node)

    except KeyboardInterrupt:

        print()

        print("Keyboard Interrupt")

    except Exception as e:

        print(e)

    finally:

        if node is not None:

            if node.cap.isOpened():

                node.cap.release()

            cv2.destroyAllWindows()

            node.destroy_node()

        rclpy.shutdown()


# =====================================================

if __name__ == "__main__":

    main()
