import json
import math
import socket
import time

import rclpy
from control_msgs.action import ParallelGripperCommand
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionClient
from rclpy.node import Node


class TcpGripperBridge(Node):
    """Coordinate gaze approach, sEMG grasp, and lift motions."""

    OPEN_POSITION = 1.74533
    CLOSED_POSITION = -0.174533

    IDLE = "IDLE"
    APPROACHING = "APPROACHING"
    PREGRASP_READY = "PREGRASP_READY"
    ARMED = "ARMED"
    DESCENDING = "DESCENDING"
    CLOSING = "CLOSING"
    LIFTING = "LIFTING"
    HOLDING = "HOLDING"
    RELEASING = "RELEASING"
    ERROR = "ERROR"

    def __init__(self):
        super().__init__("tcp_gripper_bridge")

        self.declare_parameter("host", "0.0.0.0")
        self.declare_parameter("port", 5056)
        self.declare_parameter(
            "action_name",
            "/follower/gripper_controller/gripper_cmd",
        )
        self.declare_parameter(
            "requested_target_topic",
            "/gaze/requested_target_pose",
        )
        self.declare_parameter(
            "motion_goal_topic",
            "/gaze/motion_goal",
        )
        self.declare_parameter(
            "motion_completed_topic",
            "/gaze/motion_completed",
        )
        self.declare_parameter("descent_distance", 0.06)
        self.declare_parameter("minimum_grasp_z", -0.04)
        self.declare_parameter("completion_tolerance", 0.035)
        self.declare_parameter("motion_timeout", 25.0)

        host = str(self.get_parameter("host").value)
        port = int(self.get_parameter("port").value)
        action_name = str(
            self.get_parameter("action_name").value
        )
        requested_target_topic = str(
            self.get_parameter("requested_target_topic").value
        )
        motion_goal_topic = str(
            self.get_parameter("motion_goal_topic").value
        )
        motion_completed_topic = str(
            self.get_parameter("motion_completed_topic").value
        )
        self.descent_distance = float(
            self.get_parameter("descent_distance").value
        )
        self.minimum_grasp_z = float(
            self.get_parameter("minimum_grasp_z").value
        )
        self.completion_tolerance = float(
            self.get_parameter("completion_tolerance").value
        )
        self.motion_timeout = float(
            self.get_parameter("motion_timeout").value
        )

        if not 0.01 <= self.descent_distance <= 0.20:
            raise ValueError(
                "descent_distance must be between 0.01 and 0.20 m"
            )
        if not 0.005 <= self.completion_tolerance <= 0.10:
            raise ValueError(
                "completion_tolerance must be between 0.005 and 0.10 m"
            )
        if self.motion_timeout <= 0.0:
            raise ValueError("motion_timeout must be positive")

        self.action_client = ActionClient(
            self,
            ParallelGripperCommand,
            action_name,
        )
        self.motion_goal_publisher = self.create_publisher(
            PoseStamped,
            motion_goal_topic,
            10,
        )
        self.create_subscription(
            PoseStamped,
            requested_target_topic,
            self.requested_target_callback,
            10,
        )
        self.create_subscription(
            PoseStamped,
            motion_completed_topic,
            self.motion_completed_callback,
            10,
        )

        self.server_socket = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM,
        )
        self.server_socket.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1,
        )
        self.server_socket.bind((host, port))
        self.server_socket.listen(1)
        self.server_socket.setblocking(False)

        self.client_socket = None
        self.receive_buffer = ""
        self.command_in_progress = False
        self.pending_gripper_command = None
        self.gripper_is_open = False

        self.state = self.IDLE
        self.approach_pose = None
        self.expected_motion_goal = None
        self.motion_deadline = None

        self.create_timer(0.02, self.poll_socket)
        self.create_timer(0.10, self.check_motion_timeout)

        self.get_logger().info(
            f"Listening for sEMG commands on TCP {host}:{port}"
        )
        self.get_logger().info(
            "Grasp sequence: gaze approach -> EXTEND open -> "
            "CLOSE descend/close/lift"
        )
        self.log_state()

    def log_state(self):
        self.get_logger().info(f"Grasp state: {self.state}")

    def set_state(self, state):
        if state != self.state:
            self.state = state
            self.log_state()

    def copy_pose(self, source):
        copied = PoseStamped()
        copied.header.stamp = self.get_clock().now().to_msg()
        copied.header.frame_id = source.header.frame_id
        copied.pose.position.x = source.pose.position.x
        copied.pose.position.y = source.pose.position.y
        copied.pose.position.z = source.pose.position.z
        copied.pose.orientation.x = source.pose.orientation.x
        copied.pose.orientation.y = source.pose.orientation.y
        copied.pose.orientation.z = source.pose.orientation.z
        copied.pose.orientation.w = source.pose.orientation.w
        return copied

    def requested_target_callback(self, target):
        if target.header.frame_id != "base_link":
            self.get_logger().error(
                "Rejected gaze target outside base_link"
            )
            return

        if self.state in {
            self.APPROACHING,
            self.DESCENDING,
            self.CLOSING,
            self.LIFTING,
            self.HOLDING,
            self.RELEASING,
        }:
            self.get_logger().warning(
                f"Rejected new target while state is {self.state}"
            )
            return

        self.approach_pose = self.copy_pose(target)
        self.set_state(self.APPROACHING)
        self.publish_motion_goal(self.approach_pose, "approach")

    def publish_motion_goal(self, pose, description):
        goal = self.copy_pose(pose)
        self.expected_motion_goal = self.copy_pose(goal)
        self.motion_deadline = time.monotonic() + self.motion_timeout
        self.motion_goal_publisher.publish(goal)

        self.get_logger().info(
            f"Published {description} goal: "
            f"x={goal.pose.position.x:.3f}, "
            f"y={goal.pose.position.y:.3f}, "
            f"z={goal.pose.position.z:.3f}"
        )

    def motion_completed_callback(self, completed):
        if self.expected_motion_goal is None:
            self.get_logger().warning(
                "Ignored motion completion without an expected goal"
            )
            return

        error = self.position_error(
            completed,
            self.expected_motion_goal,
        )

        if error > self.completion_tolerance:
            self.get_logger().warning(
                "Ignored mismatched completion: "
                f"position error {error * 1000.0:.1f} mm"
            )
            return

        self.motion_deadline = None
        self.expected_motion_goal = None

        if self.state == self.APPROACHING:
            self.set_state(
                self.ARMED
                if self.gripper_is_open
                else self.PREGRASP_READY
            )

            if self.state == self.PREGRASP_READY:
                self.get_logger().info(
                    "Pre-grasp reached. Perform EXTEND to open."
                )
            else:
                self.get_logger().info(
                    "Pre-grasp reached with gripper open. "
                    "Return to REST, then perform CLOSE."
                )

        elif self.state == self.DESCENDING:
            self.set_state(self.CLOSING)

            if not self.send_gripper_action("close"):
                self.abort_sequence(
                    "Could not command gripper close after descent"
                )

        elif self.state == self.LIFTING:
            self.set_state(self.HOLDING)
            self.get_logger().info(
                "Lift completed. EXTEND releases the object."
            )

    def position_error(self, first, second):
        dx = first.pose.position.x - second.pose.position.x
        dy = first.pose.position.y - second.pose.position.y
        dz = first.pose.position.z - second.pose.position.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def check_motion_timeout(self):
        if (
            self.motion_deadline is not None
            and time.monotonic() > self.motion_deadline
        ):
            self.abort_sequence(
                f"Timed out waiting for motion in state {self.state}"
            )

    def abort_sequence(self, reason):
        self.get_logger().error(reason)
        self.motion_deadline = None
        self.expected_motion_goal = None
        self.set_state(self.ERROR)
        self.get_logger().error(
            "Sequence stopped. Send a new gaze target to reset it."
        )

    def poll_socket(self):
        if self.client_socket is None:
            try:
                client_socket, address = self.server_socket.accept()
            except BlockingIOError:
                return

            client_socket.setblocking(False)
            self.client_socket = client_socket

            self.get_logger().info(
                f"sEMG client connected from {address}"
            )

        try:
            received_data = self.client_socket.recv(4096)
        except BlockingIOError:
            return
        except (ConnectionError, OSError):
            self.disconnect_client()
            return

        if not received_data:
            self.disconnect_client()
            return

        try:
            self.receive_buffer += received_data.decode("utf-8")
        except UnicodeDecodeError:
            self.get_logger().error(
                "Received invalid UTF-8 data"
            )
            self.disconnect_client()
            return

        while "\n" in self.receive_buffer:
            line, self.receive_buffer = self.receive_buffer.split(
                "\n",
                1,
            )

            if line.strip():
                self.process_command(line)

    def process_command(self, json_line):
        try:
            payload = json.loads(json_line)
            command = str(payload["command"]).strip().lower()

            if command not in {"open", "close"}:
                raise ValueError(
                    "command must be 'open' or 'close'"
                )

        except (
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            self.get_logger().error(
                f"Rejected sEMG command: {error}"
            )
            return

        if self.command_in_progress:
            self.get_logger().warning(
                "Rejected command because the gripper is busy"
            )
            return

        if command == "open":
            self.process_open_command()
        else:
            self.process_close_command()

    def process_open_command(self):
        if self.state in {
            self.DESCENDING,
            self.CLOSING,
            self.LIFTING,
            self.RELEASING,
        }:
            self.get_logger().warning(
                f"Ignored OPEN while state is {self.state}"
            )
            return

        if self.gripper_is_open and self.state != self.HOLDING:
            self.get_logger().info("Gripper is already open")
            return

        if self.state == self.HOLDING:
            self.set_state(self.RELEASING)

        if not self.send_gripper_action("open"):
            self.abort_sequence("Could not command gripper open")

    def process_close_command(self):
        if self.state == self.ARMED:
            if self.approach_pose is None:
                self.abort_sequence("Stored approach pose is missing")
                return

            grasp_pose = self.copy_pose(self.approach_pose)
            grasp_pose.pose.position.z -= self.descent_distance

            if grasp_pose.pose.position.z < self.minimum_grasp_z:
                self.abort_sequence(
                    "Calculated grasp goal is below the minimum safe z"
                )
                return

            self.set_state(self.DESCENDING)
            self.publish_motion_goal(grasp_pose, "grasp descent")
            return

        if self.state == self.PREGRASP_READY:
            self.get_logger().warning(
                "CLOSE rejected: perform EXTEND to open first"
            )
            return

        if self.state == self.HOLDING:
            self.get_logger().info("Object is already being held")
            return

        if self.state == self.IDLE:
            if not self.send_gripper_action("close"):
                self.abort_sequence(
                    "Could not command manual gripper close"
                )
            return

        self.get_logger().warning(
            f"CLOSE ignored while state is {self.state}"
        )

    def send_gripper_action(self, command):
        if not self.action_client.server_is_ready():
            self.get_logger().error(
                "Gripper action server is not available"
            )
            return False

        position = (
            self.OPEN_POSITION
            if command == "open"
            else self.CLOSED_POSITION
        )

        goal = ParallelGripperCommand.Goal()
        goal.command.name = ["gripper"]
        goal.command.position = [position]

        self.command_in_progress = True
        self.pending_gripper_command = command

        self.get_logger().info(
            f"Sending gripper command: {command}"
        )

        goal_future = self.action_client.send_goal_async(goal)
        goal_future.add_done_callback(
            self.goal_response_callback
        )
        return True

    def goal_response_callback(self, future):
        try:
            goal_handle = future.result()
        except Exception as error:
            self.finish_failed_gripper_command(
                f"Failed to send gripper goal: {error}"
            )
            return

        if not goal_handle.accepted:
            self.finish_failed_gripper_command(
                "Gripper goal was rejected"
            )
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            self.result_callback
        )

    def finish_failed_gripper_command(self, reason):
        self.command_in_progress = False
        self.pending_gripper_command = None
        self.abort_sequence(reason)

    def result_callback(self, future):
        command = self.pending_gripper_command
        self.command_in_progress = False
        self.pending_gripper_command = None

        try:
            result = future.result().result
        except Exception as error:
            self.abort_sequence(
                f"Failed to receive gripper result: {error}"
            )
            return

        command_succeeded = result.reached_goal or result.stalled

        if not command_succeeded:
            self.abort_sequence(
                f"Gripper {command} command did not complete"
            )
            return

        if result.stalled:
            self.get_logger().info(
                "Gripper contacted the object before full closure"
            )
        else:
            self.get_logger().info(
                f"Gripper completed {command} command"
            )

        if command == "open":
            self.gripper_is_open = True

            if self.state == self.PREGRASP_READY:
                self.set_state(self.ARMED)
                self.get_logger().info(
                    "Grasp armed. Return to REST, then perform CLOSE."
                )
            elif self.state == self.RELEASING:
                self.approach_pose = None
                self.set_state(self.IDLE)

        elif command == "close":
            self.gripper_is_open = False

            if self.state == self.CLOSING:
                if self.approach_pose is None:
                    self.abort_sequence(
                        "Stored lift pose is missing"
                    )
                    return

                self.set_state(self.LIFTING)
                self.publish_motion_goal(
                    self.approach_pose,
                    "lift",
                )

    def disconnect_client(self):
        if self.client_socket is not None:
            self.client_socket.close()
            self.client_socket = None
            self.receive_buffer = ""

            self.get_logger().info(
                "sEMG client disconnected"
            )

    def destroy_node(self):
        self.disconnect_client()
        self.server_socket.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = TcpGripperBridge()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
