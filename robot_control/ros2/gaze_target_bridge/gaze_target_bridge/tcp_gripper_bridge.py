import json
import socket

import rclpy
from control_msgs.action import ParallelGripperCommand
from rclpy.action import ActionClient
from rclpy.node import Node

class TcpGripperBridge(Node):
    OPEN_POSITION = 1.74533
    CLOSED_POSITION = -0.174533

    def __init__(self):
        super().__init__("tcp_gripper_bridge")

        self.declare_parameter("host", "0.0.0.0")
        self.declare_parameter("port", 5056)
        self.declare_parameter(
            "action_name",
            "/follower/gripper_controller/gripper_cmd",
        )

        host = self.get_parameter("host").value
        port = self.get_parameter("port").value
        action_name = self.get_parameter("action_name").value

        self.action_client = ActionClient(
            self,
            ParallelGripperCommand,
            action_name,
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

        self.create_timer(0.02, self.poll_socket)

        self.get_logger().info(
            f"Listening for gripper commands on TCP {host}:{port}"
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
                f"Client connected from {address}"
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

            if command == "open":
                position = self.OPEN_POSITION
            elif command == "close":
                position = self.CLOSED_POSITION
            else:
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
                f"Rejected gripper command: {error}"
            )
            return
        
        if self.command_in_progress:
            self.get_logger().warning(
                "Rejected command because the gripper is busy"
            )
            return
        
        if not self.action_client.server_is_ready():
            self.get_logger().error(
                "Gripper action server is not available"
            )
            return

        goal = ParallelGripperCommand.Goal()
        goal.command.name = ["gripper"]
        goal.command.position = [position]

        self.command_in_progress = True 

        self.get_logger().info(
            f"Sending gripper command: {command}"
        )

        goal_future = self.action_client.send_goal_async(goal)
        goal_future.add_done_callback(
            self.goal_response_callback
        )

    def goal_response_callback(self, future):
        try:
            goal_handle = future.result()
        except Exception as error:
            self.command_in_progress = False
            self.get_logger().error(
                f"Failed to send gripper goal: {error}"
            )
            return
        
        if not goal_handle.accepted:
            self.command_in_progress = False
            self.get_logger().error(
                "Gripper goal was rejected"
            )
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            self.result_callback
        )

    def result_callback(self, future):
        self.command_in_progress = False

        try:
            result = future.result().result
        except Exception as error:
            self.get_logger().error(
                f"Failed to receive gripper result: {error}"
            )
            return

        if result.reached_goal:
            self.get_logger().info(
                "Gripper reached the commanded position"
            )
        elif result.stalled:
            self.get_logger().warning(
                "Gripper stalled before reaching its goal"
            )
        else:
            self.get_logger().error(
                "Gripper did not reach its goal"
            )

    def disconnect_client(self):
        if self.client_socket is not None:
            self.client_socket.close()
            self.client_socket = None
            self.receive_buffer = ""

            self.get_logger().info(
                "Client disconnected"
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