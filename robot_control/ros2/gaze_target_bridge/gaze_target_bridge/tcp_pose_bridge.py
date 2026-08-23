import json
import math
import socket

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node

class TcpPoseBridge(Node):
    def __init__(self):
        super().__init__("gaze_target_bridge")

        self.declare_parameter("host", "0.0.0.0")
        self.declare_parameter("port", 5055)
        self.declare_parameter(
            "output_topic",
            "/gaze/target_pose",
        )

        host = self.get_parameter("host").value
        port = self.get_parameter("port").value
        output_topic = self.get_parameter("output_topic").value

        self.pose_publisher = self.create_publisher(
            PoseStamped,
            output_topic,
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

        self.create_timer(0.02, self.poll_socket)

        self.get_logger().info(
            f"Listening for target poses on TCP {host}:{port}"
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
                f"Windows client connected from {address}"
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
            self.get_logger().error("Received invalid UTF-8 data")
            self.disconnect_client()
            return

        while "\n" in self.receive_buffer:
            line, self.receive_buffer = self.receive_buffer.split(
                "\n",
                1,
            )

            if line.strip():
                self.publish_pose(line)

    def publish_pose(self, json_line):
        try:
            payload = json.loads(json_line)

            frame_id = payload["frame_id"]
            position = payload["position"]
            orientation = payload["orientation"]

            x = float(position["x"])
            y = float(position["y"])
            z = float(position["z"])

            qx = float(orientation["x"])
            qy = float(orientation["y"])
            qz = float(orientation["z"])
            qw = float(orientation["w"])

            values = [x, y, z, qx, qy, qz, qw]

            if not frame_id or not all(
                math.isfinite(value) for value in values
            ):
                raise ValueError("Pose contains invalid values")

            quaternion_norm = math.sqrt(
                qx * qx
                + qy * qy
                + qz * qz
                + qw * qw
            )

            if quaternion_norm < 1e-9:
                raise ValueError("Quaternion has zero magnitude")

            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = frame_id

            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = z

            pose.pose.orientation.x = qx / quaternion_norm
            pose.pose.orientation.y = qy / quaternion_norm
            pose.pose.orientation.z = qz / quaternion_norm
            pose.pose.orientation.w = qw / quaternion_norm

            self.pose_publisher.publish(pose)

            self.get_logger().info(
                f"Published target: x={x:.3f}, "
                f"y={y:.3f}, z={z:.3f}"
            )

        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self.get_logger().error(
                f"Rejected target data: {error}"
            )

    def disconnect_client(self):
        if self.client_socket is not None:
            self.client_socket.close()
            self.client_socket = None
            self.receive_buffer = ""
            self.get_logger().info("Windows client disconnected")

    def destroy_node(self):
        self.disconnect_client()
        self.server_socket.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)

    node = TcpPoseBridge()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()