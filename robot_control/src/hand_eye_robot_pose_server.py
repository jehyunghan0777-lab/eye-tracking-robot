#!/usr/bin/env python3
"""Serve SO-101 gripper poses to the Windows hand-eye calibration app.

Run this script inside WSL.  The protocol is newline-delimited JSON over TCP.
The server starts with motor torque disabled.  A client can request:

* ``hold``: latch the current joint positions and enable torque.
* ``pose``: read the current joints and return the gripper FK pose.
* ``release``: disable torque so the arm can be repositioned by hand.
* ``status``: report whether the arm is currently being held.

Always support the arm before releasing torque or stopping this process.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import time
from typing import Any

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.motors.feetech import OperatingMode
from lerobot.robots.so_follower import SOFollower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig


DEFAULT_ROBOT_PORT = "/dev/so101_follower"
DEFAULT_ROBOT_ID = "follower"
DEFAULT_SERVER_HOST = "0.0.0.0"
DEFAULT_SERVER_PORT = 5055
ENCODER_COUNTS_PER_REVOLUTION = 4096
ENCODER_CENTER_TICKS = ENCODER_COUNTS_PER_REVOLUTION // 2
DEFAULT_URDF_PATH = (
    Path.home()
    / "SO-ARM100"
    / "Simulation"
    / "SO101"
    / "so101_new_calib.urdf"
)


def encoder_ticks_to_degrees(position_ticks: float) -> float:
    """Convert a raw STS3215 position to the angle expected by the URDF.

    The Feetech servo reports one revolution as 0..4095, with the URDF joint
    zero at encoder tick 2048.  LeRobot's normalized observations are not
    angles and must never be passed directly to ``forward_kinematics``.
    """
    wrapped_ticks = float(position_ticks) % ENCODER_COUNTS_PER_REVOLUTION
    return (
        (wrapped_ticks - ENCODER_CENTER_TICKS)
        * 360.0
        / ENCODER_COUNTS_PER_REVOLUTION
    )


class RobotPoseServer:
    def __init__(
        self,
        robot_port: str,
        robot_id: str,
        urdf_path: Path,
    ) -> None:
        if not urdf_path.is_file():
            raise FileNotFoundError(f"SO-101 URDF not found: {urdf_path}")

        config = SOFollowerRobotConfig(
            port=robot_port,
            id=robot_id,
            disable_torque_on_disconnect=True,
        )

        self.robot = SOFollower(config)
        self.bus = self.robot.bus
        self.joint_names = list(self.bus.motors.keys())
        self.kinematics = RobotKinematics(
            urdf_path=str(urdf_path),
            target_frame_name="moving_jaw_so101_v1_link",
            joint_names=self.joint_names,
        )
        self.holding = False

    def connect_torque_off(self) -> None:
        """Open the bus and configure position mode while torque stays off."""
        self.bus.connect(handshake=False)
        self.bus.disable_torque(num_retry=3)

        config = self.robot.config

        for motor_name in self.joint_names:
            self.bus.write(
                "Operating_Mode",
                motor_name,
                OperatingMode.POSITION.value,
            )
            self.bus.write(
                "P_Coefficient",
                motor_name,
                getattr(config, "position_p_coefficient", 16),
            )
            self.bus.write(
                "I_Coefficient",
                motor_name,
                getattr(config, "position_i_coefficient", 0),
            )
            self.bus.write(
                "D_Coefficient",
                motor_name,
                getattr(config, "position_d_coefficient", 32),
            )

        self.holding = False

    def read_raw_joint_positions(self, sample_count: int = 5) -> dict[str, int]:
        """Read median raw encoder positions without LeRobot normalization."""
        readings: dict[str, list[int]] = {
            name: [] for name in self.joint_names
        }

        for _ in range(sample_count):
            positions = self.bus.sync_read(
                "Present_Position",
                normalize=False,
                num_retry=3,
            )
            for name in self.joint_names:
                readings[name].append(int(positions[name]))
            time.sleep(0.02)

        return {
            name: int(round(float(np.median(values))))
            for name, values in readings.items()
        }

    def read_joint_positions_degrees(
        self,
        sample_count: int = 5,
    ) -> dict[str, float]:
        raw_positions = self.read_raw_joint_positions(sample_count)
        return {
            name: encoder_ticks_to_degrees(raw_positions[name])
            for name in self.joint_names
        }

    def hold_current_pose(self) -> dict[str, Any]:
        """Set the current position as the goal before enabling torque."""
        self.bus.disable_torque(num_retry=3)
        current_ticks = self.read_raw_joint_positions()
        self.bus.sync_write(
            "Goal_Position",
            current_ticks,
            normalize=False,
        )
        self.bus.enable_torque(num_retry=3)
        self.holding = True
        time.sleep(0.25)
        return self.pose_payload()

    def release(self) -> None:
        self.bus.disable_torque(num_retry=3)
        self.holding = False

    def pose_payload(self) -> dict[str, Any]:
        positions_degrees = self.read_joint_positions_degrees()
        joint_vector = np.asarray(
            [positions_degrees[name] for name in self.joint_names],
            dtype=np.float64,
        )
        pose = self.kinematics.forward_kinematics(joint_vector)

        return {
            "ok": True,
            "holding": self.holding,
            "timestamp": time.time(),
            "joint_positions_degrees": positions_degrees,
            "gripper_to_base": pose.tolist(),
        }

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        command = str(request.get("command", "")).strip().lower()

        if command == "hold":
            return self.hold_current_pose()
        if command == "pose":
            return self.pose_payload()
        if command == "release":
            self.release()
            return {"ok": True, "holding": False}
        if command == "status":
            return {"ok": True, "holding": self.holding}

        return {"ok": False, "error": f"Unknown command: {command!r}"}

    def close(self) -> None:
        if not self.bus.is_connected:
            return

        try:
            self.bus.disable_torque(num_retry=3)
        finally:
            self.holding = False
            self.bus.disconnect(disable_torque=False)


def receive_json_line(connection: socket.socket) -> dict[str, Any]:
    chunks: list[bytes] = []

    while True:
        chunk = connection.recv(4096)
        if not chunk:
            break
        chunks.append(chunk)
        if b"\n" in chunk:
            break

    raw = b"".join(chunks).split(b"\n", 1)[0]
    if not raw:
        raise ValueError("Empty request")
    return json.loads(raw.decode("utf-8"))


def send_json_line(connection: socket.socket, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload) + "\n").encode("utf-8")
    connection.sendall(encoded)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-port", default=DEFAULT_ROBOT_PORT)
    parser.add_argument("--robot-id", default=DEFAULT_ROBOT_ID)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF_PATH)
    parser.add_argument("--host", default=DEFAULT_SERVER_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_SERVER_PORT)
    arguments = parser.parse_args()

    server = RobotPoseServer(
        robot_port=arguments.robot_port,
        robot_id=arguments.robot_id,
        urdf_path=arguments.urdf,
    )

    listener: socket.socket | None = None

    try:
        server.connect_torque_off()

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((arguments.host, arguments.port))
        listener.listen(4)

        print("SO-101 hand-eye pose server ready")
        print(f"Listening on {arguments.host}:{arguments.port}")
        print("Torque is OFF. Position the arm while supporting it.")
        print("The Windows app will use H=hold and R=release.")

        while True:
            connection, address = listener.accept()
            with connection:
                connection.settimeout(5.0)
                try:
                    request = receive_json_line(connection)
                    response = server.handle_request(request)
                except Exception as error:
                    response = {"ok": False, "error": str(error)}

                send_json_line(connection, response)

                command = request.get("command", "?") if "request" in locals() else "?"
                print(f"{address[0]}: {command} -> {response.get('ok')}")

    except KeyboardInterrupt:
        print("\nShutdown requested.")

    finally:
        if listener is not None:
            listener.close()

        if server.holding:
            try:
                input(
                    "SUPPORT THE ARM, then press ENTER to disable torque: "
                )
            except EOFError:
                print("No terminal input available; disabling torque now.")

        server.close()
        print("Torque OFF. Motor bus closed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
