from __future__ import annotations

import argparse
import json
import socket
import time

import serial


INTENT_TO_GRIPPER_COMMAND = {
    "CLOSE": "close",
    "EXTEND": "open",
}


def parse_intent(
    received_line: bytes,
) -> str | None:
    try:
        decoded_line = (
            received_line
            .decode("utf-8")
            .strip()
        )

    except UnicodeDecodeError:
        return None

    fields = decoded_line.split(",")

    if (
        len(fields) != 2
        or fields[0] != "SEMG"
    ):
        return None

    intent = fields[1].strip().upper()

    if intent not in {
        "CLOSE",
        "EXTEND",
        "REST",
    }:
        return None

    return intent


def send_gripper_command(
    command: str,
    host: str,
    port: int,
    timeout: float,
) -> None:
    payload = {
        "command": command,
    }

    encoded_message = (
        json.dumps(payload) + "\n"
    ).encode("utf-8")

    with socket.create_connection(
        (host, port),
        timeout=timeout,
    ) as connection:
        connection.sendall(
            encoded_message
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Forward STM32 sEMG intent "
            "classifications to ROS."
        )
    )

    parser.add_argument(
        "--serial-port",
        required=True,
        help="STM32 USB CDC port, such as COM6.",
    )

    parser.add_argument(
        "--baud",
        type=int,
        default=115200,
    )

    parser.add_argument(
        "--host",
        default="localhost",
        help="Address of the ROS TCP bridge.",
    )

    parser.add_argument(
        "--tcp-port",
        type=int,
        default=5056,
    )

    parser.add_argument(
        "--tcp-timeout",
        type=float,
        default=2.0,
    )

    arguments = parser.parse_args()

    pending_command = None
    next_connection_attempt = 0.0
    last_intent = None

    print("sEMG to ROS gripper bridge")
    print("--------------------------")
    print(
        f"Serial: "
        f"{arguments.serial_port}"
    )
    print(
        f"ROS TCP bridge: "
        f"{arguments.host}:"
        f"{arguments.tcp_port}"
    )
    print()
    print("CLOSE  -> close gripper")
    print("EXTEND -> open gripper")
    print("REST   -> hold current state")
    print()

    try:
        with serial.Serial(
            port=arguments.serial_port,
            baudrate=arguments.baud,
            timeout=0.1,
        ) as serial_connection:
            serial_connection.reset_input_buffer()

            print("Connected. Waiting for sEMG intent...")

            while True:
                received_line = (
                    serial_connection.readline()
                )

                if received_line:
                    intent = parse_intent(
                        received_line
                    )

                    if (
                        intent is not None
                        and intent != last_intent
                    ):
                        print(
                            f"STM32 intent: {intent}"
                        )

                        last_intent = intent

                        if intent == "REST":
                            pending_command = None

                            print(
                                "Gripper command: HOLD"
                            )
                        else:
                            pending_command = (
                                INTENT_TO_GRIPPER_COMMAND[
                                    intent
                                ]
                            )

                            next_connection_attempt = (
                                0.0
                            )

                current_time = (
                    time.perf_counter()
                )

                if (
                    pending_command is not None
                    and current_time
                        >= next_connection_attempt
                ):
                    try:
                        send_gripper_command(
                            command=pending_command,
                            host=arguments.host,
                            port=arguments.tcp_port,
                            timeout=(
                                arguments.tcp_timeout
                            ),
                        )

                        print(
                            "Sent ROS gripper command: "
                            f"{pending_command}"
                        )

                        pending_command = None

                    except OSError as error:
                        print(
                            "ROS TCP bridge unavailable: "
                            f"{error}"
                        )

                        next_connection_attempt = (
                            current_time + 1.0
                        )

    except serial.SerialException as error:
        print(f"Serial error: {error}")
        return 1

    except KeyboardInterrupt:
        print("\nBridge stopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())