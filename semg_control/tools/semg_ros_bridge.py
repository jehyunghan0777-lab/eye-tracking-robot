from __future__ import annotations

import argparse
from collections import Counter, deque
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


def determine_stable_intent(
    intent_history: deque[str],
    required_votes: int,
) -> str | None:
    if len(intent_history) < required_votes:
        return None

    vote_counts = Counter(intent_history)

    intent, vote_count = (
        vote_counts.most_common(1)[0]
    )

    if vote_count < required_votes:
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
            "Filter STM32 sEMG classifications "
            "and forward stable commands to ROS."
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

    parser.add_argument(
        "--vote-window",
        type=int,
        default=5,
        help="Number of recent predictions used for voting.",
    )

    parser.add_argument(
        "--required-votes",
        type=int,
        default=4,
        help="Votes required to confirm an intent.",
    )

    parser.add_argument(
        "--cooldown",
        type=float,
        default=0.75,
        help="Minimum seconds between gripper commands.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test filtering without connecting to ROS.",
    )

    arguments = parser.parse_args()

    if arguments.vote_window <= 0:
        parser.error("--vote-window must be positive")

    if (
        arguments.required_votes <= 0
        or arguments.required_votes
        > arguments.vote_window
    ):
        parser.error(
            "--required-votes must be between 1 "
            "and --vote-window"
        )

    if arguments.cooldown < 0.0:
        parser.error("--cooldown cannot be negative")

    intent_history: deque[str] = deque(
        maxlen=arguments.vote_window
    )

    pending_command = None
    next_connection_attempt = 0.0
    last_command_time = float("-inf")

    last_raw_intent = None
    stable_intent = None

    # The bridge starts disarmed. A stable REST must
    # be detected before the first command is accepted.
    command_armed = False

    print("Filtered sEMG to ROS gripper bridge")
    print("-----------------------------------")
    print(f"Serial: {arguments.serial_port}")
    print(
        f"ROS TCP bridge: "
        f"{arguments.host}:{arguments.tcp_port}"
    )
    print(
        f"Voting: {arguments.required_votes} of "
        f"{arguments.vote_window}"
    )
    print(f"Cooldown: {arguments.cooldown:.2f} seconds")

    if arguments.dry_run:
        print("Mode: DRY RUN — ROS commands are disabled")

    print()
    print("Begin with your hand resting.")
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

                current_time = time.perf_counter()

                if received_line:
                    intent = parse_intent(
                        received_line
                    )

                    if intent is not None:
                        if intent != last_raw_intent:
                            print(
                                f"Raw STM32 intent: {intent}"
                            )
                            last_raw_intent = intent

                        intent_history.append(intent)

                        voted_intent = (
                            determine_stable_intent(
                                intent_history,
                                arguments.required_votes,
                            )
                        )

                        if (
                            voted_intent is not None
                            and voted_intent
                            != stable_intent
                        ):
                            stable_intent = voted_intent

                            print(
                                "Confirmed intent: "
                                f"{stable_intent}"
                            )

                            if stable_intent == "REST":
                                if pending_command is not None:
                                    print(
                                        "Cancelled pending command"
                                    )

                                pending_command = None
                                command_armed = True

                                print(
                                    "System armed for one gesture"
                                )

                            elif not command_armed:
                                print(
                                    "Ignored gesture: return to "
                                    "REST before another command"
                                )

                            else:
                                pending_command = (
                                    INTENT_TO_GRIPPER_COMMAND[
                                        stable_intent
                                    ]
                                )

                                command_armed = False

                                next_connection_attempt = max(
                                    current_time,
                                    last_command_time
                                    + arguments.cooldown,
                                )

                                print(
                                    "Accepted gripper command: "
                                    f"{pending_command}"
                                )

                if (
                    pending_command is not None
                    and current_time
                    >= next_connection_attempt
                ):
                    if arguments.dry_run:
                        print(
                            "DRY RUN command: "
                            f"{pending_command}"
                        )

                        pending_command = None
                        last_command_time = current_time
                        continue

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
                        last_command_time = current_time

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