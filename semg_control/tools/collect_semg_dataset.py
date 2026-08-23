from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import random
import time

import serial


SEMG_DIRECTORY = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT_DIRECTORY = (
    SEMG_DIRECTORY
    / "data"
    / "raw"
)


def parse_sample(
    received_line: bytes,
) -> tuple[int, int, int] | None:
    """
    Convert one STM32 serial line into three integers.

    Expected STM32 format:
    sample_number,ch1,ch2
    """

    try:
        decoded_line = (
            received_line
            .decode("utf-8")
            .strip()
        )

        fields = decoded_line.split(",")

        if len(fields) != 3:
            return None

        sample_number = int(fields[0])
        channel_1 = int(fields[1])
        channel_2 = int(fields[2])

        return (
            sample_number,
            channel_1,
            channel_2,
        )

    except (
        UnicodeDecodeError,
        ValueError,
    ):
        return None


def capture_segment(
    connection: serial.Serial,
    csv_writer: csv.writer,
    trial_number: int,
    label: str,
    duration_seconds: float,
) -> int:
    """
    Record one REST, CLOSE, or EXTEND segment.
    """

    connection.reset_input_buffer()

    segment_start_time = time.perf_counter()

    segment_end_time = (
        segment_start_time
        + duration_seconds
    )

    captured_samples = 0

    print(f"\nRecording {label}...")
    print("\a", end="", flush=True)

    while time.perf_counter() < segment_end_time:
        received_line = connection.readline()

        parsed_sample = parse_sample(
            received_line
        )

        if parsed_sample is None:
            continue

        (
            sample_number,
            channel_1,
            channel_2,
        ) = parsed_sample

        segment_time = (
            time.perf_counter()
            - segment_start_time
        )

        csv_writer.writerow(
            [
                trial_number,
                label,
                time.time_ns(),
                segment_time,
                sample_number,
                channel_1,
                channel_2,
            ]
        )

        captured_samples += 1

    print(
        f"Finished {label}: "
        f"{captured_samples} samples captured."
    )

    return captured_samples


def wait_for_user(
    instruction: str,
) -> None:
    """
    Wait until the user confirms that the hand is ready.
    """

    input(
        f"\n{instruction}\n"
        "Press Enter when ready..."
    )

    print("Recording starts in 1 second...")

    time.sleep(1.0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Collect labeled two-channel "
            "sEMG training data."
        )
    )

    parser.add_argument(
        "--port",
        required=True,
        help=(
            "STM32 serial port, "
            "for example COM5."
        ),
    )

    parser.add_argument(
        "--baud",
        type=int,
        default=115200,
        help="Serial baud rate.",
    )

    parser.add_argument(
        "--repetitions",
        type=int,
        default=10,
        help=(
            "Number of CLOSE repetitions "
            "and EXTEND repetitions."
        ),
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help=(
            "Recording duration for each "
            "gesture in seconds."
        ),
    )

    arguments = parser.parse_args()

    if arguments.repetitions <= 0:
        print(
            "ERROR: Repetitions must be "
            "greater than zero."
        )
        return 1

    if arguments.duration <= 0:
        print(
            "ERROR: Duration must be "
            "greater than zero."
        )
        return 1

    DEFAULT_OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_path = (
        DEFAULT_OUTPUT_DIRECTORY
        / f"semg_training_{timestamp}.csv"
    )

    gesture_order = (
        ["CLOSE"] * arguments.repetitions
        + ["EXTEND"] * arguments.repetitions
    )

    random.shuffle(gesture_order)

    sample_counts = {
        "REST": 0,
        "CLOSE": 0,
        "EXTEND": 0,
    }

    print()
    print("Two-channel sEMG training collection")
    print("------------------------------------")
    print("REST:")
    print("  Relax your measured hand completely.")
    print()
    print("CLOSE:")
    print("  Make a firm fist without bending your wrist.")
    print()
    print("EXTEND:")
    print(
        "  Spread your fingers and slightly "
        "lift your wrist."
    )
    print()
    print(
        "Use your opposite hand to press Enter."
    )
    print(
        "Do not move or reposition the sensors."
    )
    print()
    print(f"Serial port: {arguments.port}")
    print(f"Output file: {output_path}")

    input(
        "\nPress Enter to connect to the STM32..."
    )

    try:
        with serial.Serial(
            port=arguments.port,
            baudrate=arguments.baud,
            timeout=0.1,
        ) as connection:
            print(
                f"\nConnected to {arguments.port}."
            )

            time.sleep(2.0)

            connection.reset_input_buffer()

            with output_path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as output_file:
                csv_writer = csv.writer(
                    output_file
                )

                csv_writer.writerow(
                    [
                        "trial",
                        "label",
                        "host_time_ns",
                        "segment_time_s",
                        "sample",
                        "ch1",
                        "ch2",
                    ]
                )

                total_trials = len(
                    gesture_order
                )

                for (
                    trial_number,
                    gesture,
                ) in enumerate(
                    gesture_order,
                    start=1,
                ):
                    print()
                    print(
                        "================================"
                    )
                    print(
                        f"Trial {trial_number}"
                        f"/{total_trials}"
                    )
                    print(
                        "================================"
                    )

                    wait_for_user(
                        "RELAX your measured hand."
                    )

                    sample_counts["REST"] += (
                        capture_segment(
                            connection=connection,
                            csv_writer=csv_writer,
                            trial_number=trial_number,
                            label="REST",
                            duration_seconds=(
                                arguments.duration
                            ),
                        )
                    )

                    if gesture == "CLOSE":
                        gesture_instruction = (
                            "Make a firm fist. "
                            "Keep your wrist straight."
                        )
                    else:
                        gesture_instruction = (
                            "Spread your fingers and "
                            "slightly lift your wrist."
                        )

                    wait_for_user(
                        gesture_instruction
                    )

                    sample_counts[gesture] += (
                        capture_segment(
                            connection=connection,
                            csv_writer=csv_writer,
                            trial_number=trial_number,
                            label=gesture,
                            duration_seconds=(
                                arguments.duration
                            ),
                        )
                    )

                    output_file.flush()

    except serial.SerialException as error:
        print()
        print(f"Serial error: {error}")
        print(
            "Close any other serial terminal "
            "using the STM32 port."
        )
        return 1

    except KeyboardInterrupt:
        print()
        print(
            "Collection stopped. Any samples "
            "already written were preserved."
        )
        return 1

    print()
    print("Data collection completed.")
    print("--------------------------")
    print(
        f"REST:   {sample_counts['REST']} samples"
    )
    print(
        f"CLOSE:  {sample_counts['CLOSE']} samples"
    )
    print(
        f"EXTEND: {sample_counts['EXTEND']} samples"
    )
    print()
    print(f"Saved to:\n{output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())