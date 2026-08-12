import time

from lerobot.robots.so_follower import SOFollower
from lerobot.robots.so_follower.config_so_follower import (
    SOFollowerRobotConfig,
)

ROBOT_PORT = "/dev/so101_follower"
ROBOT_ID = "follower"

ENCODER_COUNTS = 4096

def read_raw_positions(bus):
    """Read uncalibrated encoder values from all six motors."""
    positions = bus.sync_read(
        "Present_Position",
        normalize=False,
        num_retry=3,
    )

    return {
        name: int(position)
        for name, position in positions.items()
    }

def calculate_encoder_change(before,after):
    """
    Calculate the shortest encoder change.

    This prevets crossing 0/4095 from incorrectly appearing
    as an almost-complete rotaiton.
    """

    difference = after - before

    return(
        (difference + ENCODER_COUNTS // 2)
        % ENCODER_COUNTS
        - ENCODER_COUNTS // 2
    )

def test_physical_joint(bus, physical_joint):
    input(
        f"\nPlace the physical {physical_joint} at its starting position.\n"
        "Press ENTER to record the starting encoder values: "
    )

    before = read_raw_positions(bus)

    print("\nStarting raw positions:")

    for name, position in before.items():
        motor_id = bus.motors[name].id

        print(
            f"ID {motor_id}  "
            f"{name:15s}: {position:4d}"
        )

    input(
        f"\nSlowly move only physical {physical_joint} by about 10 deg.\n"
        "Keep it in the new positon and press ENTER: "
    )

    after = read_raw_positions(bus)

    changes = {
        name: calculate_encoder_change(
            before[name],
            after[name],
        )
        for name in before
    }

    print(
        f"\nRaw encoder changes after moving "
        f"physical {physical_joint}:"
    )

    for name, change in changes.items():
        motor_id = bus.motors[name].id

        print(
            f"ID {motor_id}   "
            f"{name:15s}: {change:+6d}"
        )

    detected_name = max(
        changes,
        key=lambda name: abs(changes[name]),
    )

    detected_id = bus.motors[detected_name].id
    detected_change = changes[detected_name]

    print(
        f"\nLargest change: {detected_name} "
        f"(motor ID {detected_id}, "
        f"change {detected_change:+d})"
    )

def main():
    config = SOFollowerRobotConfig(
        port=ROBOT_PORT,
        id=ROBOT_ID,
    )

    robot = SOFollower(config)
    bus = robot.bus

    try:
        print(
            "Opening the motor bus without calling robot.connect()..."
        )

        # This opens the serial port without configuring the motors
        # or enabling their torque
        bus.connect(handshake=False)

        print("\nKeep the 12V motor power disconnected.")
        input(
            "Press ENTER, then connect the 12 V power within 10 seconds:"
        )

        print("\n Continously forcing torque OFF...")

        # Repeatedly trasmit Torque_Enable = 0 while the
        # motor power is being connected.
        for _ in range(200):
            bus.sync_write(
                "Torque_Enable",
                0,
                normalize=False,
            )

            time.sleep(0.05)

        # Disable torque individually as an additional safeguard
        bus.disable_torque(num_retry=3)

        torque_states = bus.sync_read(
            "Torque_Enable",
            normalize=False,
            num_retry=3,
        )

        print("\nTorque states:")

        for name, state in torque_states.items():
            print(f"{name:15s}: {int(state)}")

        if any(
            int(state) != 0
            for state in torque_states.values()
        ):
            raise RuntimeError(
                "Torque was not disabled on every motor. "
                "Disconnect the 12 V power immedietly."
            )

        print("\nTorque is confirmed off.")
        print("The joints should move freely by hand.")

        test_physical_joint(
            bus,
            "SHOULDER PAN",
        )

        test_physical_joint(
            bus,
            "WRIST ROLL",
        )

    finally:
        if bus.is_connected:
            try:
                bus.disable_torque(num_retry=2)
            except Exception as error:
                print(
                    "\nCould not reconfirm torque-off during cleanup:",
                    error,
                )
                print("Disconnected the 12 V power.")
            finally:
                bus.disconnect(disable_torque=False)

        print("\nMotor bus closed.")

if __name__ == "__main__":
    main()