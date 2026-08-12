import time

from lerobot.robots.so_follower import SOFollower
from lerobot.robots.so_follower.config_so_follower import (
    SOFollowerRobotConfig,
)

ROBOT_PORT = "/dev/so101_follower"
ROBOT_ID = "follower"

JOINTS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

def main():
    config = SOFollowerRobotConfig(
        port=ROBOT_PORT,
        id=ROBOT_ID,
    )

    robot = SOFollower(config)

    try:
        print("Connecting...")
        robot.connect(calibrate=False)

        # Read the current positions.
        before = robot.get_observation()

        # Initially command every joint to remain at its current position.
        action = {joint: before[joint] for joint in JOINTS}

        # Move only the base by 3 degrees.
        action["shoulder_pan.pos"] += 3.0

        print(
            f"Moving shoulder pan from "
            f"{before['shoulder_pan.pos']:.2f} deg to "
            f"{action['shoulder_pan.pos']:.2f} deg..."
        )

        robot.send_action(action)
        time.sleep(2.0)

        after = robot.get_observation()

        print(
            f"Actual shoulder pan position after move: "
            f"{after['shoulder_pan.pos']:.2f} deg"
        )

    finally:
        if robot.is_connected:
            robot.disconnect()
            print("\nDisconnected from Follower.")

if __name__ == "__main__":
    main()