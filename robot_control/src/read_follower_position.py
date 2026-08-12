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
        print("Connecting to Follower..")
        robot.connect(calibrate=False)
        print("Connected successfully!\n")

        observation = robot.get_observation()

        print("Current joint positions:")
        for joint in JOINTS:
            print(f"{joint:20s}: {observation[joint]:8.2f}")

    finally:
        if robot.is_connected:
            robot.disconnect()
            print("\nDisconnected from Follower.")

if __name__ == "__main__":
    main()
