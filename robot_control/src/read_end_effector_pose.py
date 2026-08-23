from pathlib import Path

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower import SOFollower
from lerobot.robots.so_follower.config_so_follower import (
    SOFollowerRobotConfig,
)

ROBOT_PORT = "/dev/so101_follower"
ROBOT_ID = "follower"
ENCODER_COUNTS_PER_REVOLUTION = 4096
ENCODER_CENTER_TICKS = ENCODER_COUNTS_PER_REVOLUTION // 2

URDF_PATH = (
    Path.home()
    / "SO-ARM100"
    / "Simulation"
    / "SO101"
    / "so101_new_calib.urdf"
)


def encoder_ticks_to_degrees(position_ticks: float) -> float:
    """Convert a raw STS3215 encoder position to a URDF joint angle."""
    wrapped_ticks = float(position_ticks) % ENCODER_COUNTS_PER_REVOLUTION
    return (
        (wrapped_ticks - ENCODER_CENTER_TICKS)
        * 360.0
        / ENCODER_COUNTS_PER_REVOLUTION
    )


def main():
    config = SOFollowerRobotConfig(
        port=ROBOT_PORT,
        id=ROBOT_ID,
    )

    robot = SOFollower(config)

    joint_names = list(robot.bus.motors.keys())

    kinematics = RobotKinematics(
        urdf_path=str(URDF_PATH),
        target_frame_name="gripper_frame_link",
        joint_names=joint_names,
    )

    robot.connect(calibrate=False)

    try:
        raw_positions = robot.bus.sync_read(
            "Present_Position",
            normalize=False,
            num_retry=3,
        )

        joint_positions = np.array(
            [
                encoder_ticks_to_degrees(raw_positions[name])
                for name in joint_names
            ],
            dtype=float,
        )

        pose = kinematics.forward_kinematics(joint_positions)
        position = pose[:3, 3]

        print("\nJoint positions:")
        for name, value in zip(joint_names, joint_positions):
            print(f"{name:15s}: {value:8.2f} deg")

        print("\nGripper-tip position relative to robot base:")
        print(f"x: {position[0]:.4f} m")
        print(f"y: {position[1]:.4f} m")
        print(f"z: {position[2]:.4f} m")

        print("\nComplete 4x4 gripper pose:")
        print(pose)

    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
