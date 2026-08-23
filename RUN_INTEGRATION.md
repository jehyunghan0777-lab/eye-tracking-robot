# Gaze + RealSense + sEMG + SO-101 runtime

This update connects the calibrated Windows perception stack to the physical
SO-101 through ROS 2 in WSL. It deliberately starts with a 60 mm hover target
and requires the `M` key before any gaze-selected pose is sent.

## What changed

- The perception app stores a gaze-selected target and waits for `M`.
- The selected object point is raised by a guarded approach height.
- Both Windows and ROS reject targets outside the configured workspace.
- The ROS launch file now supports `hardware_type:=real` and starts both TCP
  bridges automatically.
- MoveIt physical execution remains explicitly gated by
  `execute_motion:=true` and uses 10% velocity and acceleration scaling.
- The STM32 repeats its embedded stable classification at 10 Hz. This lets the
  host vote over several predictions and reconnect without getting stuck.

## 1. Install the update

Extract `integration_runtime_update.zip` directly into:

```text
C:\Users\jehyu\eye-tracking-robot-semG
```

Allow Windows to merge the folders and replace the matching files.

## 2. Rebuild and flash the STM32 once

Open this existing STM32CubeIDE project:

```text
semg_control\firmware\semg_intent_controller
```

Run **Project > Clean**, **Build Project**, then flash it. The model itself has
not changed; this build only adds a periodic stable-intent heartbeat required
by the host-side voter.

Close every serial terminal using COM6 after flashing.

## 3. Copy and build the updated ROS packages

Run in WSL:

```bash
conda deactivate 2>/dev/null || true
source /opt/ros/jazzy/setup.bash
mkdir -p ~/ros2_ws/src
cp -r /mnt/c/Users/jehyu/eye-tracking-robot-semG/robot_control/ros2/gaze_motion_planner ~/ros2_ws/src/
cp -r /mnt/c/Users/jehyu/eye-tracking-robot-semG/robot_control/ros2/gaze_target_bridge ~/ros2_ws/src/
cd ~/ros2_ws
colcon build --symlink-install --packages-select gaze_target_bridge gaze_motion_planner
```

## 4. Start physical ROS control in WSL

Stop the hand-eye pose server, LeRobot scripts, and anything else using the
follower serial port. Clear the robot workspace. Keep a hand next to the 12 V
power disconnect.

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
ls -l /dev/so101_follower
ros2 launch gaze_motion_planner gaze_moveit_demo.launch.py \
  hardware_type:=real \
  follower_usb_port:=/dev/so101_follower \
  execute_motion:=true \
  use_rviz:=true
```

The terminal must print `PHYSICAL EXECUTION ENABLED` and both TCP bridge ports.

## 5. Start the Windows perception app

Open Anaconda Prompt in the repository:

```bat
conda activate gaze-tracking
cd C:\Users\jehyu\eye-tracking-robot-semG
python gaze_tracking\src\perception_app.py --enable-robot-send --robot-host 127.0.0.1 --approach-height 0.06
```

Look at an object until it says `READY`. Confirm that the robot workspace is
clear, then press `M` once. The arm plans and moves to the guarded hover point.
`R` clears the selection; `Q` or Escape quits. Do not use `--auto-send` during
physical commissioning.

## 6. Start sEMG gripper control in a second Windows prompt

```bat
conda activate gaze-tracking
cd C:\Users\jehyu\eye-tracking-robot-semG
python semg_control\tools\semg_ros_bridge.py --serial-port COM6 --host 127.0.0.1
```

Begin at `REST`. A confirmed `CLOSE` closes the gripper. Return to `REST` before
using `EXTEND` to open it. The one-gesture-per-REST interlock prevents a held or
repeated classification from issuing repeated commands.

## Network fallback

If Windows cannot reach the WSL TCP ports through `127.0.0.1`, obtain the WSL
address:

```bat
wsl -d Ubuntu-24.04 hostname -I
```

Use the first address as `--robot-host` and `--host` in the two Windows
commands.

## Emergency stop

If motion is unexpected, disconnect the robot's 12 V motor power immediately.
Stopping a terminal is not a guaranteed hardware emergency stop.
