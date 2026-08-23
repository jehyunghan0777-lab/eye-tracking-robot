import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    namespace = LaunchConfiguration("namespace")
    variant = LaunchConfiguration("variant")
    hardware_type = LaunchConfiguration("hardware_type")
    follower_usb_port = LaunchConfiguration("follower_usb_port")
    use_rviz = LaunchConfiguration("use_rviz")
    use_sim_time = LaunchConfiguration("use_sim_time")
    execute_motion = LaunchConfiguration("execute_motion")
    pose_bridge_port = LaunchConfiguration("pose_bridge_port")
    gripper_bridge_port = LaunchConfiguration("gripper_bridge_port")
    grasp_descent_distance = LaunchConfiguration(
        "grasp_descent_distance"
    )
    wrist_roll_lock_degrees = LaunchConfiguration(
        "wrist_roll_lock_degrees"
    )

    xacro_path = os.path.join(
        get_package_share_directory("so101_description"),
        "urdf",
        "so101_arm.urdf.xacro",
    )

    moveit_config = (
        MoveItConfigsBuilder(
            "so101_arm",
            package_name="so101_moveit_config",
        )
        .robot_description(
            file_path=xacro_path,
            mappings={
                "variant": variant,
                "use_ros2_control": "false",
            },
        )
        .robot_description_semantic()
        .robot_description_kinematics()
        .planning_pipelines(
            pipelines=[
                "ompl",
                "pilz_industrial_motion_planner",
            ]
        )
        .pilz_cartesian_limits(
            file_path="config/pilz_cartesian_limits.yaml"
        )
        .joint_limits()
        .trajectory_execution(
            file_path="config/moveit_controllers.yaml",
            moveit_manage_controllers=False,
        )
        .to_moveit_configs()
    )

    so101_demo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("so101_bringup"),
                "launch",
                "follower_moveit_demo.launch.py",
            )
        ),
        launch_arguments={
            "hardware_type": hardware_type,
            "follower_usb_port": follower_usb_port,
            "namespace": namespace,
            "use_cameras": "false",
            "use_rviz": use_rviz,
        }.items(),
    )

    pose_bridge_node = Node(
        package="gaze_target_bridge",
        executable="tcp_pose_bridge",
        output="screen",
        parameters=[
            {
                "host": "0.0.0.0",
                "port": ParameterValue(
                    pose_bridge_port,
                    value_type=int,
                ),
                "output_topic": "/gaze/requested_target_pose",
            }
        ],
    )

    gripper_bridge_node = Node(
        package="gaze_target_bridge",
        executable="tcp_gripper_bridge",
        output="screen",
        parameters=[
            {
                "host": "0.0.0.0",
                "port": ParameterValue(
                    gripper_bridge_port,
                    value_type=int,
                ),
                "action_name": (
                    "/follower/gripper_controller/gripper_cmd"
                ),
                "requested_target_topic": (
                    "/gaze/requested_target_pose"
                ),
                "motion_goal_topic": "/gaze/motion_goal",
                "motion_completed_topic": (
                    "/gaze/motion_completed"
                ),
                "descent_distance": ParameterValue(
                    grasp_descent_distance,
                    value_type=float,
                ),
            }
        ],
    )

    planner_node = Node(
        package="gaze_motion_planner",
        executable="gaze_motion_planner",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {
                "execute_motion": ParameterValue(
                    execute_motion,
                    value_type=bool,
                ),
                "use_sim_time": use_sim_time,
                "wrist_roll_lock_degrees": ParameterValue(
                    wrist_roll_lock_degrees,
                    value_type=float,
                ),
            },
        ],
        remappings=[
            (
                "joint_states",
                ["/", namespace, "/joint_states"],
            )
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "namespace",
                default_value="follower",
            ),
            DeclareLaunchArgument(
                "variant",
                default_value="follower",
            ),
            DeclareLaunchArgument(
                "hardware_type",
                default_value="mock",
                description="Use 'real' only for the physical follower.",
            ),
            DeclareLaunchArgument(
                "follower_usb_port",
                default_value="/dev/so101_follower",
            ),
            DeclareLaunchArgument(
                "use_rviz",
                default_value="true",
            ),
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="false",
            ),
            DeclareLaunchArgument(
                "execute_motion",
                default_value="false",
                description="Execute planned motion when true",
            ),
            DeclareLaunchArgument(
                "pose_bridge_port",
                default_value="5055",
            ),
            DeclareLaunchArgument(
                "gripper_bridge_port",
                default_value="5056",
            ),
            DeclareLaunchArgument(
                "grasp_descent_distance",
                default_value="0.06",
                description=(
                    "Vertical descent from pre-grasp to grasp in meters"
                ),
            ),
            DeclareLaunchArgument(
                "wrist_roll_lock_degrees",
                default_value="32.57",
                description=(
                    "Calibrated wrist-roll value for a straight gripper"
                ),
            ),
            so101_demo,
            pose_bridge_node,
            gripper_bridge_node,
            planner_node,
        ]
    )
