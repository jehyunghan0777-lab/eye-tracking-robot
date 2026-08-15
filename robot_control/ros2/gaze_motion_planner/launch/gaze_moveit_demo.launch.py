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
    use_sim_time = LaunchConfiguration("use_sim_time")
    execute_motion = LaunchConfiguration("execute_motion")

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
            "hardware_type": "mock",
            "namespace": namespace,
            "use_cameras": "false",
            "use_rviz": "true",
        }.items(),
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
                "use_sim_time",
                default_value="false",
            ),
            DeclareLaunchArgument(
                "execute_motion",
                default_value="false",
                description="Execute planned motion when true",
            ),
            so101_demo,
            planner_node,
        ]
    )