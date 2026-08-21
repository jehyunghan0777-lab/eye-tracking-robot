#include <atomic>
#include <cmath>
#include <memory>
#include <string>
#include <thread>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit/move_group_interface/move_group_interface.hpp>
#include <rclcpp/rclcpp.hpp>

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<rclcpp::Node>(
        "gaze_motion_planner",
        rclcpp::NodeOptions()
            .automatically_declare_parameters_from_overrides(true)
    );

    const auto logger = node->get_logger();

    const bool execute_motion =
        node->has_parameter("execute_motion")
            ? node->get_parameter("execute_motion").as_bool()
            : node->declare_parameter<bool>(
                "execute_motion",
                false
            );

    const auto get_double_parameter =
        [&node](const std::string& name, double default_value)
        {
            return node->has_parameter(name)
                ? node->get_parameter(name).as_double()
                : node->declare_parameter<double>(name, default_value);
        };

    const double workspace_min_x =
        get_double_parameter("workspace_min_x", 0.08);
    const double workspace_max_x =
        get_double_parameter("workspace_max_x", 0.45);
    const double workspace_min_y =
        get_double_parameter("workspace_min_y", -0.28);
    const double workspace_max_y =
        get_double_parameter("workspace_max_y", 0.28);
    const double workspace_min_z =
        get_double_parameter("workspace_min_z", -0.05);
    const double workspace_max_z =
        get_double_parameter("workspace_max_z", 0.35);

    if (
        workspace_min_x >= workspace_max_x
        || workspace_min_y >= workspace_max_y
        || workspace_min_z >= workspace_max_z
    ) {
        RCLCPP_FATAL(logger, "Invalid workspace bounds.");
        rclcpp::shutdown();
        return 1;
    }

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);

    std::thread spinner(
        [&executor]()
        {
            executor.spin();
        }
    );

    moveit::planning_interface::MoveGroupInterface move_group(
        node,
        "manipulator"
    );

    move_group.setPlanningTime(5.0);
    move_group.setMaxVelocityScalingFactor(0.10);
    move_group.setMaxAccelerationScalingFactor(0.10);
    move_group.setGoalPositionTolerance(0.020);
    move_group.setGoalOrientationTolerance(0.35);
    move_group.setWorkspace(
        workspace_min_x,
        workspace_min_y,
        workspace_min_z,
        workspace_max_x,
        workspace_max_y,
        workspace_max_z
    );

    std::atomic_bool planning{false};

    auto target_subscription =
        node->create_subscription<
        geometry_msgs::msg::PoseStamped
        >(
            "/gaze/target_pose",
            10,
            [&](const geometry_msgs::msg::PoseStamped::SharedPtr target)
            {
                if (planning.exchange(true)) {
                    RCLCPP_WARN(
                        logger,
                        "Ignoring target: planning is already active"
                    );
                    return;
                }

                const auto& position = target->pose.position;
                const auto& orientation = target->pose.orientation;

                const double quaternion_norm = std::sqrt(
                    orientation.x * orientation.x
                    + orientation.y * orientation.y
                    + orientation.z * orientation.z
                    + orientation.w * orientation.w
                );

                if (
                    target->header.frame_id != "base_link"
                    || !std::isfinite(position.x)
                    || !std::isfinite(position.y)
                    || !std::isfinite(position.z)
                    || !std::isfinite(orientation.x)
                    || !std::isfinite(orientation.y)
                    || !std::isfinite(orientation.z)
                    || !std::isfinite(orientation.w)
                    || quaternion_norm < 1e-9
                ) {
                    RCLCPP_ERROR(
                        logger,
                        "Rejected invalid target pose."
                    );
                    planning.store(false);
                    return;
                }

                if (
                    position.x < workspace_min_x
                    || position.x > workspace_max_x
                    || position.y < workspace_min_y
                    || position.y > workspace_max_y
                    || position.z < workspace_min_z
                    || position.z > workspace_max_z
                ) {
                    RCLCPP_ERROR(
                        logger,
                        "Rejected target outside guarded workspace."
                    );
                    planning.store(false);
                    return;
                }

                RCLCPP_INFO(
                    logger,
                    "Planning for target: x=%.3f y=%.3f z=%.3f",
                    position.x,
                    position.y,
                    position.z
                );

                try {
                    move_group.setStartStateToCurrentState();
                    move_group.setPoseTarget(*target);

                    moveit::planning_interface::MoveGroupInterface::Plan plan;

                    const bool plan_succeeded =
                        move_group.plan(plan)
                        == moveit::core::MoveItErrorCode::SUCCESS;

                    if(!plan_succeeded) {
                        RCLCPP_ERROR(logger, "Motion planning failed.");
                    } else if (!execute_motion) {
                        RCLCPP_INFO(
                            logger,
                            "Plan succeeded. Execution is disabled."
                        );
                    } else {
                        const bool execution_succeeded =
                            move_group.execute(plan)
                            == moveit::core::MoveItErrorCode::SUCCESS;

                        RCLCPP_INFO(
                            logger,
                            execution_succeeded
                                ? "Motion execution succeeded."
                                : "Motion execution failed."
                        );
                    }
                } catch (const std::exception& error) {
                    RCLCPP_ERROR(
                        logger,
                        "Motion pipeline exception: %s",
                        error.what()
                    );
                }

                move_group.clearPoseTargets();
                planning.store(false);
            }
        );

    RCLCPP_INFO(
        logger,
        execute_motion
            ? "PHYSICAL EXECUTION ENABLED; waiting for targets."
            : "Planning-only mode; execution is disabled."
    );

    spinner.join();
    rclcpp::shutdown();
    return 0;
}
