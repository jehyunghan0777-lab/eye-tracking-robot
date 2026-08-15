#include <atomic>
#include <cmath>
#include <memory>
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
    move_group.setMaxVelocityScalingFactor(0.15);
    move_group.setMaxAccelerationScalingFactor(0.15);

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

                if (
                    target->header.frame_id.empty()
                    || !std::isfinite(position.x)
                    || !std::isfinite(position.y)
                    || !std::isfinite(position.z)
                ) {
                    RCLCPP_ERROR(
                        logger,
                        "Rejected invalid target pose."
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

                move_group.clearPoseTargets();
                planning.store(false);
            }
        );

    RCLCPP_INFO(
        logger,
        "Waiting for target poses on /gaze/target_pose"
    );

    spinner.join();
    rclcpp::shutdown();
    return 0;
}