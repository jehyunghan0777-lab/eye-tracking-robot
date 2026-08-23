#include <Eigen/Cholesky>
#include <Eigen/Geometry>

#include <atomic>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit/kinematics_base/kinematics_base.hpp>
#include <moveit/robot_state/robot_state.hpp>
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

    const auto get_bool_parameter =
        [&node](const std::string& name, bool default_value)
        {
            return node->has_parameter(name)
                ? node->get_parameter(name).as_bool()
                : node->declare_parameter<bool>(name, default_value);
        };

    const auto get_integer_parameter =
        [&node](
            const std::string& name,
            std::int64_t default_value
        )
        {
            return node->has_parameter(name)
                ? node->get_parameter(name).as_int()
                : node->declare_parameter<std::int64_t>(
                    name,
                    default_value
                );
        };

    const auto get_string_parameter =
        [&node](
            const std::string& name,
            const std::string& default_value
        )
        {
            return node->has_parameter(name)
                ? node->get_parameter(name).as_string()
                : node->declare_parameter<std::string>(
                    name,
                    default_value
                );
        };

    const std::string tcp_link =
        get_string_parameter(
            "tcp_link",
            "gripper_frame_link"
        );

    const double maximum_tcp_error =
        get_double_parameter(
            "maximum_tcp_error",
            0.015
        );

    const double maximum_axis_error_degrees =
        get_double_parameter(
            "maximum_axis_error_degrees",
            45.0
        );

    const double wrist_roll_lock_degrees =
        get_double_parameter(
            "wrist_roll_lock_degrees",
            32.57
        );

    const bool allow_workspace_search =
        get_bool_parameter(
            "allow_workspace_search",
            false
        );

    const bool use_pose_ik =
        get_bool_parameter(
            "use_pose_ik",
            false
        );

    const std::int64_t workspace_search_samples =
        get_integer_parameter(
            "workspace_search_samples",
            150000
        );

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
        get_double_parameter("workspace_max_z", 0.40);

    if (
        workspace_min_x >= workspace_max_x
        || workspace_min_y >= workspace_max_y
        || workspace_min_z >= workspace_max_z
        || maximum_tcp_error <= 0.0
        || maximum_axis_error_degrees <= 0.0
        || maximum_axis_error_degrees >= 90.0
        || !std::isfinite(wrist_roll_lock_degrees)
        || wrist_roll_lock_degrees < -180.0
        || wrist_roll_lock_degrees > 180.0
        || workspace_search_samples <= 0
    ) {
        RCLCPP_FATAL(logger, "Invalid motion-planner parameters.");
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

    const std::string default_end_effector_link =
        move_group.getEndEffectorLink();

    const auto robot_model = move_group.getRobotModel();

    const moveit::core::LinkModel* tcp_link_model =
        robot_model
            ? robot_model->getLinkModel(tcp_link)
            : nullptr;

    if (
        !robot_model
        || tcp_link_model == nullptr
    ) {
        RCLCPP_FATAL(
            logger,
            "Configured TCP link '%s' is not present in the robot model.",
            tcp_link.c_str()
        );
        executor.cancel();

        if (spinner.joinable()) {
            spinner.join();
        }

        rclcpp::shutdown();
        return 1;
    }

    if (!move_group.setEndEffectorLink(tcp_link)) {
        RCLCPP_FATAL(
            logger,
            "MoveIt rejected TCP link '%s'.",
            tcp_link.c_str()
        );
        executor.cancel();

        if (spinner.joinable()) {
            spinner.join();
        }

        rclcpp::shutdown();
        return 1;
    }

    RCLCPP_INFO(
        logger,
        "MoveIt default end-effector: '%s'; physical TCP: '%s'.",
        default_end_effector_link.c_str(),
        tcp_link.c_str()
    );

    move_group.setPlanningTime(10.0);
    move_group.setNumPlanningAttempts(10);
    move_group.setMaxVelocityScalingFactor(0.10);
    move_group.setMaxAccelerationScalingFactor(0.10);
    move_group.setGoalPositionTolerance(maximum_tcp_error);
    move_group.setGoalOrientationTolerance(
        maximum_axis_error_degrees
        * 3.14159265358979323846
        / 180.0
    );
    move_group.setWorkspace(
        workspace_min_x,
        workspace_min_y,
        workspace_min_z,
        workspace_max_x,
        workspace_max_y,
        workspace_max_z
    );

    move_group.startStateMonitor(30.0);

    if (!move_group.getCurrentState(5.0)) {
        RCLCPP_FATAL(
            logger,
            "No current robot joint state received."
        );
        executor.cancel();

        if (spinner.joinable()) {
            spinner.join();
        }

        rclcpp::shutdown();
        return 1;
    }

    RCLCPP_INFO(
        logger,
        "Robot joint-state monitor is ready."
    );

    auto startup_state = move_group.getCurrentState(2.0);

    if (!startup_state) {
        RCLCPP_FATAL(
            logger,
            "Could not read the startup robot state."
        );
        executor.cancel();

        if (spinner.joinable()) {
            spinner.join();
        }

        rclcpp::shutdown();
        return 1;
    }

    startup_state->update();

    const Eigen::Vector3d startup_tcp_position =
        startup_state->getGlobalLinkTransform(tcp_link).translation();

    RCLCPP_INFO(
        logger,
        "Current modeled TCP: x=%.4f y=%.4f z=%.4f m.",
        startup_tcp_position.x(),
        startup_tcp_position.y(),
        startup_tcp_position.z()
    );

    auto target_callback_group =
        node->create_callback_group(
            rclcpp::CallbackGroupType::Reentrant
        );

    rclcpp::SubscriptionOptions subscription_options;
    subscription_options.callback_group =
        target_callback_group;

    std::atomic_bool planning{false};

    auto motion_completed_publisher =
        node->create_publisher<
            geometry_msgs::msg::PoseStamped
        >(
            "/gaze/motion_completed",
            10
        );

    auto target_subscription =
        node->create_subscription<
        geometry_msgs::msg::PoseStamped
        >(
            "/gaze/motion_goal",
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
                      auto current_state =
                          move_group.getCurrentState(2.0);

                      if (!current_state) {
                          RCLCPP_ERROR(
                              logger,
                              "Could not read current robot state."
                          );
                          planning.store(false);
                          return;
                      }

                      const auto* joint_model_group =
                          current_state->getJointModelGroup(
                              "manipulator"
                          );

                      if (joint_model_group == nullptr) {
                          RCLCPP_ERROR(
                              logger,
                              "Robot IK group is unavailable."
                          );
                          planning.store(false);
                          return;
                      }

                      std::vector<double> current_joint_values;
                      current_state->copyJointGroupPositions(
                          joint_model_group,
                          current_joint_values
                      );

                      const auto& joint_variable_names =
                          joint_model_group->getVariableNames();

                      std::size_t wrist_roll_index =
                          joint_variable_names.size();

                      for (
                          std::size_t index = 0;
                          index < joint_variable_names.size();
                          ++index
                      ) {
                          if (
                              joint_variable_names[index]
                              == "wrist_roll"
                          ) {
                              wrist_roll_index = index;
                              break;
                          }
                      }

                      if (
                          wrist_roll_index
                          >= current_joint_values.size()
                      ) {
                          RCLCPP_ERROR(
                              logger,
                              "Could not locate wrist_roll in the manipulator group."
                          );
                          planning.store(false);
                          return;
                      }

                      const Eigen::Quaterniond nominal_orientation(
                          orientation.w,
                          orientation.x,
                          orientation.y,
                          orientation.z
                      );

                      constexpr double pi =
                          3.14159265358979323846;

                      const double locked_wrist_roll =
                          wrist_roll_lock_degrees * pi / 180.0;

                      const double maximum_axis_error =
                          maximum_axis_error_degrees
                          * pi / 180.0;

                      const Eigen::Vector3d requested_position(
                          position.x,
                          position.y,
                          position.z
                      );

                      const Eigen::Vector3d desired_axis =
                          nominal_orientation
                              .normalized()
                              .toRotationMatrix()
                              .col(0);

                      const std::vector<int> tilt_angles{
                          0,
                          15,
                          30,
                          45
                      };

                      const std::vector<int> tilt_azimuth_angles{
                          0,
                          45,
                          90,
                          135,
                          180,
                          225,
                          270,
                          315
                      };

                      double best_score =
                          std::numeric_limits<double>::infinity();

                      double best_position_error =
                          std::numeric_limits<double>::infinity();

                      double best_axis_error =
                          std::numeric_limits<double>::infinity();

                      double best_joint_travel =
                          std::numeric_limits<double>::infinity();

                      int best_roll_degrees = 0;
                      int best_tilt_degrees = 0;
                      int best_tilt_azimuth_degrees = 0;

                      std::vector<double> best_joint_values;

                      kinematics::KinematicsQueryOptions ik_options;
                      ik_options.return_approximate_solution = true;

                      if (use_pose_ik) {
                      for (const int tilt_degrees : tilt_angles) {
                          for (
                              const int tilt_azimuth_degrees :
                                  tilt_azimuth_angles
                          ) {
                              if (
                                  tilt_degrees == 0
                                  && tilt_azimuth_degrees != 0
                              ) {
                                  continue;
                              }

                              for (
                                  int roll_degrees = -180;
                                  roll_degrees < 180;
                                  roll_degrees += 30
                              ) {
                              auto candidate_pose =
                                  target->pose;

                              const double tilt_azimuth_radians =
                                  tilt_azimuth_degrees * pi / 180.0;

                              const Eigen::Vector3d tilt_axis =
                                  (
                                      std::cos(tilt_azimuth_radians)
                                      * Eigen::Vector3d::UnitY()
                                      + std::sin(tilt_azimuth_radians)
                                      * Eigen::Vector3d::UnitZ()
                                  ).normalized();

                              const Eigen::Quaterniond tilt_rotation(
                                  Eigen::AngleAxisd(
                                      tilt_degrees * pi / 180.0,
                                      tilt_axis
                                  )
                              );

                              const Eigen::Quaterniond roll_rotation(
                                  Eigen::AngleAxisd(
                                      roll_degrees * pi / 180.0,
                                      Eigen::Vector3d::UnitX()
                                  )
                              );

                              const Eigen::Quaterniond candidate_orientation =
                                  (
                                      nominal_orientation
                                      * tilt_rotation
                                      * roll_rotation
                                  ).normalized();

                              candidate_pose.orientation.x =
                                  candidate_orientation.x();
                              candidate_pose.orientation.y =
                                  candidate_orientation.y();
                              candidate_pose.orientation.z =
                                  candidate_orientation.z();
                              candidate_pose.orientation.w =
                                  candidate_orientation.w();

                              moveit::core::RobotState candidate_state(
                                  *current_state
                              );

                              const bool found =
                                  candidate_state.setFromIK(
                                      joint_model_group,
                                      candidate_pose,
                                      tcp_link,
                                      0.05,
                                      moveit::core::
                                          GroupStateValidityCallbackFn(),
                                      ik_options
                                  );

                              if (!found) {
                                  continue;
                              }

                              std::vector<double> candidate_joint_values;
                              candidate_state.copyJointGroupPositions(
                                  joint_model_group,
                                  candidate_joint_values
                              );

                              candidate_joint_values[wrist_roll_index] =
                                  locked_wrist_roll;

                              candidate_state.setJointGroupPositions(
                                  joint_model_group,
                                  candidate_joint_values
                              );

                              if (
                                  !candidate_state.satisfiesBounds(
                                      joint_model_group
                                  )
                              ) {
                                  continue;
                              }

                              candidate_state.update();

                              const Eigen::Isometry3d achieved_pose =
                                  candidate_state.getGlobalLinkTransform(
                                      tcp_link
                                  );

                              const double position_error =
                                  (
                                      achieved_pose.translation()
                                      - requested_position
                                  ).norm();

                              const Eigen::Vector3d achieved_axis =
                                  achieved_pose.rotation().col(0);

                              const double axis_dot = std::clamp(
                                  desired_axis.dot(achieved_axis),
                                  -1.0,
                                  1.0
                              );

                              const double axis_error =
                                  std::acos(axis_dot);

                              if (axis_error > maximum_axis_error) {
                                  continue;
                              }

                              double joint_travel = 0.0;

                              for (
                                  std::size_t index = 0;
                                  index < candidate_joint_values.size();
                                  ++index
                              ) {
                                  const double difference =
                                      candidate_joint_values[index]
                                      - current_joint_values[index];

                                  joint_travel +=
                                      difference * difference;
                              }

                              joint_travel = std::sqrt(joint_travel);

                              const double score =
                                  position_error
                                  + axis_error * 0.010
                                  + joint_travel * 0.001
                                  + std::abs(tilt_degrees) * 0.0001;

                              if (score < best_score) {
                                  best_score = score;
                                  best_position_error =
                                      position_error;
                                  best_axis_error = axis_error;
                                  best_joint_travel = joint_travel;
                                  best_roll_degrees =
                                      roll_degrees;
                                  best_tilt_degrees =
                                      tilt_degrees;
                                  best_tilt_azimuth_degrees =
                                      tilt_azimuth_degrees;
                                  best_joint_values =
                                      std::move(candidate_joint_values);
                              }
                              }
                          }
                      }
                      }

                      bool used_position_axis_ik = false;

                      if (
                          best_joint_values.empty()
                          || best_position_error > maximum_tcp_error
                      ) {
                          RCLCPP_WARN(
                              logger,
                              "Pose IK unavailable; solving the five-DOF TCP position-and-axis task."
                          );

                          constexpr int numerical_ik_restarts = 64;
                          constexpr int numerical_ik_iterations = 250;
                          constexpr double damping = 0.025;
                          constexpr double maximum_joint_step = 0.12;
                          constexpr double axis_weight = 0.060;

                          best_joint_values.clear();
                          best_score =
                              std::numeric_limits<double>::infinity();
                          best_position_error =
                              std::numeric_limits<double>::infinity();
                          best_axis_error =
                              std::numeric_limits<double>::infinity();
                          best_joint_travel =
                              std::numeric_limits<double>::infinity();

                          for (
                              int restart = 0;
                              restart < numerical_ik_restarts;
                              ++restart
                          ) {
                              moveit::core::RobotState candidate_state(
                                  *current_state
                              );

                              if (restart > 0) {
                                  candidate_state.setToRandomPositions(
                                      joint_model_group
                                  );
                              }

                              candidate_state.enforceBounds(
                                  joint_model_group
                              );

                              std::vector<double> seed_joint_values;
                              candidate_state.copyJointGroupPositions(
                                  joint_model_group,
                                  seed_joint_values
                              );
                              seed_joint_values[wrist_roll_index] =
                                  locked_wrist_roll;
                              candidate_state.setJointGroupPositions(
                                  joint_model_group,
                                  seed_joint_values
                              );

                              for (
                                  int iteration = 0;
                                  iteration < numerical_ik_iterations;
                                  ++iteration
                              ) {
                                  candidate_state.update();

                                  const Eigen::Isometry3d achieved_pose =
                                      candidate_state
                                          .getGlobalLinkTransform(
                                              tcp_link_model
                                          );

                                  const Eigen::Vector3d position_error_vector =
                                      requested_position
                                      - achieved_pose.translation();

                                  const Eigen::Vector3d achieved_axis =
                                      achieved_pose.rotation().col(0);

                                  const double axis_dot = std::clamp(
                                      desired_axis.dot(achieved_axis),
                                      -1.0,
                                      1.0
                                  );

                                  const double position_error =
                                      position_error_vector.norm();

                                  const double axis_error =
                                      std::acos(axis_dot);

                                  if (
                                      position_error
                                          <= maximum_tcp_error * 0.5
                                      && axis_error
                                          <= maximum_axis_error
                                  ) {
                                      break;
                                  }

                                  Eigen::MatrixXd geometric_jacobian;

                                  if (
                                      !candidate_state.getJacobian(
                                          joint_model_group,
                                          tcp_link_model,
                                          Eigen::Vector3d::Zero(),
                                          geometric_jacobian
                                      )
                                      || geometric_jacobian.rows() < 6
                                  ) {
                                      break;
                                  }

                                  const Eigen::Index joint_count =
                                      geometric_jacobian.cols();

                                  Eigen::MatrixXd task_jacobian(
                                      6,
                                      joint_count
                                  );

                                  task_jacobian.topRows(3) =
                                      geometric_jacobian.topRows(3);

                                  task_jacobian.bottomRows(3) =
                                      axis_weight
                                      * geometric_jacobian.bottomRows(3);

                                  Eigen::VectorXd task_error(6);
                                  task_error.head(3) =
                                      position_error_vector;
                                  task_error.tail(3) =
                                      axis_weight
                                      * achieved_axis.cross(desired_axis);

                                  Eigen::MatrixXd damped_normal_matrix =
                                      task_jacobian
                                      * task_jacobian.transpose();

                                  damped_normal_matrix.diagonal().array() +=
                                      damping * damping;

                                  Eigen::VectorXd joint_update =
                                      task_jacobian.transpose()
                                      * damped_normal_matrix
                                            .ldlt()
                                            .solve(task_error);

                                  if (!joint_update.allFinite()) {
                                      break;
                                  }

                                  joint_update[
                                      static_cast<Eigen::Index>(
                                          wrist_roll_index
                                      )
                                  ] = 0.0;

                                  const double largest_update =
                                      joint_update.cwiseAbs().maxCoeff();

                                  if (largest_update > maximum_joint_step) {
                                      joint_update *=
                                          maximum_joint_step
                                          / largest_update;
                                  }

                                  std::vector<double> joint_values;
                                  candidate_state
                                      .copyJointGroupPositions(
                                          joint_model_group,
                                          joint_values
                                      );

                                  if (
                                      joint_values.size()
                                      != static_cast<std::size_t>(
                                          joint_update.size()
                                      )
                                  ) {
                                      break;
                                  }

                                  for (
                                      std::size_t index = 0;
                                      index < joint_values.size();
                                      ++index
                                  ) {
                                      if (index == wrist_roll_index) {
                                          joint_values[index] =
                                              locked_wrist_roll;
                                      } else {
                                          joint_values[index] +=
                                              joint_update[
                                                  static_cast<Eigen::Index>(
                                                      index
                                                  )
                                              ];
                                      }
                                  }

                                  candidate_state.setJointGroupPositions(
                                      joint_model_group,
                                      joint_values
                                  );
                                  candidate_state.enforceBounds(
                                      joint_model_group
                                  );
                              }

                              candidate_state.update();

                              const Eigen::Isometry3d achieved_pose =
                                  candidate_state.getGlobalLinkTransform(
                                      tcp_link_model
                                  );

                              const double position_error =
                                  (
                                      achieved_pose.translation()
                                      - requested_position
                                  ).norm();

                              const Eigen::Vector3d achieved_axis =
                                  achieved_pose.rotation().col(0);

                              const double axis_error = std::acos(
                                  std::clamp(
                                      desired_axis.dot(achieved_axis),
                                      -1.0,
                                      1.0
                                  )
                              );

                              if (
                                  position_error > maximum_tcp_error
                                  || axis_error > maximum_axis_error
                                  || !candidate_state.satisfiesBounds(
                                      joint_model_group
                                  )
                              ) {
                                  continue;
                              }

                              std::vector<double> candidate_joint_values;
                              candidate_state.copyJointGroupPositions(
                                  joint_model_group,
                                  candidate_joint_values
                              );

                              double joint_travel = 0.0;

                              for (
                                  std::size_t index = 0;
                                  index < candidate_joint_values.size();
                                  ++index
                              ) {
                                  const double difference =
                                      candidate_joint_values[index]
                                      - current_joint_values[index];

                                  joint_travel +=
                                      difference * difference;
                              }

                              joint_travel = std::sqrt(joint_travel);

                              const double score =
                                  position_error
                                  + axis_error * 0.010
                                  + joint_travel * 0.001;

                              if (score < best_score) {
                                  best_score = score;
                                  best_position_error = position_error;
                                  best_axis_error = axis_error;
                                  best_joint_travel = joint_travel;
                                  best_joint_values =
                                      std::move(candidate_joint_values);
                                  used_position_axis_ik = true;
                              }
                          }
                      }

                      bool used_workspace_fallback = false;

                      if (
                          best_joint_values.empty()
                          || best_position_error > maximum_tcp_error
                      ) {
                          if (!allow_workspace_search) {
                              RCLCPP_ERROR(
                                  logger,
                                  "No safe TCP IK solution was found."
                              );
                              planning.store(false);
                              return;
                          }

                          RCLCPP_WARN(
                              logger,
                              "Pose IK failed; running constrained TCP search."
                          );

                          used_workspace_fallback = true;
                          best_joint_values.clear();
                          best_score =
                              std::numeric_limits<double>::infinity();
                          best_position_error =
                              std::numeric_limits<double>::infinity();
                          best_axis_error =
                              std::numeric_limits<double>::infinity();
                          best_joint_travel =
                              std::numeric_limits<double>::infinity();

                          for (
                              std::int64_t sample = 0;
                              sample < workspace_search_samples;
                              ++sample
                          ) {
                              moveit::core::RobotState candidate_state(
                                  *current_state
                              );

                              candidate_state.setToRandomPositions(
                                  joint_model_group
                              );

                              std::vector<double> random_joint_values;
                              candidate_state.copyJointGroupPositions(
                                  joint_model_group,
                                  random_joint_values
                              );
                              random_joint_values[wrist_roll_index] =
                                  locked_wrist_roll;
                              candidate_state.setJointGroupPositions(
                                  joint_model_group,
                                  random_joint_values
                              );
                              candidate_state.enforceBounds(
                                  joint_model_group
                              );
                              candidate_state.update();

                              const Eigen::Isometry3d candidate_fk =
                                  candidate_state.getGlobalLinkTransform(
                                      tcp_link
                                  );

                              const double position_error =
                                  (
                                      candidate_fk.translation()
                                      - requested_position
                                  ).norm();

                              const Eigen::Vector3d candidate_axis =
                                  candidate_fk.rotation().col(0);

                              const double axis_dot = std::clamp(
                                  desired_axis.dot(candidate_axis),
                                  -1.0,
                                  1.0
                              );

                              const double axis_error =
                                  std::acos(axis_dot);

                              if (axis_error > maximum_axis_error) {
                                  continue;
                              }

                              std::vector<double> candidate_joint_values;
                              candidate_state.copyJointGroupPositions(
                                  joint_model_group,
                                  candidate_joint_values
                              );

                              double joint_travel = 0.0;

                              for (
                                  std::size_t index = 0;
                                  index < candidate_joint_values.size();
                                  ++index
                              ) {
                                  const double difference =
                                      candidate_joint_values[index]
                                      - current_joint_values[index];

                                  joint_travel +=
                                      difference * difference;
                              }

                              joint_travel = std::sqrt(joint_travel);

                              const double score =
                                  position_error
                                  + axis_error * 0.010
                                  + joint_travel * 0.001;

                              if (score < best_score) {
                                  best_score = score;
                                  best_position_error =
                                      position_error;
                                  best_axis_error =
                                      axis_error;
                                  best_joint_travel = joint_travel;
                                  best_joint_values =
                                      std::move(candidate_joint_values);
                              }
                          }

                          if (
                              best_joint_values.empty()
                              || best_position_error
                                  > maximum_tcp_error
                              || best_axis_error
                                  > maximum_axis_error
                          ) {
                              RCLCPP_ERROR(
                                  logger,
                                  "No safe TCP solution: position error %.1f mm, axis error %.1f deg.",
                                  best_position_error * 1000.0,
                                  best_axis_error * 180.0 / pi
                              );
                              planning.store(false);
                              return;
                          }
                      }

                      RCLCPP_INFO(
                          logger,
                          "Wrist roll locked at %.1f deg.",
                          best_joint_values[wrist_roll_index]
                              * 180.0 / pi
                      );

                      if (used_workspace_fallback) {
                          RCLCPP_INFO(
                              logger,
                              "Constrained TCP target: error=%.1f mm, axis error=%.1f deg, joint travel=%.2f rad",
                              best_position_error * 1000.0,
                              best_axis_error * 180.0 / pi,
                              best_joint_travel
                          );
                      } else if (used_position_axis_ik) {
                          RCLCPP_INFO(
                              logger,
                              "Five-DOF IK selected: TCP error=%.1f mm, axis error=%.1f deg, joint travel=%.2f rad",
                              best_position_error * 1000.0,
                              best_axis_error * 180.0 / pi,
                              best_joint_travel
                          );
                      } else {
                          RCLCPP_INFO(
                              logger,
                              "IK selected: TCP error=%.1f mm, axis error=%.1f deg, roll=%d deg, tilt=%d deg at azimuth=%d deg",
                              best_position_error * 1000.0,
                              best_axis_error * 180.0 / pi,
                              best_roll_degrees,
                              best_tilt_degrees,
                              best_tilt_azimuth_degrees
                          );
                      }

                      move_group.setStartState(*current_state);

                      if (
                          !move_group.setJointValueTarget(
                              best_joint_values
                          )
                      ) {
                          RCLCPP_ERROR(
                              logger,
                              "IK joint target violates robot limits."
                          );
                          planning.store(false);
                          return;
                      }

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

                        if (!execution_succeeded) {
                            RCLCPP_ERROR(
                                logger,
                                "Motion execution failed."
                            );
                        } else {
                            RCLCPP_INFO(
                                logger,
                                "Motion execution succeeded."
                            );

                            moveit::core::RobotState commanded_state(
                                *current_state
                            );

                            commanded_state.setJointGroupPositions(
                                joint_model_group,
                                best_joint_values
                            );

                            commanded_state.update();

                            const Eigen::Isometry3d commanded_tcp_pose =
                                commanded_state.getGlobalLinkTransform(
                                    tcp_link
                                );

                            const Eigen::Vector3d commanded_position =
                                commanded_tcp_pose.translation();

                            const double commanded_position_error =
                                (
                                    commanded_position
                                    - requested_position
                                ).norm();

                            RCLCPP_INFO(
                                logger,
                                "Executed commanded TCP: x=%.3f y=%.3f z=%.3f; target error=%.1f mm",
                                commanded_position.x(),
                                commanded_position.y(),
                                commanded_position.z(),
                                commanded_position_error * 1000.0
                            );

                            if (
                                commanded_position_error
                                > maximum_tcp_error
                            ) {
                                RCLCPP_ERROR(
                                    logger,
                                    "Commanded TCP error exceeded the safety threshold."
                                );
                            } else {
                                const Eigen::Quaterniond
                                    commanded_orientation(
                                        commanded_tcp_pose.rotation()
                                    );

                                geometry_msgs::msg::PoseStamped completion;

                                completion.header.stamp =
                                    node->now();
                                completion.header.frame_id =
                                    "base_link";
                                completion.pose.position.x =
                                    commanded_position.x();
                                completion.pose.position.y =
                                    commanded_position.y();
                                completion.pose.position.z =
                                    commanded_position.z();
                                completion.pose.orientation.x =
                                    commanded_orientation.x();
                                completion.pose.orientation.y =
                                    commanded_orientation.y();
                                completion.pose.orientation.z =
                                    commanded_orientation.z();
                                completion.pose.orientation.w =
                                    commanded_orientation.w();

                                motion_completed_publisher->publish(
                                    completion
                                );

                                RCLCPP_INFO(
                                    logger,
                                    "Published motion completion after successful controller execution."
                                );
                            }
                        }
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
            },
              subscription_options
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