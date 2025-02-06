###### datetime:2025/01/27 11:26

###### author:nzb

# [规划场景ROS接口](https://moveit.picknik.ai/main/doc/examples/planning_scene_ros_api/planning_scene_ros_api_tutorial.html#planning-scene-ros-api)

## 功能目录

- 添加一个对象到环境中
- 可视化 同步更新与异步更新
- 附加对象到机器人上
- 删除机器人上的对象
- 从碰撞世界(环境)移除对象

## 代码解读

```c++
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>

// MoveIt
#include <moveit_msgs/msg/planning_scene.hpp>
#include <moveit_msgs/msg/attached_collision_object.hpp>
#include <moveit_msgs/srv/get_state_validity.hpp>
#include <moveit_msgs/msg/display_robot_state.hpp>
#include <moveit_msgs/srv/apply_planning_scene.hpp>

#include <moveit/robot_model_loader/robot_model_loader.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <moveit/robot_state/conversions.hpp>

#include <rviz_visual_tools/rviz_visual_tools.hpp>

static const rclcpp::Logger LOGGER = rclcpp::get_logger("planning_scene_ros_api_tutorial");

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions node_options;
  node_options.automatically_declare_parameters_from_overrides(true);
  auto node = rclcpp::Node::make_shared("planning_scene_ros_api_tutorial", node_options);

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  std::thread([&executor]() { executor.spin(); }).detach();
  // Visualization
  // 可视化
  // ^^^^^^^^^^^^^
  // MoveItVisualTools包提供了在RViz中可视化对象、机器人和轨迹以及调试工具（如脚本的一步一步的调试）等多种功能。
  rviz_visual_tools::RvizVisualTools visual_tools("panda_link0", "planning_scene_ros_api_tutorial", node);
  visual_tools.loadRemoteControl();
  visual_tools.deleteAllMarkers();

  // ROS API
  // ROS 接口
  // ^^^^^^^
  // ROS API通过使用“差分”的topic接口来发布规划场景。规划场景差分是当前规划场景（由move_group节点维护）和用户所需的新规划场景之间的差异。
  //
  // Advertise the required topic
  // 发布所需的topic
  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // 我们创建一个发布者并等待订阅者。请注意，此topic可能需要在启动文件中重新映射。
  rclcpp::Publisher<moveit_msgs::msg::PlanningScene>::SharedPtr planning_scene_diff_publisher =
      node->create_publisher<moveit_msgs::msg::PlanningScene>("planning_scene", 1);
  while (planning_scene_diff_publisher->get_subscription_count() < 1)
  {
    rclcpp::sleep_for(std::chrono::milliseconds(500));
  }
  visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to continue the demo");

  // Define the attached object message
  // 定义附加对象消息
  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // 我们将使用此消息将对象添加到世界或从世界中移除，并将对象附加到机器人上。
  moveit_msgs::msg::AttachedCollisionObject attached_object;
  attached_object.link_name = "panda_hand";
  /* The header must contain a valid TF frame*/
  attached_object.object.header.frame_id = "panda_hand";
  /* The id of the object */
  attached_object.object.id = "box";

  /* A default pose */
  geometry_msgs::msg::Pose pose;
  pose.position.z = 0.11;
  pose.orientation.w = 1.0;

  /* Define a box to be attached */
  shape_msgs::msg::SolidPrimitive primitive;
  primitive.type = primitive.BOX;
  primitive.dimensions.resize(3);
  primitive.dimensions[0] = 0.075;
  primitive.dimensions[1] = 0.075;
  primitive.dimensions[2] = 0.075;

  attached_object.object.primitives.push_back(primitive);
  attached_object.object.primitive_poses.push_back(pose);

  // 将对象附加到机器人上需要将相应的操作指定为ADD操作。
  attached_object.object.operation = attached_object.object.ADD;

  //由于我们将对象附加到机器人手上以模拟拾取对象，我们希望碰撞检查器忽略对象和机器人手之间的碰撞。
  attached_object.touch_links = std::vector<std::string>{ "panda_hand", "panda_leftfinger", "panda_rightfinger" };

  // Add an object into the environment
  // 将一个对象添加到环境中
  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // 通过将对象添加到规划场景中“world”部分的碰撞对象集合中，将对象添加到环境中。请注意，我们在这里只使用attached_object消息的“object”字段。
  RCLCPP_INFO(LOGGER, "Adding the object into the world at the location of the hand.");
  moveit_msgs::msg::PlanningScene planning_scene;
  planning_scene.world.collision_objects.push_back(attached_object.object);
  planning_scene.is_diff = true;
  planning_scene_diff_publisher->publish(planning_scene);
  visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to continue the demo");

  // Interlude: Synchronous vs Asynchronous updates
  // 插曲：同步与异步更新
  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // 使用差分与move_group节点交互有两种不同的机制：
  //
  // * 通过rosservice调用发送差分，并阻塞直到应用差分（同步更新）
  // * 通过topic发送差分，即使差分尚未应用也继续（异步更新）
  //
  // While most of this tutorial uses the latter mechanism (given the long sleeps inserted for visualization purposes asynchronous updates do not pose a problem),
  // 虽然本教程的大部分内容使用后者机制（由于插入了用于可视化的长时间睡眠，异步更新不会出现问题），
  //  it would be perfectly justified to replace the planning_scene_diff_publisher by the following service client:
  //  可以完全合理地用以下服务客户端替换planning_scene_diff_publisher：
  rclcpp::Client<moveit_msgs::srv::ApplyPlanningScene>::SharedPtr planning_scene_diff_client =
      node->create_client<moveit_msgs::srv::ApplyPlanningScene>("apply_planning_scene");
  planning_scene_diff_client->wait_for_service();
  // and send the diffs to the planning scene via a service call:
  auto request = std::make_shared<moveit_msgs::srv::ApplyPlanningScene::Request>();
  request->scene = planning_scene;
  std::shared_future<std::shared_ptr<moveit_msgs::srv::ApplyPlanningScene_Response>> response_future;
  response_future = planning_scene_diff_client->async_send_request(request).future.share();

  // wait for the service to respond
  std::chrono::seconds wait_time(1);
  std::future_status fs = response_future.wait_for(wait_time);
  if (fs == std::future_status::timeout)
  {
    RCLCPP_ERROR(LOGGER, "Service timed out.");
  }
  else
  {
    std::shared_ptr<moveit_msgs::srv::ApplyPlanningScene_Response> planning_response;
    planning_response = response_future.get();
    if (planning_response->success)
    {
      RCLCPP_INFO(LOGGER, "Service successfully added object.");
    }
    else
    {
      RCLCPP_ERROR(LOGGER, "Service failed to add object.");
    }
  }

  // 请注意，这不会继续，直到我们确定差分已应用。
  //
  // Attach an object to the robot
  // 将一个对象附加到机器人上
  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // 当机器人从环境中拿起一个物体时，我们需要将物体“附加”到机器人上，以便任何处理机器人模型的组件都知道要考虑附加的物体，例如进行碰撞检查。
  // 将一个物体附加到机器人上需要两个操作
  // * 将原始对象从环境中移除
  // * 将对象附加到机器人上

  /* First, define the REMOVE object message*/
  moveit_msgs::msg::CollisionObject remove_object;
  remove_object.id = "box";
  remove_object.header.frame_id = "panda_hand";
  remove_object.operation = remove_object.REMOVE;

  // 请注意，我们如何通过首先清除那些字段来确保差分消息不包含其他附加对象或碰撞对象。
  /* Carry out the REMOVE + ATTACH operation */
  RCLCPP_INFO(LOGGER, "Attaching the object to the hand and removing it from the world.");
  planning_scene.world.collision_objects.clear();
  planning_scene.world.collision_objects.push_back(remove_object);
  planning_scene.robot_state.attached_collision_objects.push_back(attached_object);
  planning_scene.robot_state.is_diff = true;
  planning_scene_diff_publisher->publish(planning_scene);
  visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to continue the demo");

  // Detach an object from the robot
  // 从机器人上分离一个物体
  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // 从机器人上分离一个物体需要两个操作
  //  * 从机器人上分离对象
  //  * 将对象重新引入环境

  /* First, define the DETACH object message*/
  moveit_msgs::msg::AttachedCollisionObject detach_object;
  detach_object.object.id = "box";
  detach_object.link_name = "panda_hand";
  detach_object.object.operation = attached_object.object.REMOVE;

  // 请注意，我们如何通过首先清除那些字段来确保差分消息不包含其他附加对象或碰撞对象。
  /* Carry out the DETACH + ADD operation */
  RCLCPP_INFO(LOGGER, "Detaching the object from the robot and returning it to the world.");
  planning_scene.robot_state.attached_collision_objects.clear();
  planning_scene.robot_state.attached_collision_objects.push_back(detach_object);
  planning_scene.robot_state.is_diff = true;
  planning_scene.world.collision_objects.clear();
  planning_scene.world.collision_objects.push_back(attached_object.object);
  planning_scene.is_diff = true;
  planning_scene_diff_publisher->publish(planning_scene);
  visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to continue the demo");

  // Remove the object from the collision world
  // 从碰撞世界中移除对象
  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // 从碰撞世界中移除对象只需要使用之前定义的移除对象消息。
  // 注意，我们如何通过首先清除那些字段来确保差分消息不包含其他附加对象或碰撞对象。
  RCLCPP_INFO(LOGGER, "Removing the object from the world.");
  planning_scene.robot_state.attached_collision_objects.clear();
  planning_scene.world.collision_objects.clear();
  planning_scene.world.collision_objects.push_back(remove_object);
  planning_scene_diff_publisher->publish(planning_scene);
  // END_TUTORIAL

  visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to end the demo");

  rclcpp::shutdown();
  return 0;
}
```