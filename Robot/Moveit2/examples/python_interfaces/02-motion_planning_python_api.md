###### datetime:2025/02/17 16:24

###### author:nzb

# [Motion Planning Python API](https://moveit.picknik.ai/main/doc/examples/motion_planning_python_api/motion_planning_python_api_tutorial.html#)

### 功能目录

- 入门：教程配置要求概述。
- 了解规划参数：支持规划器的参数配置概述。
- 单管道规划（默认配置）：使用预先指定的机器人配置进行规划。
- 单管道规划（机器人状态）：使用机器人状态实例进行规划。
- 单管道规划（姿势目标）：使用姿势目标进行规划。
- 单管道规划（自定义约束）：使用自定义约束进行规划。
- 多管道规划：并行运行多个规划管道。
- 使用规划场景：添加和删除碰撞对象以及碰撞检查。

### 创建项目

```shell
ros2 pkg create --build-type ament_python motion_planning_python_api

src/learning_demo/motion_planning_py_api/
├── config
│   ├── motion_planning_python_api_tutorial.rviz
│   └── motion_planning_python_api_tutorial.yaml
├── launch
│   └── motion_planning_python_api_tutorial.launch.py
├── motion_planning_py_api
│   ├── __init__.py
│   ├── motion_planning_python_api_planning_scene.py
│   └── motion_planning_python_api_tutorial.py
├── package.xml
├── resource
│   └── motion_planning_py_api
├── setup.cfg
├── setup.py
└── test
    ├── test_copyright.py
    ├── test_flake8.py
    └── test_pep257.py
```

### 编写代码

- `motion_planning_python_api_planning_scene.py`

```python
"""
Shows how to use a planning scene in MoveItPy to add collision objects and perform collision checking.
"""

import time
import rclpy
from rclpy.logging import get_logger

from moveit.planning import MoveItPy

from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive


def plan_and_execute(
    robot,
    planning_component,
    logger,
    sleep_time=0.0,
):
    """Helper function to plan and execute a motion."""
    # plan to goal
    logger.info("Planning trajectory")
    plan_result = planning_component.plan()

    # execute the plan
    if plan_result:
        logger.info("Executing plan")
        robot_trajectory = plan_result.trajectory
        robot.execute(robot_trajectory, controllers=[])
    else:
        logger.error("Planning failed")

    time.sleep(sleep_time)


def add_collision_objects(planning_scene_monitor):
    """Helper function that adds collision objects to the planning scene."""
    object_positions = [
        (0.15, 0.1, 0.5),
        (0.25, 0.0, 1.0),
        (-0.25, -0.3, 0.8),
        (0.25, 0.3, 0.75),
    ]
    object_dimensions = [
        (0.1, 0.4, 0.1),
        (0.1, 0.4, 0.1),
        (0.2, 0.2, 0.2),
        (0.15, 0.15, 0.15),
    ]

    with planning_scene_monitor.read_write() as scene:
        collision_object = CollisionObject()
        collision_object.header.frame_id = "panda_link0"
        collision_object.id = "boxes"

        for position, dimensions in zip(object_positions, object_dimensions):
            box_pose = Pose()
            box_pose.position.x = position[0]
            box_pose.position.y = position[1]
            box_pose.position.z = position[2]

            box = SolidPrimitive()
            box.type = SolidPrimitive.BOX
            box.dimensions = dimensions

            collision_object.primitives.append(box)
            collision_object.primitive_poses.append(box_pose)
            collision_object.operation = CollisionObject.ADD

        scene.apply_collision_object(collision_object)
        scene.current_state.update()  # Important to ensure the scene is updated


def main():
    ###################################################################
    # MoveItPy Setup
    ###################################################################
    rclpy.init()
    logger = get_logger("moveit_py_planning_scene")

    # instantiate MoveItPy instance and get planning component
    panda = MoveItPy(node_name="moveit_py_planning_scene")
    panda_arm = panda.get_planning_component("panda_arm")
    planning_scene_monitor = panda.get_planning_scene_monitor()
    logger.info("MoveItPy instance created")

    ###################################################################
    # Plan with collision objects
    ###################################################################

    add_collision_objects(planning_scene_monitor)
    panda_arm.set_start_state(configuration_name="ready")
    panda_arm.set_goal_state(configuration_name="extended")
    plan_and_execute(panda, panda_arm, logger, sleep_time=3.0)

    ###################################################################
    # Check collisions
    ###################################################################
    with planning_scene_monitor.read_only() as scene:
        robot_state = scene.current_state
        original_joint_positions = robot_state.get_joint_group_positions("panda_arm")

        # Set the pose goal
        pose_goal = Pose()
        pose_goal.position.x = 0.25
        pose_goal.position.y = 0.25
        pose_goal.position.z = 0.5
        pose_goal.orientation.w = 1.0

        # Set the robot state and check collisions
        robot_state.set_from_ik("panda_arm", pose_goal, "panda_hand")
        robot_state.update()  # required to update transforms
        robot_collision_status = scene.is_state_colliding(
            robot_state=robot_state, joint_model_group_name="panda_arm", verbose=True
        )
        logger.info(f"\nRobot is in collision: {robot_collision_status}\n")

        # Restore the original state
        robot_state.set_joint_group_positions(
            "panda_arm",
            original_joint_positions,
        )
        robot_state.update()  # required to update transforms

    time.sleep(3.0)

    ###################################################################
    # Remove collision objects and return to the ready pose
    ###################################################################

    with planning_scene_monitor.read_write() as scene:
        scene.remove_all_collision_objects()
        scene.current_state.update()

    panda_arm.set_start_state_to_current_state()
    panda_arm.set_goal_state(configuration_name="ready")
    plan_and_execute(panda, panda_arm, logger, sleep_time=3.0)


if __name__ == "__main__":
    main()
```


- `motion_planning_python_api_tutorial.py`

```python
#!/usr/bin/env python3
"""
A script to outline the fundamentals of the moveit_py motion planning API.
"""

import time

# generic ros libraries
import rclpy
from rclpy.logging import get_logger

# moveit python library
from moveit.core.robot_state import RobotState
from moveit.planning import (
    MoveItPy,
    MultiPipelinePlanRequestParameters,
)


def plan_and_execute(
    robot,
    planning_component,
    logger,
    single_plan_parameters=None,
    multi_plan_parameters=None,
    sleep_time=0.0,
):
    """Helper function to plan and execute a motion."""
    # plan to goal
    logger.info("Planning trajectory")
    if multi_plan_parameters is not None:
        plan_result = planning_component.plan(
            multi_plan_parameters=multi_plan_parameters
        )
    elif single_plan_parameters is not None:
        plan_result = planning_component.plan(
            single_plan_parameters=single_plan_parameters
        )
    else:
        plan_result = planning_component.plan()

    # execute the plan
    if plan_result:
        logger.info("Executing plan")
        robot_trajectory = plan_result.trajectory
        robot.execute(robot_trajectory, controllers=[])
    else:
        logger.error("Planning failed")

    time.sleep(sleep_time)


def main():

    ###################################################################
    # MoveItPy Setup
    ###################################################################
    rclpy.init()
    logger = get_logger("moveit_py.pose_goal")

    # instantiate MoveItPy instance and get planning component
    panda = MoveItPy(node_name="moveit_py")
    panda_arm = panda.get_planning_component("panda_arm")
    logger.info("MoveItPy instance created")

    ###########################################################################
    # Plan 1 - set states with predefined string
    ###########################################################################

    # set plan start state using predefined state
    panda_arm.set_start_state(configuration_name="ready")

    # set pose goal using predefined state
    panda_arm.set_goal_state(configuration_name="extended")

    # plan to goal
    plan_and_execute(panda, panda_arm, logger, sleep_time=3.0)

    ###########################################################################
    # Plan 2 - set goal state with RobotState object
    ###########################################################################

    # instantiate a RobotState instance using the current robot model
    robot_model = panda.get_robot_model()
    robot_state = RobotState(robot_model)

    # randomize the robot state
    robot_state.set_to_random_positions()

    # set plan start state to current state
    panda_arm.set_start_state_to_current_state()

    # set goal state to the initialized robot state
    logger.info("Set goal state to the initialized robot state")
    panda_arm.set_goal_state(robot_state=robot_state)

    # plan to goal
    plan_and_execute(panda, panda_arm, logger, sleep_time=3.0)

    ###########################################################################
    # Plan 3 - set goal state with PoseStamped message
    ###########################################################################

    # set plan start state to current state
    panda_arm.set_start_state_to_current_state()

    # set pose goal with PoseStamped message
    from geometry_msgs.msg import PoseStamped

    pose_goal = PoseStamped()
    pose_goal.header.frame_id = "panda_link0"
    pose_goal.pose.orientation.w = 1.0
    pose_goal.pose.position.x = 0.28
    pose_goal.pose.position.y = -0.2
    pose_goal.pose.position.z = 0.5
    panda_arm.set_goal_state(pose_stamped_msg=pose_goal, pose_link="panda_link8")

    # plan to goal
    plan_and_execute(panda, panda_arm, logger, sleep_time=3.0)

    ###########################################################################
    # Plan 4 - set goal state with constraints
    ###########################################################################

    # set plan start state to current state
    panda_arm.set_start_state_to_current_state()

    # set constraints message
    from moveit.core.kinematic_constraints import construct_joint_constraint

    joint_values = {
        "panda_joint1": -1.0,
        "panda_joint2": 0.7,
        "panda_joint3": 0.7,
        "panda_joint4": -1.5,
        "panda_joint5": -0.7,
        "panda_joint6": 2.0,
        "panda_joint7": 0.0,
    }
    robot_state.joint_positions = joint_values
    joint_constraint = construct_joint_constraint(
        robot_state=robot_state,
        joint_model_group=panda.get_robot_model().get_joint_model_group("panda_arm"),
    )
    panda_arm.set_goal_state(motion_plan_constraints=[joint_constraint])

    # plan to goal
    plan_and_execute(panda, panda_arm, logger, sleep_time=3.0)

    ###########################################################################
    # Plan 5 - Planning with Multiple Pipelines simultaneously
    ###########################################################################

    # set plan start state to current state
    panda_arm.set_start_state_to_current_state()

    # set pose goal with PoseStamped message
    panda_arm.set_goal_state(configuration_name="ready")

    # initialise multi-pipeline plan request parameters
    multi_pipeline_plan_request_params = MultiPipelinePlanRequestParameters(
        panda, ["ompl_rrtc", "pilz_lin", "chomp_planner"]
    )

    # plan to goal
    plan_and_execute(
        panda,
        panda_arm,
        logger,
        multi_plan_parameters=multi_pipeline_plan_request_params,
        sleep_time=3.0,
    )


if __name__ == "__main__":
    main()
```

- `motion_planning_python_api_tutorial.launch.py`

```python
"""
A launch file for running the motion planning python api tutorial
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder(
            robot_name="panda", package_name="moveit_resources_panda_moveit_config"
        )
        .robot_description(file_path="config/panda.urdf.xacro")
        .trajectory_execution(file_path="config/gripper_moveit_controllers.yaml")
        .moveit_cpp(
            file_path=get_package_share_directory("motion_planning_py_api")
            + "/config/motion_planning_python_api_tutorial.yaml"
        )
        .to_moveit_configs()
    )

    # ros2 launch motion_planning_py_api motion_planning_python_api_tutorial.launch.py
    # ros2 launch motion_planning_py_api motion_planning_python_api_tutorial.launch.py execute_name:=moveit_py_planning_scene
    execute_name = DeclareLaunchArgument(
        "execute_name",
        default_value="moveit_py_tutorial",
        description="Python API tutorial file name",
    )

    moveit_py_node = Node(
        name="moveit_py",
        package="motion_planning_py_api",
        executable=LaunchConfiguration("execute_name"),
        output="both",
        parameters=[moveit_config.to_dict()],
    )

    rviz_config_file = os.path.join(
        get_package_share_directory("motion_planning_py_api"),
        "config",
        "motion_planning_python_api_tutorial.rviz",
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
        ],
    )

    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        arguments=["--frame-id", "world", "--child-frame-id", "panda_link0"],
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="log",
        parameters=[moveit_config.robot_description],
    )

    ros2_controllers_path = os.path.join(
        get_package_share_directory("moveit_resources_panda_moveit_config"),
        "config",
        "ros2_controllers.yaml",
    )
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[ros2_controllers_path],
        remappings=[
            ("/controller_manager/robot_description", "/robot_description"),
        ],
        output="log",
    )

    load_controllers = []
    for controller in [
        "panda_arm_controller",
        "panda_hand_controller",
        "joint_state_broadcaster",
    ]:
        load_controllers += [
            ExecuteProcess(
                cmd=["ros2 run controller_manager spawner {}".format(controller)],
                shell=True,
                output="log",
            )
        ]

    return LaunchDescription(
        [
            execute_name,
            moveit_py_node,
            robot_state_publisher,
            ros2_control_node,
            rviz_node,
            static_tf,
        ]
        + load_controllers
    )

```


- `setup.py`

```python
from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'motion_planning_py_api'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='blues',
    maintainer_email='nzbwork@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "moveit_py_planning_scene = motion_planning_py_api.motion_planning_python_api_planning_scene:main",
            "moveit_py_tutorial = motion_planning_py_api.motion_planning_python_api_tutorial:main",
        ],
    },
)

```

- `package.xml`

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>motion_planning_py_api</name>
  <version>0.0.0</version>
  <description>TODO: Package description</description>
  <maintainer email="nzbwork@gmail.com">blues</maintainer>
  <license>TODO: License declaration</license>

  <build_depend>moveit_py</build_depend>
  <build_depend>rclcpp</build_depend>
  <build_depend>geometry_msgs</build_depend>
  <build_depend>moveit_msgs</build_depend>
  <build_depend>shape_msgs</build_depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>

```

- `motion_planning_python_api_tutorial.yaml`
  - `planning_scene_monitor_options` 设置规划场景监视器选项，如订阅的主题（注：如果您对规划场景监视器不熟悉，请参[考本教程](../cplus_interfaces/03-planning_scene_monitor.md)）
  - `planning_pipelines` 设置了我们希望使用的规划管道。 `MoveIt` 支持多种运动规划库，包括 `OMPL`、`Pilz Industrial Motion Planner`、`STOMP`、`SBPL` 和 `CHOMP` 等。 
    在配置 `moveit_py` 节点时，我们需要指定希望使用的规划管道配置,对于每个命名管道，我们必须提供一个配置，通过 `planner_id` 和其他设置（例如规划尝试次数）来标识要使用的规划器

```yaml
planning_scene_monitor_options:
  name: "planning_scene_monitor"
  robot_description: "robot_description"
  joint_state_topic: "/joint_states"
  attached_collision_object_topic: "/moveit_cpp/planning_scene_monitor"
  publish_planning_scene_topic: "/moveit_cpp/publish_planning_scene"
  monitored_planning_scene_topic: "/moveit_cpp/monitored_planning_scene"
  wait_for_initial_state_timeout: 10.0

planning_pipelines:
  # pipeline_names: ["ompl", "pilz_industrial_motion_planner", "chomp"]
  pipeline_names: ["ompl", "chomp"]

plan_request_params:
  planning_attempts: 1
  planning_pipeline: ompl
  max_velocity_scaling_factor: 1.0
  max_acceleration_scaling_factor: 1.0

ompl_rrtc:  # Namespace for individual plan request
  plan_request_params:  # PlanRequestParameters similar to the ones that are used by the single pipeline planning of moveit_cpp
    planning_attempts: 1  # Number of attempts the planning pipeline tries to solve a given motion planning problem
    planning_pipeline: ompl  # Name of the pipeline that is being used
    planner_id: "RRTConnectkConfigDefault"  # Name of the specific planner to be used by the pipeline
    max_velocity_scaling_factor: 1.0  # Velocity scaling parameter for the trajectory generation algorithm that is called (if configured) after the path planning
    max_acceleration_scaling_factor: 1.0  # Acceleration scaling parameter for the trajectory generation algorithm that is called (if configured) after the path planning
    planning_time: 1.0  # Time budget for the motion plan request. If the planning problem cannot be solved within this time, an empty solution with error code is returned

# pilz_lin:
#   plan_request_params:
#     planning_attempts: 1
#     planning_pipeline: pilz_industrial_motion_planner
#     planner_id: "PTP"
#     max_velocity_scaling_factor: 1.0
#     max_acceleration_scaling_factor: 1.0
#     planning_time: 0.8

chomp_planner:
  plan_request_params:
    planning_attempts: 1
    planning_pipeline: chomp
    max_velocity_scaling_factor: 1.0
    max_acceleration_scaling_factor: 1.0
    planning_time: 1.5
```

- [`motion_planning_python_api_tutorial.rviz`](https://github.com/moveit/moveit2_tutorials/blob/main/doc/examples/motion_planning_python_api/config/motion_planning_python_api_tutorial.rviz)