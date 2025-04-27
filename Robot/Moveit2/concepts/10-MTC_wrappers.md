###### datetime:2025/04/27 16:39

###### author:nzb

# 封装器

封装器封装另一个阶段，以修改或过滤结果。

`MTC` 提供以下封装阶段：
- ComputeIK
- PredicateFilter
- PassThrough

## ComputeIK

`ComputeIK` 是任何姿态生成器阶段的封装器，用于计算生成姿态阶段生成的笛卡尔空间中姿态的逆运动学。

| Property Name | Function to set property | Description |
| ----- | ----- | ----- |
| eef | void setEndEffector(std::string eef) | Name of end effector group |
| group | void setGroup(std::string group) | Name of active group. Derived from eef if not provided. （活动组名称。如果未提供，则从 eef 派生。）|
| max_ik_solutions | void setMaxIKSolutions(uint32_t n) | Default is 1 |
| ignore_collisions | void setIgnoreCollisions(bool flag) | Default is false. |
| min_solution_distance | void setMinSolutionDistance(double distance) | Minimum distance between separate IK solutions for the same target. Default is 0.1. |


[API doc for ComputeIK.](https://moveit.github.io/moveit_task_constructor/_static/classmoveit_1_1task__constructor_1_1stages_1_1ComputeIK.html)

示例代码

```cpp
auto stage = std::make_unique<moveit::task_constructor::stages::GenerateVacuumGraspPose>("generate pose");
auto wrapper = std::make_unique<moveit::task_constructor::stages::ComputeIK>("pose IK", std::move(stage));
wrapper->setTimeout(0.05);
wrapper->setIKFrame("tool_frame");
wrapper->properties().configureInitFrom(moveit::task_constructor::Stage::PARENT, { "eef", "group" }); // Property value derived from parent stage
wrapper->properties().configureInitFrom(moveit::task_constructor::Stage::INTERFACE, { "target_pose" }); // Property value derived from child stage

// Users can add a callback function when grasp solutions are generated.
// Here we have added a custom function called publishGraspSolution which can publish the grasp solution to a certain topic.
wrapper->addSolutionCallback(
    [this](const moveit::task_constructor::SolutionBase& solution) { return publishGraspSolution(solution); });
```

## PredicateFilter

`PredicateFilter` 是一个阶段包装器，用于根据自定义条件过滤生成的解决方案。

| Property Name | Function to set property | Description |
| ----- | ----- | ----- |
| predicate | void setPredicate(std::function<bool(const SolutionBase, std::string)> predicate) | 筛选解决方案的断言 |
| ignore_filter | void setIgnoreFilter(bool ignore) | 忽略断言并转发所有解决方案 |

[API doc for PredicateFilter.](https://moveit.github.io/moveit_task_constructor/_static/classmoveit_1_1task__constructor_1_1stages_1_1PredicateFilter.html)


示例代码

```cpp
auto current_state = std::make_unique<moveit::task_constructor::stages::CurrentState>(kStageNameCurrentState);

// Use Predicate filter to fail the MTC task if any links are in collision in planning scene
auto applicability_filter =
    std::make_unique<moveit::task_constructor::stages::PredicateFilter>("current state", std::move(current_state));

applicability_filter->setPredicate([this](const moveit::task_constructor::SolutionBase& s, std::string& comment) {
  if (s.start()->scene()->isStateColliding())
  {
    // Get links that are in collision
    std::vector<std::string> colliding_links;
    s.start()->scene()->getCollidingLinks(colliding_links);

    // Publish the results
    publishLinkCollisions(colliding_links);

    comment = "Links are in collision";
    return false;
  }
  return true;
});
```

## PassThrough

`PassThrough` 是一个传递解决方案的通用包装器。这对于通过 `Stage::setCostTerm` 设置自定义成本转换非常有用，可以在不丢失原始值的情况下更改解决方案的成本。












