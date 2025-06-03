###### datetime:2025/05/07 11:01

###### author:nzb

# [控制器链接/级联控制](https://control.ros.org/humble/doc/ros2_control/controller_manager/doc/controller_chaining.html)

本文档提出了串行控制器链接的最小可行实现，如 [`Chaining Controllers`](https://github.com/ros-controls/roadmap/blob/master/design_drafts/controller_chaining.md) 设计文档中所述。级联控制是一种特定类型的控制器链接。

## 文档范围和背景知识

这种方法只关注控制器的串行链接，并试图重用现有的机制。它侧重于控制器的输入和输出及其在 `controller manager` 中的管理。出于清晰的原因，将引入 `controller groups` 的概念，其唯一含义是该组中的 `controller` 可以按任意顺序更新。这并不意味着将来不会引入和使用控制器链接文档中描述的控制器组。尽管如此，作者相信，这只会在这个阶段增加不必要的复杂性，尽管从长远来看 ，它们可以提供更清晰的结构和接口。

## 动机、目的和用途 

为了描述本文档的意图，让我们关注 `controllers_chaining` 设计文档中简单但充分的示例示例 2：

![](../../imgs/chaining_example2.png)

在此示例中，我们希望将 `position_tracking` 控制器与 `diff_drive_controller` 和两个 `PID` 控制器链接起来。现在让我们想象一个用例，我们不仅希望将所有这些控制器作为一个组运行，而且还希望灵活地添加前面的步骤。这意味着：

- 当机器人启动时，我们要检查电机速度控制是否正常工作，因此仅激活 `PID` 控制器。在这个阶段，我们也可以使用 `topics` 从外部控制 `PID` 控制器的输入。但是，这些控制器也提供了虚拟接口，因此我们可以将它们链接起来。
- 然后，`diff_drive_controller`被激活并连接到 `PID` 控制器的虚拟输入接口。`PID` 控制器还会被告知它们正在链式模式下工作，因此通过订阅者禁用其外部接口。现在我们检查差速机器人的运动学是否正常运行。
- 之后，可以激活 `position_tracking` 并将其附加到禁用其外部接口的 `diff_drive_controller`。
- 如果任何 `Controller` 被停用，则前面的所有 `Controller` 也会被停用。

> 注意：仅当其他控制器使用其引用接口时，暴露引用接口的控制器才会切换到链接模式。当其他控制器未使用其引用接口时，它们会切换回从订阅者获取引用。但是，当其他控制器使用其状态接口时，暴露状态接口的控制器不会触发到链接模式。

> 注意：本文档使用了前控制器和后控制器两个术语。这些术语指控制器的排序，即如果控制器 A 要求（将其输出连接到）控制器 B 的参考接口（输入），则控制器 A 优先于控制器 B。在本节开头的示例图中，`diff_drive_controller`排在 "`pid` 左轮 "和 "`pid` 右轮 "之前。因此，"`pid` 左轮 "和 "`pid` 右轮 "是位于`diff_drive_controller`之后的控制器。

## 实现

### 控制器基类：ChainableController

`ChainableController` 扩展了 `ControllerInterface` 类，并带有虚拟 `InterfaceConfiguration input_interface_configuration() const = 0` 方法。每个控制器的前面都可以有另一个控制器导出所有输入接口，每个控制器都应使用该方法。为简单起见，目前假定使用控制器的所有输入接口。因此，请不要尝试实现任何输入接口的排他性组合，如果需要排他性，请编写多个控制器。


`ChainableController` 基类实现了 `void set_chained_mode(bool activate)`，用于设置控制器被另一个控制器使用（链式模式）的内部标记，并调用虚函数 `void on_set_chained_mode(bool activate) = 0`，用于在链式模式激活或停用时实现控制器的特定操作，例如停用订阅者。

例如，PID 控制器导出一个虚拟接口 `pid_reference`，并在连锁模式下使用时停止其订阅用户 `<controller_name>/pid_reference`。`diff_drive_controller` 控制器输出虚拟接口 `<controller_name>/v_x、<controller_name>/v_y` 和 `<controller_name>/w_z` 的列表，并停止主题 `<controller_name>/cmd_vel` 和 `<controller_name>/cmd_vel_unstamped` 的订阅者。其发布程序可继续运行。

### 内部资源管理

配置可链控制器后，控制器管理器会调用 `input_interface_configuration` 方法，并接管控制器的输入接口。这一过程与 `ResourceManager` 和硬件接口的过程相同。控制器管理器将接口的 "已认领 "状态保存在一个向量中（与 `ResourceManager` 的做法相同）。

### 激活和停用链式控制器

链接的控制器必须一起激活和停用，或者按正确的顺序激活和停用。这意味着您必须首先激活所有后续控制器才能激活前一个控制器。对于停用，则有相反的规则 - 必须先停用所有前面的控制器，然后才能停用下一个控制器。也可以把它看作是一条实际的链，你不能添加链环或在中间断开链。

## 调试输出

如果引用接口当前没有提供有关任何内容的太多信息，则标记为 `unavailable`。所以不要被它弄糊涂。我们之所以拥有它，是因为内部实现的原因，与用法无关。

## 结束语

也许没有必要添加新控制器的 `ChainableController` 类型。在 `ControllerInterface` 类中添加 `input_interface_configuration()` 方法的实现也是可行的，默认结果为 `interface_configuration_type::NONE`。
