###### datetime:2023/05/15 10:05

###### author:nzb

# BT14：自定义的用于枚举类型的SwitchNode

## 内建SwitchNode的局限

`BehaviorTree.CPP`中有内建的`SwitchNode`，定义在 `BehaviorTree.CPP/include/behaviortree_cpp_v3/controls/switch_node.h`。
但是其只能接收`string`类型的值，因为定义的`input port`就是`string`类型。

```cpp
static PortsList providedPorts() {
  PortsList ports;
  ports.insert(BT::InputPort<std::string>("variable"));
  for (unsigned i = 0; i < NUM_CASES; i++) {
    char case_str[20];
    sprintf(case_str, "case_%d", i + 1);
    ports.insert(BT::InputPort<std::string>(case_str));
  }
  return ports;
}
```

这样用起来很不方便，大多数`switch-case`的逻辑都是和枚举值一起用的。因此我实现了一个自定义的`control node`，用来接收枚举类型的值，称为`EnumSwitchNode`。 改动很简单，只需将input
port的类型改为模板，并检查输入类是枚举类型。

## 自定义的EnumSwitchNode

```cpp
#ifndef BTNODES_CONTROL_NODES_ENUM_SWITCH_NODE_H
#define BTNODES_CONTROL_NODES_ENUM_SWITCH_NODE_H

#include "behaviortree_cpp_v3/control_node.h"

namespace BT {
/**
 * @brief BehaviorTree.CPP的SwitchNode只能输入string类型的值
 * 这里添加可以输入枚举类型的值
 */
template <size_t NUM_CASES, typename T>
class EnumSwitchNode : public ControlNode {
 public:
  EnumSwitchNode(const std::string& name, const BT::NodeConfiguration& config)
      : ControlNode::ControlNode(name, config), running_child_(-1) {}

  virtual ~EnumSwitchNode() override = default;

  void halt() override {
    running_child_ = -1;
    ControlNode::halt();
  }

  static PortsList providedPorts() {
    PortsList ports;
    ports.insert(BT::InputPort<T>("enum_variable"));
    for (unsigned i = 0; i < NUM_CASES; i++) {
      char case_str[20];
      sprintf(case_str, "case_%d", i + 1);
      ports.insert(BT::InputPort<std::string>(case_str));
    }
    return ports;
  }

 private:
  int running_child_;
  virtual BT::NodeStatus tick() override;
};

template <size_t NUM_CASES, typename T>
inline NodeStatus EnumSwitchNode<NUM_CASES, T>::tick() {
  // 如果是非枚举类型，会编译报错
  static_assert(std::is_enum<T>::value,
                "[registerNode]: accepts only enum classes!!!");

  constexpr const char* case_port_names[9] = {"case_1", "case_2", "case_3",
                                              "case_4", "case_5", "case_6",
                                              "case_7", "case_8", "case_9"};
  if (childrenCount() != NUM_CASES + 1) {
    throw LogicError(
        "Wrong number of children in EnumSwitchNode; "
        "must be (num_cases + default)");
  }

  T variable, value;
  int child_index = NUM_CASES;  // default index;
  if (getInput("enum_variable", variable)) {
    // check each case until you find a match
    for (unsigned index = 0; index < NUM_CASES; ++index) {
      bool found = false;
      if (index < 9) {
        found = (bool)getInput(case_port_names[index], value);
      } else {
        char case_str[20];
        sprintf(case_str, "case_%d", index + 1);
        found = (bool)getInput(case_str, value);
      }
      if (found && variable == value) {
        child_index = index;
        break;
      }
    }
  }

  // if another one was running earlier, halt it
  if (running_child_ != -1 && running_child_ != child_index) {
    haltChild(running_child_);
  }

  auto& selected_child = children_nodes_[child_index];
  NodeStatus ret = selected_child->executeTick();
  if (ret == NodeStatus::RUNNING) {
    running_child_ = child_index;
  } else {
    haltChildren();
    running_child_ = -1;
  }
  return ret;
}

}  // namespace BT

#endif  // BTNODES_CONTROL_NODES_ENUM_SWITCH_NODE_H
```

### 应用示例

```cpp
// 宠物类型
enum class PetType : uint8_t {
  UNDEFINED = 0,
  DOG,
  CAT,
  BIRD,
};
```

我们创建1个枚举类-宠物类型`PetType`，并实现由`string`向`PetType`转换的函数`convertFromString()`，它将会在读取`xml`中`port`值时自动调用。

```cpp
template <>
inline PetType BT::convertFromString(BT::StringView key) {
  auto parts = BT::splitString(key, ',');
  if (parts.size() != 1) {
    throw BT::RuntimeError("invalid input");
  } else {
    auto str = parts[0];
    if ("PetType::UNDEFINED" == str) {
      return PetType::UNDEFINED;
    } else if ("PetType::DOG" == str) {
      return PetType::DOG;
    } else if ("PetType::CAT" == str) {
      return PetType::CAT;
    } else if ("PetType::BIRD" == str) {
      return PetType::BIRD;
    } else {
      throw BT::RuntimeError(std::string("invalid input, chars=") +
                             str.to_string());
    }
  }
}
```

创建1个简单的行为树，第1个`SetBlackboard`节点负责向`{my_pet} entry`写入枚举值，第2个`SwitchNode`负责根据读到的值，执行对应的分支来打印。

```xml

<root main_tree_to_execute="MainTree">
    <BehaviorTree ID="MainTree">
        <Sequence>
            <!-- 改变my_pet的值，可以运行不同的case -->
            <SetBlackboard output_key="my_pet" value="PetType::DOG"/>
            <EnumSwitch3_PetType enum_variable="{my_pet}" case_1="PetType::DOG" case_2="PetType::CAT"
                                 case_3="PetType::BIRD">
                <Action ID="SaySomething" name="dog_say" message="wang...wang..."/>
                <Action ID="SaySomething" name="cat_say" message="miao...miao..."/>
                <Action ID="SaySomething" name="bird_say" message="bugu...bugu..."/>
                <Action ID="SaySomething" name="default_be_quiet" message="xu..."/>
            </EnumSwitch3_PetType>
        </Sequence>
    </BehaviorTree>
</root>
```

`main()`中，需要向`factory`注册用到的2种`node`: `SaySomething`和`EnumSwitchNode`，这里指定了其接受的枚举类型，还可以同时创建多个不同的`EnumSwitchNode`。

```cpp
int main() {
  BT::BehaviorTreeFactory factory;

  factory.registerNodeType<SaySomething>("SaySomething");
  // 注册要使用的枚举类switch node
  factory.registerNodeType<BT::EnumSwitchNode<3, PetType>>(
      "EnumSwitch3_PetType");

  auto tree = factory.createTreeFromText(xml_text);
  tree.tickRoot();

  return 0;
}
```

`SaySomething node`就是读取`message port`值并打印。

```cpp
class SaySomething : public BT::SyncActionNode {
 public:
  SaySomething(const std::string& name, const BT::NodeConfiguration& config)
      : BT::SyncActionNode(name, config) {}

  // You must override the virtual function tick()
  BT::NodeStatus tick() override {
    auto msg = getInput<std::string>("message");
    if (!msg) {
      throw BT::RuntimeError("missing required input [message]: ", msg.error());
    }
    std::cout << "SaySomething::tick()- " << msg.value() << std::endl;
    return BT::NodeStatus::SUCCESS;
  }

  // It is mandatory to define this static method.
  static BT::PortsList providedPorts() {
    return {BT::InputPort<std::string>("message")};
  }
};
```

直接运行，会得到如下结果。大家可以改变{my_pet} entry的值来观察输出的变化。

```cpp
SaySomething::tick()- wang...wang...
```
