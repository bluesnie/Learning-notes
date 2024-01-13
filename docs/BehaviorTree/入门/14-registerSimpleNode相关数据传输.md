###### datetime:2023/06/02 16:21

###### author:nzb

# BT14：registerSimpleNode相关数据传输

- 代码

```c++
#include "first_tree.h"

// using namespace DummyNodes;

class ApproachObject : public BT::SyncActionNode
{
public:
    ApproachObject(const std::string &name) : BT::SyncActionNode(name, {}) {}
    BT::NodeStatus tick() override
    {
        std::cout << "ApproachObject: " << this->name() << std::endl;
        return BT::NodeStatus::SUCCESS;
    }
};

class CustomCondition : public BT::ConditionNode
{
public:
    CustomCondition(const std::string &name, const BT::NodeConfiguration &config) : ConditionNode(name, config) {}
    static BT::PortsList providedPorts()
    {
        return {
            BT::InputPort<std::string>("ktype"),
            BT::InputPort<std::string>("key"),
            BT::InputPort<std::string>("value")};
    }

    BT::NodeStatus tick() override
    {
        std::cout << "--------------------CustomCondition----------------------" << std::endl;
        auto ktype = getInput<std::string>("ktype");
        auto key = getInput<std::string>("key");
        auto value = getInput<std::string>("value");
        std::cout << "ktype: " << ktype.value() << std::endl;
        std::cout << "key: " << key.value() << std::endl;
        std::cout << "value: " << value.value() << std::endl;
        std::cout << "--------------------CustomCondition----------------------\n\n\n"
                  << std::endl;
        return BT::NodeStatus::SUCCESS;
    }
};

class GripperInterface
{
public:
    GripperInterface() : _open(true) {}

    BT::NodeStatus open(BT::TreeNode &self)
    {
        std::cout << "-------------------GripperInterface open start ---------------------------" << std::endl;
        // 打印黑板的键和值
        auto vector = self.config().blackboard->getKeys();
        for (auto i = vector.begin(); i != vector.end(); i++)
        {
            std::string key_v;
            self.config().blackboard->get(i->to_string(), key_v);
            std::cout << "blackboard key is: " << *i << " value is: " << key_v << std::endl;
        }
        std::cout << "-------------------blackboard debugMessaget ---------------------------" << std::endl;
        self.config().blackboard->debugMessage();
        std::cout << "-------------------blackboard debugMessaget ---------------------------\n\n"
                  << std::endl;
        // 不同方法改变黑板 key1 中的值
        std::cout << "\nsetOutput open3, the value is: 456, the blackboard key is 'key1' \n";
        std::string key1_v;
        self.setOutput("open3", "456");

        self.config().blackboard->get("key1", key1_v);
        std::cout << "get blackboard 'key1' value --->" << key1_v << std::endl;

        std::cout << "\nself.config().blackboard->set(\"key1\", \"789\");, the value is: 789, the blackboard key is 'key1' \n";
        self.config().blackboard->set("key1", "789");
        self.config().blackboard->get("key1", key1_v);
        std::cout << "self.config().blackboard->get blackboard 'key1' value --->" << key1_v << std::endl;

        std::cout << "\nsetOutput open3, the value is: abc, the blackboard key is 'key1' \n";
        self.setOutput("open3", "abc");                                                                 // open3 指向了黑板 key1
        self.getInput("open4", key1_v);                                                                 // open4 指向了黑板 key1，所以取出来上面修改后的值，未指定类型需要第二个参数
        std::cout << "self.getInput blackboard 'key1' value --->" << key1_v << std::endl;
        
        std::cout << "-------------------port and blackboard size ---------------------------" << std::endl;
        std::cout << "input_post size: " << self.config().input_ports.size() << std::endl;              // 取的都是当前节点的输入数量
        std::cout << "output_ports size: " << self.config().output_ports.size() << std::endl;           // 取的都是当前节点的输出数量
        std::cout << "blackboard size: " << self.config().blackboard->getKeys().size() << std::endl;    // 取的都是当前树的黑板数量
        std::cout << "-------------------port and blackboard size ---------------------------" << std::endl;

        std::cout << "open1: " << self.config().input_ports.find("open1")->second << std::endl;
        std::cout << "open2: " << self.config().input_ports.find("open2")->second << std::endl;         // 取到值不是预期的，应该使用 getInput 拿到映射的黑板值，下一个为示例
        std::cout << "open2: " << self.getInput<std::string>("open2").value() << std::endl;             // getInput指定了类型不需要第二个参数，直接获取
        // std::cout << "open3: " << self.getInput<std::string>("open3").value() << std::endl;          // open3 是 OutputPort 类型，不能获取，只能设置
        std::cout << "open4: " << self.getInput<std::string>("open4").value() << std::endl;

        _open = true;
        std::cout << "-------------------GripperInterface open end ---------------------------\n\n\n"
                  << std::endl;

        std::this_thread::sleep_for(std::chrono::seconds(2));
        return BT::NodeStatus::SUCCESS;
    }

    BT::NodeStatus close()
    {
        _open = false;
        std::cout << "GripperInterface::close" << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(2));
        return BT::NodeStatus::SUCCESS;
    }

private:
    bool _open;
};

void ports_demo()
{
    BT::BehaviorTreeFactory factory;

    factory.registerNodeType<ApproachObject>("ApproachObject");
    factory.registerNodeType<CustomCondition>("CustomCondition");

    GripperInterface gripper;

    BT::PortsList ports = {BT::InputPort<std::string>("open1"),
                           BT::InputPort<std::string>("open2"),
                           BT::OutputPort<std::string>("open3"),
                           BT::InputPort<std::string>("open4")};
    factory.registerSimpleAction("OpenGripper", std::bind(&GripperInterface::open, gripper, std::placeholders::_1), ports);
    factory.registerSimpleAction("CloseGripper", std::bind([&]()
                                                           { return gripper.close(); }));

    auto tree = factory.createTreeFromFile("../src/study/xmls/001_ports_demo.xml");

    BT::PublisherZMQ pub_zmq(tree);

    tree.tickRootWhileRunning();
}
```

- 行为树

```xml

<root main_tree_to_execute="MainTree">
    <!--  //////////  -->
    <BehaviorTree ID="MainTree">
        <Sequence name="root_sequence">
            <SetBlackboard output_key="key" value="-1;3;2"/>
            <SetBlackboard output_key="key1" value="321"/>
            <Condition ID="CustomCondition" key="{key}" ktype="{key1}" name="custion_condition" value="789"/>
            <Action ID="OpenGripper" name="open_gripper" open1="open1_val" open2="{key}" open3="{key1}" open4="{key1}"/>
            <Action ID="ApproachObject" name="approach_object"/>
            <Action ID="CloseGripper" name="close_gripper"/>
        </Sequence>
    </BehaviorTree>
    <!--  //////////  -->
    <TreeNodesModel>
        <Action ID="ApproachObject"/>
        <Action ID="CloseGripper"/>
        <Condition ID="CustomCondition">
            <inout_port name="key"/>
            <inout_port name="ktype"/>
            <inout_port name="value"/>
        </Condition>
        <Action ID="OpenGripper"/>
    </TreeNodesModel>
    <!--  //////////  -->
</root>
```

![](./imgs/Screenshot%20from%202023-06-02%2016-28-17.png)

- 结果

```text
--------------------CustomCondition----------------------
ktype: 321
key: -1;3;2
value: 789
--------------------CustomCondition----------------------


-------------------GripperInterface open start ---------------------------
blackboard key is: key1 value is: 321
blackboard key is: key value is: -1;3;2
-------------------blackboard debugMessaget ---------------------------
key1 (std::string) -> full
key (std::string) -> full
-------------------blackboard debugMessaget ---------------------------


setOutput open3, the value is: 456, the blackboard key is 'key1' 
get blackboard 'key1' value --->456

self.config().blackboard->set("key1", "789");, the value is: 789, the blackboard key is 'key1' 
self.config().blackboard->get blackboard 'key1' value --->789

setOutput open3, the value is: abc, the blackboard key is 'key1' 
self.getInput blackboard 'key1' value --->abc
-------------------port and blackboard size ---------------------------
input_post size: 3
output_ports size: 1
blackboard size: 2
-------------------port and blackboard size ---------------------------
open1: open1_val
open2: {key}
open2: -1;3;2
open4: abc
-------------------GripperInterface open end ---------------------------



ApproachObject: approach_object
GripperInterface::close
```

- 结论

    - `OutputPort`：只能`setOutput`操作
    - `InputPort`：只能`getInput`操作
    - `registerSimpleAction`
        - 简单节点可以提供`ports`，**但是在`Groot`工具中刚开始不显示，需要再`xml`中`TreeNodesModel`对应节点上加上参数**，如上图的 `CustomCondition`节点一样
        - 操作数据，前提条件函数需要添加一个节点参数，如：
          `open(BT::TreeNode &self); factory.registerSimpleAction("OpenGripper", std::bind(&GripperInterface::open, gripper, std::placeholders::_1), ports);`
            - `std::placeholders::_1`：占位使用
            - 读取`ports`数据：`self.getInput`
                - 写法1，指定类型：`Position2D data = self.getInput<Position2D>("key");`
                - 写法2，先声明了数据：`Position2D pos; self.getInput("key", pos);`
            - 设置`ports`数据：`self.setOutput`
            - 读取黑板数据：`std::string key1_v; self.config().blackboard->get("key1", key1_v);`
            - 设置黑板数据：`self.config().blackboard->set("key1", "789");`