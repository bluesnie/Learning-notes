###### datetime:2023/05/11 15:12

###### author:nzb

# BT3：库中基本类型Factory和Blackboard

## BehaviorTreeFactory

定义在`BehaviorTree.CPP/include/behaviortree_cpp_v3/bt_factory.h`，主要包含3个容器来保存数据。

```cpp
/**
 * @brief The BehaviorTreeFactory is used to create instances of a
 * TreeNode at run-time.
 * Some node types are "builtin", whilst other are used defined and need
 * to be registered using a unique ID.
 */
class BehaviorTreeFactory {
private:
    std::unordered_map<std::string, NodeBuilder> builders_;
    std::unordered_map<std::string, TreeNodeManifest> manifests_;
    std::set<std::string> builtin_IDs_;
}
```

以`BehaviorTree.CPP/examples/t03_generic_ports.cpp`为例，在构造`BehaviorTreeFactory`实例后立即打印这3个容器的元素的名称，
可以发现输出结果仅顺序不同，说明`Factory`构造后默认包含了内建的29个`nodes`。

```text
builders[1]=Switch4
builders[2]=Switch6
builders[3]=BlackboardCheckDouble
builders[4]=BlackboardCheckInt
builders[5]=SubTree
builders[6]=KeepRunningUntilFailure
builders[7]=Switch5
builders[8]=ReactiveSequence
builders[9]=Parallel
builders[10]=Delay
builders[11]=SetBlackboard
builders[12]=SequenceStar
builders[13]=Fallback
builders[14]=AlwaysSuccess
builders[15]=ReactiveFallback
builders[16]=Sequence
builders[17]=Switch3
builders[18]=Switch2
builders[19]=AlwaysFailure
builders[20]=IfThenElse
builders[21]=WhileDoElse
builders[22]=SubTreePlus
builders[23]=ForceSuccess
builders[24]=Inverter
builders[25]=BlackboardCheckString
builders[26]=RetryUntilSuccesful
builders[27]=ForceFailure
builders[28]=Repeat
builders[29]=Timeout

manifests[1]=Switch4
manifests[2]=Switch6
manifests[3]=BlackboardCheckDouble
manifests[4]=BlackboardCheckInt
manifests[5]=SubTree
manifests[6]=KeepRunningUntilFailure
manifests[7]=Switch5
manifests[8]=ReactiveSequence
manifests[9]=Parallel
manifests[10]=Delay
manifests[11]=SetBlackboard
manifests[12]=SequenceStar
manifests[13]=Fallback
manifests[14]=AlwaysSuccess
manifests[15]=ReactiveFallback
manifests[16]=Sequence
manifests[17]=Switch3
manifests[18]=Switch2
manifests[19]=AlwaysFailure
manifests[20]=IfThenElse
manifests[21]=WhileDoElse
manifests[22]=SubTreePlus
manifests[23]=ForceSuccess
manifests[24]=Inverter
manifests[25]=BlackboardCheckString
manifests[26]=RetryUntilSuccesful
manifests[27]=ForceFailure
manifests[28]=Repeat
manifests[29]=Timeout

builtinNodes[1]=AlwaysFailure
builtinNodes[2]=AlwaysSuccess
builtinNodes[3]=BlackboardCheckDouble
builtinNodes[4]=BlackboardCheckInt
builtinNodes[5]=BlackboardCheckString
builtinNodes[6]=Delay
builtinNodes[7]=Fallback
builtinNodes[8]=ForceFailure
builtinNodes[9]=ForceSuccess
builtinNodes[10]=IfThenElse
builtinNodes[11]=Inverter
builtinNodes[12]=KeepRunningUntilFailure
builtinNodes[13]=Parallel
builtinNodes[14]=ReactiveFallback
builtinNodes[15]=ReactiveSequence
builtinNodes[16]=Repeat
builtinNodes[17]=RetryUntilSuccesful
builtinNodes[18]=Sequence
builtinNodes[19]=SequenceStar
builtinNodes[20]=SetBlackboard
builtinNodes[21]=SubTree
builtinNodes[22]=SubTreePlus
builtinNodes[23]=Switch2
builtinNodes[24]=Switch3
builtinNodes[25]=Switch4
builtinNodes[26]=Switch5
builtinNodes[27]=Switch6
builtinNodes[28]=Timeout
builtinNodes[29]=WhileDoElse
```

在`factory`添加了2个`node`后，再次打印，`builders`和`manifests`就多了2个对应的元素。

```cpp
factory.registerNodeType<CalculateGoal>("CalculateGoal");
factory.registerNodeType<PrintTarget>("PrintTarget");
```

`registerNodeType()`的入参，就表明了实际节点类型`T（CalculateGoal）`在树中的名称（同样是`CalculateGoal`）。建议大家实际类型和入参ID相同，可以减少很多迷惑。

```cpp
/** registerNodeType is the method to use to register your custom TreeNode.
 *
 *  It accepts only classed derived from either ActionNodeBase, DecoratorNode,
 *  ControlNode or ConditionNode.
 */
template <typename T>
void registerNodeType(const std::string& ID);
```

因此，在加载`tree`之前，开发者必须先将自定义的`node`注册进入`factory`，才能正确解析`xml`。1个`factory`可以包含多个`tree`。

```cpp
auto tree = factory.createTreeFromText(xml_text);
```

## NodeBuilder

`NodeBuilder`定义在`BehaviorTree.CPP/include/behaviortree_cpp_v3/bt_factory.h`，使用了建造者模式，可以理解为模板化的`node`构造函数，
是偏特化的模板类定义。根据构造函数的参数个数区别，选择不同的`builder`，返回一个智能指针。

```cpp
/// The term "Builder" refers to the Builder Pattern (https://en.wikipedia.org/wiki/Builder_pattern)
typedef std::function<std::unique_ptr<TreeNode>(const std::string&, const NodeConfiguration&)>
    NodeBuilder;

// 检查T是否有带参(const std::string&)的默认构造函数
template <typename T>
using has_default_constructor = typename std::is_constructible<T, const std::string&>;

// 检查T是否有带参(const std::string&, const NodeConfiguration&)的构造函数
template <typename T>
using has_params_constructor =
    typename std::is_constructible<T, const std::string&, const NodeConfiguration&>;

// 对应T既有默认构造函数，又有带2个参数构造函数的情况，即上述2条判断都为true。注意make_unique的参数区分
// 定义了1个T类型的指针，默认值是nullptr。这不重要，重要的是enable_if的条件检查
template <typename T>
inline NodeBuilder
CreateBuilder(typename std::enable_if<has_default_constructor<T>::value &&
                                      has_params_constructor<T>::value>::type* = nullptr) {
    return [](const std::string& name, const NodeConfiguration& config) {
        // Special case. Use default constructor if parameters are empty
        if (config.input_ports.empty() && config.output_ports.empty() &&
            has_default_constructor<T>::value) {
            return std::make_unique<T>(name);
        }
        return std::make_unique<T>(name, config);
    };
}

// 对应T只有带2参构造函数的情况，注意new的参数
template <typename T>
inline NodeBuilder
CreateBuilder(typename std::enable_if<!has_default_constructor<T>::value &&
                                      has_params_constructor<T>::value>::type* = nullptr) {
    return [](const std::string& name, const NodeConfiguration& params) {
        return std::unique_ptr<TreeNode>(new T(name, params));
    };
}

// 对应T只有默认构造函数（带1参）的情况，注意new的参数
template <typename T>
inline NodeBuilder
CreateBuilder(typename std::enable_if<has_default_constructor<T>::value &&
                                      !has_params_constructor<T>::value>::type* = nullptr) {
    return [](const std::string& name, const NodeConfiguration&) {
        return std::unique_ptr<TreeNode>(new T(name));
    };
}
```

### 语法参考资料

- [is_constructible - C++ Reference](https://link.zhihu.com/?target=https%3A//www.cplusplus.com/reference/type_traits/is_constructible/)
- [std::enable_if - cppreference.com](https://link.zhihu.com/?target=https%3A//www.enseignement.polytechnique.fr/informatique/INF478/docs/Cpp/en/cpp/types/enable_if.html)
- [std::enable_if 的几种用法 ← Yee](https://link.zhihu.com/?target=https%3A//yixinglu.gitlab.io/enable_if.html)
- [C++11新特性--std::enable_if和SFINAE - 简书](https://link.zhihu.com/?target=https%3A//www.jianshu.com/p/a961c35910d2)

## Blackboard

定义在`BehaviorTree.CPP/include/behaviortree_cpp_v3/blackboard.h`，是树中`nodes`传输数据的方式，所有`nodes`共享。
每棵树都有自己的`blackboard`，开发者需要显式的在不同的父树、子树的`blackboard`间创建映射关联。这个操作无需通过代码实现，只需要在`xml`中编辑。
同样，`nodes`执行顺序也是在`xml`中设定的，这使得调试时可以节省大量的程序编译时间。`Blackboard`类中存储数据的容器有3个。

```cpp
private:
    std::unordered_map<std::string, Entry> storage_;  // 保存键值对
    // 指向父blackboard的指针，会与本blackboard有重映射关系
    std::weak_ptr<Blackboard> parent_bb_; // weak_ptr可以避免循环引用
    // 保存映射的内外对应关系
    std::unordered_map<std::string,std::string> internal_to_external_;
```

`Entry`就是`blackboard`保存的1个元素。

```cpp
struct Entry {
        Any value;  // port存储的值
        const PortInfo port_info; // port的其他信息，不含名字和值
}
```

## Port

`Port`是节点间交换数据的机制，通过`Blackboard`的相同`key`联系起来。节点的`ports`的数量、名称、类型等，在编译时就已知了，体现在`xml`文件中。

如果一个节点有输入/输出`port`，必须在`providedPorts()`函数中声明。

获取`port`的值可以用`getInput()`，设置`port`的值可以用`setOutput()`。

```cpp
typedef std::unordered_map<std::string, PortInfo> PortsList;

static PortsList providedPorts();

template <typename T>
Result getInput(const std::string& key, T& destination) const;

template <typename T>
Optional<T> getInput(const std::string& key) const;

template <typename T>
Result setOutput(const std::string& key, const T& value);
```

```cpp
class PortInfo {
private:
    PortDirection _type;
    const std::type_info* _info;
    // 将port输入/输出的string自动转换为指定类型的函数
    StringConverter _converter;
    std::string description_; // port的含义描述
    std::string default_value_; // port没有设置值时的默认值
}
typedef std::function<Any(StringView)> StringConverter;
```

`PortDirection`分为`INPUT`、`OUTPUT`、`INOUT` 3种。开发者最好不要对输入的`port`和数据做任何假设，推荐在`tick()`中周期性的读取输入值，以应对外界变化，
而不是只在构造函数或初始化时只读取一次保存下来。保存历史信息会伤害行为树的`reactive`特性，使得节点的行为不仅与现在情景有关，还与过去有关。

```text
template <typename T>
inline T convertFromString(StringView /*str*/);
```

当在`xml`中通过`SetBlackboard`设置`port`后，代码中运行`getInput()`会后台调用`convertFromString()`，将读入的`string`转换为对应的类型。
`BehaviorTree.CPP/include/behaviortree_cpp_v3/basic_types.h`中实现了`StirngView`向`string`, `const char*`, `int`, `unsigned`, `long`, 
`unsigned long`, `float`, `double`, `vector<int>`, `bool`, `NodeStatus`, `Nodetype`, `PortDirection`等内建类型的转换。如果开发者想转换为自定义类型的话，可以参考`basic_types.cpp`的代码。

树中的`nodes`间互相读写`port`是不会触发`convertFromString()`的，毕竟这是一个`string`到`type`的单向转换。

```cpp
// 转换为枚举类型
template <>
NodeType convertFromString<NodeType>(StringView str) {
  if (str == "Action") return NodeType::ACTION;
  if (str == "Condition") return NodeType::CONDITION;
  if (str == "Control") return NodeType::CONTROL;
  if (str == "Decorator") return NodeType::DECORATOR;
  if (str == "SubTree" || str == "SubTreePlus") return NodeType::SUBTREE;
  return NodeType::UNDEFINED;
}
```

```cpp
struct Position2D {
  double x, y;
};

// 转换为自定义类型
template <>
inline Position2D convertFromString(StringView str) {
  // real numbers separated by semicolons
  auto parts = splitString(str, ';');
  if (parts.size() != 2) {
    throw RuntimeError("invalid input)");
  } else {
    Position2D output;
    output.x = convertFromString<double>(parts[0]);
    output.y = convertFromString<double>(parts[1]);
    return output;
  }
}
```

## 参考链接

- [std::optional - cppreference.com](https://link.zhihu.com/?target=https%3A//en.cppreference.com/w/cpp/utility/optional)
- [std::any - cppreference.com](https://link.zhihu.com/?target=https%3A//en.cppreference.com/w/cpp/utility/any)
- [C++17之std::any_janeqi1987的专栏-CSDN博客_std::any](https://link.zhihu.com/?target=https%3A//blog.csdn.net/janeqi1987/article/details/100568181)













