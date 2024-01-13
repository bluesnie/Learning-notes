###### datetime:2023/05/15 10:05

###### author:nzb

# BT12：从xml创建加载行为树的过程分析

本文主要分析`BehaviorTree.CPP/src/xml_parsing.cpp`
的内容，因为函数代码都很长，就省略了代码，可以与源文件对照理解，最好在各个阶段多加log单步调试。搞清楚行为树的解析、加载、构建过程， 有利于对其设计思路有更深刻的理解，但是对行为树的使用影响不大，可以跳过。

我认为行为树的精华在于`blackboard`的设计，实现了节点间、树间的数据共享，但代码层次较深，理解花费时间较多，待以后补充。

## 1、BehaviorTreeFactory::createTreeFromText()

树的加载和创建由`createTreeFromText()` 实现，该函数的第2个参数具有默认参数，即初创建的`blackboard`，是一个局部变量，但是由智能指针指向它。 因此，只要引用计数大于0，该变量仍然不会释放，可以访问得到。

```cpp
Tree createTreeFromText(const std::string& text,
                        Blackboard::Ptr blackboard = Blackboard::create());

Tree BehaviorTreeFactory::createTreeFromText(const std::string& text,
                                             Blackboard::Ptr blackboard) {
  XMLParser parser(*this);
  // 加载和解析文本，检查各项元素是否符合BT的概念要求。
  parser.loadFromText(text);
  // 创建树和所有节点的实例，构造好树之间、节点之间的父子关系，port的映射关系等。
  auto tree = parser.instantiateTree(blackboard);
  // 将树的节点信息绑定给树实例变量
  tree.manifests = this->manifests();
  return tree;
}
```

`createTreeFromText()` 主要有3部分。其中的`manifests`包含了树的所有节点类型信息，其实节点的`builder`和`manifest`在树建立之前已经通过`register`函数传给`factory`变量了。

```cpp
template <typename T>
void registerNodeType(const std::string& ID, PortsList ports) {
  ...
  registerBuilder(CreateManifest<T>(ID, ports), CreateBuilder<T>());
}

void BehaviorTreeFactory::registerBuilder(const TreeNodeManifest& manifest,
                                          const NodeBuilder& builder) {
  auto it = builders_.find(manifest.registration_ID);
  if (it != builders_.end()) {
    throw BehaviorTreeException("ID [", manifest.registration_ID,
                                "] already registered");
  }
  builders_.insert({manifest.registration_ID, builder});
  manifests_.insert({manifest.registration_ID, manifest});
}
```

## 2、XMLParser::loadFromText()

具体由`XMLParser::Pimpl::loadDocImpl()`执行，主要有如下几个步骤。

- 第1个for循环，递归加载本`xml`中所`include`的子树`xml`文件，先加载子树，再加载外层树，相当于深度优先搜索。
- 第2个for循环，遍历本`xml`文件中的树的名称或`ID`（相当于树的根节点），保存在类`XMLParser::Pimpl`的成员变量`tree_roots`中。
- 第3、4个for循环，将构造树之前就注册的所有节点，和2中读取的树的根节点，都存入局部变量`std::set<std::string> registered_nodes`; 然后将其传入`VerifyXML()`。
- `VerifyXML()`负责检查树的设计要求是否满足。检查项有：
    - `TreeNodesModel`标签是否合法，主要用于`Groot`可视化。
    - 各种`node`的子节点数量是否合法，是否有`ID`。比如`ControlNode`至少有1个子节点，`DecoratorNode`只有1个子节点，`Subtree`没有子节点。
    - 是否有未注册的不认识的节点。
    - 针对非`subtree`节点进行递归检查。
    - 是否指定`main_tree_to_execute` 标签。如果有多个`BehaviorTree`，则必须指定`main_tree_to_execute`，如果只有1个`BehaviorTree`，就不需要指定。

## 3、XMLParser::instantiateTree()

分为2个部分。

- 构造了行为树的实例——局部变量`output_tree`，将传入的`blackboard`（即上文创建的智能指针指向的`blackboard`）保存入 `output_tree.blackboard_stack`。
- 调用`recursivelyCreateTree()`，传入主树（最外层树）的`ID`、`tree`局部变量、`blackboard`（还是刚才同一个智能指针）、`TreeNode`指针（空指针`nullptr`，作为根节点）。

```cpp
Tree XMLParser::instantiateTree(const Blackboard::Ptr& root_blackboard) {
  Tree output_tree;
  ...
  // first blackboard
  output_tree.blackboard_stack.push_back(root_blackboard);
  _p->recursivelyCreateTree(main_tree_ID, output_tree, root_blackboard,
                            TreeNode::Ptr());
  return output_tree;
}
```

接下来对`recursivelyCreateTree()`展开分析。

## 4、XMLParser::Pimpl::recursivelyCreateTree()

函数内递归执行`recursiveStep()`，注意第1个参数是父节点。

```cpp
void BT::XMLParser::Pimpl::recursivelyCreateTree(
    const std::string& tree_ID, Tree& output_tree, Blackboard::Ptr blackboard,
    const TreeNode::Ptr& root_parent) {
  std::function<void(const TreeNode::Ptr&, const XMLElement*)> recursiveStep;
  recursiveStep = [&](const TreeNode::Ptr& parent, const XMLElement* element) {
    ...
  };
  auto root_element = tree_roots[tree_ID]->FirstChildElement();
  // start recursion
  recursiveStep(root_parent, root_element);
}
```

`recursiveStep()`分为3部分。

- 调用`XMLParser::Pimpl::createNodeFromXML()`创建节点实例，将该实例保存在树的`std::vector<TreeNode::Ptr> nodes` 成员变量中。
- 如果该节点是`SUBTREE`类型的，细分`SubtreeNode`和`SubtreePlusNode`来处理。
    - 如果是`SubtreeNode`，就根据`__shared_blackboard`的值来创建`blackboard`，并添加映射信息，然后递归调用`recursivelyCreateTree()`来创建子树。
    - 如果是`SubtreePlusNode`，就根据`__autoremap`的值来创建`blackboard`的`port`的映射，然后递归调用`recursivelyCreateTree()`来创建子树。
- 如果该节点不是`SUBTREE`类型的，递归调用`recursiveStep()`，并把该节点作为接下来待创建节点的父节点。如果该节点没有其他包含的元素了，就不再递归了， 从`recursiveStep()`
  返回，进而从`recursivelyCreateTree()`返回，进而从`instantiateTree()` 返回。

## 5、createNodeFromXML()

- 对非`subtree`的节点，将`port`映射的`key`和`value`保存入局部变量`PortsRemapping port_remap`。
- 对于有`remap`的节点，在`blackboard`中通过`Blackboard::setPortInfo()` 添加`port`映射信息， 并在父树的`blackboard`的相同`key`也保存相同`port`
  信息。基于此，实现了父子树之间的`blackboard`对相同`key`的同一性关联。
- 使用`manifest`中保存的信息，初始化`NodeConfiguration`。即在`NodeConfiguration`的`input_ports`和`output_ports`集合中添加存在外部映射的`port`。
- 对于不存在外部映射的`port`，对其中的`InputPort`赋默认值，并存`入NodeConfiguration`的`input_ports`集合中。
- `config`构造完成，调用 `instantiateTreeNode()` 来实例化子节点。
- 若传入的父节点有效，根据父节点的类型，为其添加子节点。

