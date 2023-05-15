###### datetime:2023/05/12 10:30

###### author:nzb

# BT10：Logger类实现原理解析（单例与观察者模式）

上一章中的`StdCoutLogger`、`MinitraceLogger`、`PublisherZMQ`，都只能创建1个实例。像是单例模式，却不是典型的令构造函数是`private`的实现方式，而是通过限定构造函数的执行次数实现的。

不同的`logger`都是监控`node`的状态变化，在恰当的时机执行对应的打印、发送、保存等任务，这是通过观察者模式实现的。

本文从这2个实现方法展开介绍。

## 基类

所有`Logger`都继承自`StatusChangeLogger`，该类定义在 `BehaviorTree.CPP/include/behaviortree_cpp_v3/loggers/abstract_logger.h`
，是一个纯虚基类。

```cpp
// 所有logger的基类，纯虚基类
class StatusChangeLogger {
 public:
  StatusChangeLogger(TreeNode* root_node);
  virtual ~StatusChangeLogger() = default;
  // 当node发生状态变化时要执行的操作
  virtual void callback(BT::Duration timestamp, const TreeNode& node,
                        NodeStatus prev_status, NodeStatus status) = 0;
  // 保存或发送数据
  virtual void flush() = 0;
  ...
 private:
  std::vector<TreeNode::StatusChangeSubscriber> subscribers_;
  ...
};
```

在 `StatusChangeLogger`的构造函数中，遍历树的节点，为其绑定回调函数，即不同`logger`所实现的`callback()`函数，来实现具体的打印、发送、保存等操作。

```cpp
inline StatusChangeLogger::StatusChangeLogger(TreeNode* root_node) {
  first_timestamp_ = std::chrono::high_resolution_clock::now();
  // 对回调函数callback()的封装，执行配置选项对应的callback()
  auto subscribeCallback = [this](TimePoint timestamp, const TreeNode& node,
                                  NodeStatus prev, NodeStatus status) {
    if (enabled_ && (status != NodeStatus::IDLE || show_transition_to_idle_)) {
      if (type_ == TimestampType::ABSOLUTE) {  // 真正的回调操作
        this->callback(timestamp.time_since_epoch(), node, prev, status);
      } else {
        this->callback(timestamp - first_timestamp_, node, prev, status);
      }
    }
  };
  // 增加订阅者，绑定回调函数
  auto visitor = [this, subscribeCallback](TreeNode* node) {
    subscribers_.push_back(
        node->subscribeToStatusChange(std::move(subscribeCallback)));
  };
  // 遍历树的所有节点
  applyRecursiveVisitor(root_node, visitor);
}applyRecursiveVisitor
```

`StatusChangeSignal`应用了观察者模式。

```cpp
using StatusChangeSignal = Signal<TimePoint, const TreeNode&, NodeStatus, NodeStatus>;
/**
 * @brief subscribeToStatusChange is used to attach a callback to a status change.
 * When StatusChangeSubscriber goes out of scope (it is a shared_ptr) the callback
 * is unsubscribed automatically.     
 * @param callback The callback to be execute when status change.
 * @return the subscriber handle.
 */
TreeNode::StatusChangeSubscriber
TreeNode::subscribeToStatusChange(TreeNode::StatusChangeCallback callback) {
    return state_change_signal_.subscribe(std::move(callback));
}

//Call the visitor for each node of the tree, given a root.
void applyRecursiveVisitor(TreeNode* root_node, const std::function<void(TreeNode*)>& visitor);
```

## 子类

以`StdCoutLogger`为例，其他子类仅`callback()`实现的操作不同而已，调用逻辑是一致的

```cpp
class StdCoutLogger : public StatusChangeLogger {
  static std::atomic<bool> ref_count;  // 原子类型
 public:
  StdCoutLogger(const BT::Tree& tree);
  ~StdCoutLogger() override;

  virtual void callback(Duration timestamp, const TreeNode& node,
                        NodeStatus prev_status, NodeStatus status) override;

  virtual void flush() override;
};
```

单例的重点在于利用原子变量`ref_count`
的值变化，来监控调用构造函数的次数。这个语法知识可以参考[链接](https://link.zhihu.com/?target=https%3A//en.cppreference.com/w/cpp/atomic/atomic/compare_exchange)

```cpp
std::atomic<bool> StdCoutLogger::ref_count(false);

StdCoutLogger::StdCoutLogger(const BT::Tree& tree)
    : StatusChangeLogger(tree.rootNode()) {
  bool expected = false;
  // 如果expected==ref_count，就令ref_count=true（第2个参数），并返回true；
  // 否则，将ref_count的值赋给expected，并返回false
  // 即，ref_count初始化为false。第1次执行构造函数时，ref_count会被置为true，并返回true
  // 后面再进入构造函数时，expected!=ref_count，会抛出异常。
  if (!ref_count.compare_exchange_strong(expected, true)) {
    throw LogicError("Only one instance of StdCoutLogger shall be created");
  }
}
StdCoutLogger::~StdCoutLogger() { ref_count.store(false); }
```

## 观察者模式

`TreeNode`的状态变化，都要经由`setStatus()`函数实现。在该函数内，会在`NodeStatus`变化时，通知订阅了该消息的订阅者。

```cpp
void TreeNode::setStatus(NodeStatus new_status) {
  NodeStatus prev_status;
  {
    std::unique_lock<std::mutex> UniqueLock(state_mutex_);
    prev_status = status_;
    status_ = new_status;
  }
  if (prev_status != new_status) {
    // 当状态变化时，条件变量通知其他等待者
    state_condition_variable_.notify_all();
    // 通知该Signal的订阅者
    state_change_signal_.notify(std::chrono::high_resolution_clock::now(),
                                *this, prev_status, new_status);
  }
}
```

`notify() `通知到位的同时，会执行订阅者绑定的`callback()`。

```cpp
template <typename... CallableArgs>
class Signal {
 public:
  using CallableFunction = std::function<void(CallableArgs...)>;
  using Subscriber = std::shared_ptr<CallableFunction>;

  void notify(CallableArgs... args) {
    for (size_t i = 0; i < subscribers_.size();) {
      if (auto sub = subscribers_[i].lock()) {
        (*sub)(args...);  // 执行callback
        i++;
      } else {
        subscribers_.erase(subscribers_.begin() + i);
      }
    }
  }
  // 添加一个订阅者
  Subscriber subscribe(CallableFunction func) {
    Subscriber sub = std::make_shared<CallableFunction>(std::move(func));
    subscribers_.emplace_back(sub);
    return sub;
  }

 private:
  std::vector<std::weak_ptr<CallableFunction>> subscribers_;
};
```













