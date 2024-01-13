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

## PublisherZMQ JSON版本

- `bt_zmq_pub_json.h`

```c++
#ifndef _BT_ZMQ_PUB_JSON_H_
#define _BT_ZMQ_PUB_JSON_H_

#include "map"
#include "cppzmq/zmq.hpp"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "rapidjson/document.h"

typedef rapidjson::Document JSON;
typedef rapidjson::Value JValue;

class PublisherJsonZMQ : public BT::StatusChangeLogger
{
    static std::atomic<bool> ref_cnt;

public:
    PublisherJsonZMQ(const BT::Tree &tree, unsigned max_msg_per_second = 25, unsigned publisher_port = 1666, unsigned server_port = 1667);
    virtual ~PublisherJsonZMQ();

private:
    virtual void callback(BT::Duration timestamp, const BT::TreeNode &node, BT::NodeStatus prev_status, BT::NodeStatus status) override;
    virtual void flush() override;

    void CreateJsonBehaviorTree();
    void createStatusData();

    const BT::Tree &tree_;          // 树指针引用
    JSON tree_data_;                // 树数据
    JSON status_transition_data_;   // 状态和变更数据
    JValue status_data_;            // 当前树状态数据
    JValue transition_data_;        // 状态变更数据
    std::chrono::microseconds min_time_between_msgs_;

    std::atomic_bool active_server_;
    std::thread thread_;

    std::mutex mutex_;
    std::atomic_bool send_pending_;
    std::condition_variable send_condition_variable_;
    std::future<void> send_future_;

    struct Pimpl;
    Pimpl *zmq_;
};

#endif
```

- `bt_zmq_pub_json.cpp`

```c++
#include "bt_zmq_pub_json.h"

const char *json_to_str(rapidjson::Document &json, rapidjson::StringBuffer &buffer)
{
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    json.Accept(writer);
    auto str_msg = buffer.GetString();
    return str_msg;
}

std::atomic<bool> PublisherJsonZMQ::ref_cnt(false);

struct PublisherJsonZMQ::Pimpl
{
    Pimpl() : ctx(1), publisher(ctx, ZMQ_PUB), server(ctx, ZMQ_REP) {}

    zmq::context_t ctx;
    zmq::socket_t publisher;
    zmq::socket_t server;
};

PublisherJsonZMQ::PublisherJsonZMQ(const BT::Tree &tree, unsigned max_msg_per_second, unsigned publisher_port, unsigned server_port) : StatusChangeLogger(tree.rootNode()),
                                                                                                                                       tree_(tree),
                                                                                                                                       min_time_between_msgs_(std::chrono::microseconds(1000 * 1000) / max_msg_per_second),
                                                                                                                                       send_pending_(false),
                                                                                                                                       zmq_(new Pimpl()),
                                                                                                                                       status_data_(rapidjson::kArrayType),
                                                                                                                                       transition_data_(rapidjson::kArrayType)
{
    bool expected = false;
    if (!ref_cnt.compare_exchange_strong(expected, true))
    {
        throw BT::LogicError("Only one instace of PublisherJsonZMQ shall be created");
    }
    if (publisher_port == server_port)
    {
        throw BT::LogicError("The TCP ports of the publisher and the server must be different");
    }
    // 创建json树
    CreateJsonBehaviorTree();

    char str[100];
    sprintf(str, "tcp://*:%d", publisher_port);
    zmq_->publisher.bind(str);                      // 状态发布话题 pub
    sprintf(str, "tcp://*:%d", server_port);
    zmq_->server.bind(str);                         // 树数据rep

    int timeout_ms = 100;
    zmq_->server.set(zmq::sockopt::rcvtimeo, timeout_ms);

    active_server_ = true;
    // 启线程接收 req 请求
    thread_ = std::thread([this]()
                          {
    while (active_server_)
    {
      zmq::message_t req;
      try
      {
        zmq::recv_result_t received = zmq_->server.recv(req);
        if (received)
        {
        // to string
          rapidjson::StringBuffer buffer;
          auto tree_str = json_to_str(tree_data_, buffer);
          SPDLOG_DEBUG("behavior_tree-> {}", tree_str);
          zmq::message_t reply(strlen(tree_str));
          memcpy(reply.data(), tree_str, strlen(tree_str));
          // send data
          zmq_->server.send(reply, zmq::send_flags::none);
        }
      }
      catch (zmq::error_t& err)
      {
        if (err.num() == ETERM)
        {
          SPDLOG_INFO("[PublisherZMQ] Server quitting.");
        }
        SPDLOG_ERROR("[PublisherZMQ] just died. Exception {} ",  err.what());
        active_server_ = false;
      }
    } });
}

PublisherJsonZMQ::~PublisherJsonZMQ()
{
    active_server_ = false;
    if (thread_.joinable())
    {
        thread_.join();
    }
    if (send_pending_)
    {
        send_condition_variable_.notify_all();
        send_future_.get();
    }
    flush();
    zmq_->ctx.shutdown();
    delete zmq_;
    ref_cnt = false;
}

void PublisherJsonZMQ::CreateJsonBehaviorTree()
{
    rapidjson::Document::AllocatorType &allocator = tree_data_.GetAllocator();
    tree_data_.SetObject();
    // 当前所在树节点
    JValue tree_nodes(rapidjson::kArrayType);
    BT::applyRecursiveVisitor(tree_.rootNode(), [&](BT::TreeNode *node)
                              {
                                  // 子节点
                                  JValue children_uid(rapidjson::kArrayType);
                                  // 控制节点
                                  if (auto control = dynamic_cast<BT::ControlNode *>(node))
                                  {
                                      for (const auto &child : control->children())
                                      {
                                          children_uid.PushBack(child->UID(), allocator);
                                      }
                                  }
                                  // 装饰节点
                                  else if (auto decorator = dynamic_cast<BT::DecoratorNode *>(node))
                                  {
                                      const auto &child = decorator->child();
                                      children_uid.PushBack(child->UID(), allocator);
                                  }
                                  // 节点输入端口
                                  JValue ports(rapidjson::kArrayType);
                                  for (const auto &it : node->config().input_ports)
                                  {
                                      JValue port_obj(rapidjson::kObjectType);
                                      JValue key(it.first.c_str(), allocator);
                                      JValue val(it.second.c_str(), allocator);
                                      port_obj.AddMember(key, val, allocator);
                                      ports.PushBack(port_obj, allocator);
                                  }
                                  // 节点输出端口
                                  for (const auto &it : node->config().output_ports)
                                  {
                                      JValue port_obj(rapidjson::kObjectType);
                                      JValue key(it.first.c_str(), allocator);
                                      JValue val(it.second.c_str(), allocator);
                                      port_obj.AddMember(key, val, allocator);
                                      ports.PushBack(port_obj, allocator);
                                  }
                                  // 树节点
                                  JValue tn(rapidjson::kObjectType);
                                  tn.AddMember("uid", node->UID(), allocator);
                                  tn.AddMember("children_uid", children_uid, allocator);
                                  tn.AddMember("status", static_cast<int8_t>(node->status()), allocator);
                                  JValue name(node->name().c_str(), allocator);
                                  JValue registration_name(node->registrationName().c_str(), allocator);
                                  tn.AddMember("name", name, allocator);
                                  tn.AddMember("registration_name", registration_name, allocator);
                                  tn.AddMember("ports", ports, allocator);
                                  tree_nodes.PushBack(tn, allocator); });
    // 所有节点模型
    JValue node_models(rapidjson::kArrayType);
    
    for (const auto &node_it : tree_.manifests)
    {
        const auto &manifest = node_it.second;

        JValue port_models(rapidjson::kArrayType);
        // 节点端口
        for (const auto &port_it : manifest.ports)
        {
            const auto &port_name = port_it.first;
            const auto &port = port_it.second;

            JValue port_model(rapidjson::kObjectType);
            JValue port_name_val(port_name.c_str(), allocator);
            port_model.AddMember("port_name", port_name_val, allocator);
            port_model.AddMember("direction", static_cast<int8_t>(port.direction()), allocator);
            JValue type_info(BT::demangle(port.type()).c_str(), allocator);
            port_model.AddMember("type_info", type_info, allocator);
            JValue description(port.description().c_str(), allocator);
            port_model.AddMember("description", description, allocator);

            port_models.PushBack(port_model, allocator);
        }
        // 节点信息
        JValue node_model(rapidjson::kObjectType);
        JValue registration_id(manifest.registration_ID.c_str(), allocator);
        node_model.AddMember("registration_id", registration_id, allocator);
        node_model.AddMember("node_type", static_cast<int8_t>(manifest.type), allocator);
        node_model.AddMember("port_models", port_models, allocator);

        node_models.PushBack(node_model, allocator);
    }
    // 树
    tree_data_.AddMember("uid", tree_.rootNode()->UID(), allocator);
    tree_data_.AddMember("tree_nodes", tree_nodes, allocator);
    tree_data_.AddMember("node_models", node_models, allocator);
}

// 更新状态
void PublisherJsonZMQ::createStatusData()
{
    status_data_.SetArray();
    rapidjson::Document::AllocatorType &allocator = status_transition_data_.GetAllocator();
    // 递归获取当前树所有节点状态
    BT::applyRecursiveVisitor(tree_.rootNode(), [&](BT::TreeNode *node)
                              {
        JSON tmp(rapidjson::kObjectType);
        tmp.AddMember("uid", node->UID(), allocator); 
        tmp.AddMember("status", static_cast<int8_t>(node->status()), allocator); 
        status_data_.PushBack(tmp, allocator); });
}

void PublisherJsonZMQ::callback(BT::Duration timestamp, const BT::TreeNode &node, BT::NodeStatus prev_status, BT::NodeStatus status)
{
    rapidjson::Document::AllocatorType &allocator = status_transition_data_.GetAllocator();
    JValue transition(rapidjson::kObjectType);
    int64_t usec = std::chrono::duration_cast<std::chrono::microseconds>(timestamp).count();
    transition.AddMember("uid", node.UID(), allocator);
    transition.AddMember("prev_status", static_cast<int8_t>(prev_status), allocator);
    transition.AddMember("status", static_cast<int8_t>(status), allocator);
    transition.AddMember("t_sec", usec / 1000000, allocator);
    transition.AddMember("t_usec", usec % 1000000, allocator);

    {
        std::unique_lock<std::mutex> lock(mutex_);
        transition_data_.PushBack(transition, allocator);
    }

    if (!send_pending_.exchange(true))
    {
        send_future_ = std::async(std::launch::async, [this]()
                                  {
      std::unique_lock<std::mutex> lock(mutex_);
      const bool is_server_inactive = send_condition_variable_.wait_for(
          lock, min_time_between_msgs_, [this]() { return !active_server_; });
      lock.unlock();
      if (!is_server_inactive)
      {
        flush();
      } });
    }
}

void PublisherJsonZMQ::flush()
{
    zmq::message_t message;
    {
        {
            std::unique_lock<std::mutex> lock(mutex_);

            status_transition_data_.SetObject();
            rapidjson::Document::AllocatorType &allocator = status_transition_data_.GetAllocator();
            status_transition_data_.AddMember("status", status_data_, allocator);
            status_transition_data_.AddMember("transition", transition_data_, allocator);
            // to string
            rapidjson::StringBuffer buffer;
            auto status_transition_str = json_to_str(status_transition_data_, buffer);
            message.rebuild(strlen(status_transition_str));
            memcpy(message.data(), status_transition_str, strlen(status_transition_str));
            SPDLOG_DEBUG("pub status and transition-> {}", status_transition_str);

            transition_data_.SetArray();
            createStatusData();
        }
        try
        {
            zmq_->publisher.send(message, zmq::send_flags::none);
        }
        catch (zmq::error_t &err)
        {
            if (err.num() == ETERM)
            {
                SPDLOG_INFO("[PublisherZMQ] Publisher quitting.");
            }
            SPDLOG_ERROR("[PublisherZMQ] just died. Exception {} ", err.what());
        }

        send_pending_ = false;
    }
}
```

## Groot 数据使用源码解析

- 源码路径：`Groot/bt_editor/sidepanel_monitor.cpp`

```c++
// 订阅状态变更
void SidepanelMonitor::on_timer()
{
    if( !_connected ) return;

    zmq::message_t msg;
    try{
        while(  _zmq_subscriber.recv(msg) )
        {
            _msg_count++;
            ui->labelCount->setText( QString("Messages received: %1").arg(_msg_count) );

            const char* buffer = reinterpret_cast<const char*>(msg.data());
            // status数据长度
            const uint32_t header_size = flatbuffers::ReadScalar<uint32_t>( buffer );
            // transitions 数组长度
            const uint32_t num_transitions = flatbuffers::ReadScalar<uint32_t>( &buffer[4+header_size] );

            std::vector<std::pair<int, NodeStatus>> node_status;
            // check uid in the index, if failed load tree from server
            try{
                for(size_t offset = 4; offset < header_size +4; offset +=3 )
                {
                    const uint16_t uid = flatbuffers::ReadScalar<uint16_t>(&buffer[offset]);
                    _uid_to_index.at(uid);  // 如果不存在就报错，树变了，重新请求树数据
                }

                for(size_t t=0; t < num_transitions; t++)
                {
                    size_t offset = 8 + header_size + 12*t;
                    const uint16_t uid = flatbuffers::ReadScalar<uint16_t>(&buffer[offset+8]);
                    _uid_to_index.at(uid);
                }
                // 更新节点状态
                for(size_t offset = 4; offset < header_size +4; offset +=3 )
                {
                    const uint16_t uid = flatbuffers::ReadScalar<uint16_t>(&buffer[offset]);
                    const uint16_t index = _uid_to_index.at(uid);
                    AbstractTreeNode* node = _loaded_tree.node( index );
                    node->status = convert(flatbuffers::ReadScalar<Serialization::NodeStatus>(&buffer[offset+2] ));
                }

                //qDebug() << "--------";
                // 更新节点流转状态
                for(size_t t=0; t < num_transitions; t++)
                {
                    size_t offset = 8 + header_size + 12*t;
                    const uint16_t uid = flatbuffers::ReadScalar<uint16_t>(&buffer[offset+8]);
                    const uint16_t index = _uid_to_index.at(uid);
                    NodeStatus status  = convert(flatbuffers::ReadScalar<Serialization::NodeStatus>(&buffer[offset+11] ));

                    _loaded_tree.node(index)->status = status;
                    node_status.push_back( {index, status} );

                }
            }
            catch( std::out_of_range& err) {
                qDebug() << "Reload tree from server";
                if( !getTreeFromServer() ) {
                    _connected = false;
                    ui->lineEdit_address->setDisabled(false);
                    _timer->stop();
                    connectionUpdate(false);
                    return;
                }
            }

            // update the graphic part
            emit changeNodeStyle( "BehaviorTree", node_status );

            // lock editing of nodes
            auto main_win = dynamic_cast<MainWindow*>( _parent );
            main_win->lockEditing(true);
        }
    }
    catch( zmq::error_t& err)
    {
        qDebug() << "ZMQ receive failed: " << err.what();
    }
}

// 获取树信息
bool SidepanelMonitor::getTreeFromServer()
{
    try{
        zmq::message_t request(0);
        zmq::message_t reply;

        zmq::socket_t  zmq_client( _zmq_context, ZMQ_REQ );
        zmq_client.connect( _connection_address_req.c_str() );

        zmq_client.setsockopt(ZMQ_RCVTIMEO, &_load_tree_timeout_ms, sizeof(int) );

        zmq_client.send(request, zmq::send_flags::none);

        auto bytes_received  = zmq_client.recv(reply, zmq::recv_flags::none);
        if( !bytes_received || *bytes_received == 0 )
        {
            return false;
        }

        const char* buffer = reinterpret_cast<const char*>(reply.data());
        auto fb_behavior_tree = Serialization::GetBehaviorTree( buffer );

        auto res_pair = BuildTreeFromFlatbuffers( fb_behavior_tree );

        _loaded_tree  = std::move( res_pair.first );
        _uid_to_index = std::move( res_pair.second );  // 键为uid，值为 _loaded_tree的nodes属性里面对应节点的索引

        // add new models to registry
        for(const auto& tree_node: _loaded_tree.nodes())
        {
            const auto& registration_ID = tree_node.model.registration_ID;
            if( BuiltinNodeModels().count(registration_ID) == 0)
            {
                addNewModel( tree_node.model );
            }
        }

        try {
            loadBehaviorTree( _loaded_tree, "BehaviorTree" );
        }
        catch (std::exception& err) {
            QMessageBox messageBox;
            messageBox.critical(this,"Error Connecting to remote server", err.what() );
            messageBox.show();
            return false;
        }

        std::vector<std::pair<int, NodeStatus>> node_status;
        node_status.reserve(_loaded_tree.nodesCount());

        //  qDebug() << "--------";

        for(size_t t=0; t < _loaded_tree.nodesCount(); t++)
        {
            node_status.push_back( { t, _loaded_tree.nodes()[t].status } );
        }
        emit changeNodeStyle( "BehaviorTree", node_status );
    }
    catch( zmq::error_t& err)
    {
        qDebug() << "ZMQ client receive failed: " << err.what();
        return false;
    }
    return true;
}
```