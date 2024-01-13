###### datetime:2023/01/19 15:41

###### author:nzb

# 2、roslaunch

roslaunch 主要功能用于启动节点，rosmaster也是使用该模块启动

## 2.1、roslaunch 脚本

对应的可执行脚本路径是`ros/melodic/bin/roslaunch`, 这是一个python脚本

```python
import roslaunch

roslaunch.main()
```

## 2.2、节点启动流程

### 2.2.1、 创建`ROSLaunchParent`

源码路径：`ros\melodic\lib\python2.7\dist-packages\roslaunch\__init__.py`

```python
def main(argv=sys.argv):
    # ....
    logger = logging.getLogger('roslaunch')
    # 对应日志：[roslaunch][INFO] 2023-01-16 14:40:50,171: roslaunch starting with args ['/opt/ros/melodic/bin/roslaunch', '/set_urdf.launch', '--wait']
    # args是上面roscore传进来的参数
    logger.info("roslaunch starting with args %s" % str(argv))
    # 对应日志：[roslaunch][INFO] 2023-01-18 14:57:35,628: roslaunch env is {'ROS_DISTRO': 'melodic', 
    # 'ROS_IP': '192.168.111.111', 'HOME': '/root', 'PATH': '/opt/ros/melodic/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
    # 'ROS_PACKAGE_PATH': '/opt/ros/melodic/share', 'CMAKE_PREFIX_PATH': '/opt/ros/melodic', 'LD_LIBRARY_PATH': '/opt/ros/melodic/lib', 
    # 'LANG': 'C.UTF-8', 'TERM': 'xterm', 'SHLVL': '1', 'ROS_LOG_FILENAME': '/root/.ros/log/769ca6bc-96d1-11ed-8514-8ec2aee29851/roslaunch-quicktron-RK-1753.log',
    # 'ROS_MASTER_URI': 'http://192.168.111.111:11311', 'ROS_PYTHON_VERSION': '2', 'PYTHONPATH': '/opt/ros/melodic/lib/python2.7/dist-packages',
    # 'ROS_ROOT': '/opt/ros/melodic/share/ros', 'PKG_CONFIG_PATH': '/opt/ros/melodic/lib/pkgconfig', 'LC_ALL': 'C.UTF-8', '_': '/opt/ros/melodic/bin/roscore', 
    # 'HOSTNAME': 'quicktron-RK', 'ROSLISP_PACKAGE_DIRECTORIES': '', 'PWD': '/', 'ROS_ETC_DIR': '/opt/ros/melodic/etc/ros', 'ROS_VERSION': '1',...}
    logger.info("roslaunch env is %s" % os.environ)

    if options.child_name:
        logger.info('starting in child mode')
        # ...
    else:
        logger.info('starting in server mode')
        ...
        # This is a roslaunch parent, spin up parent server and launch processes.
        # args are the roslaunch files to load
        from . import parent as roslaunch_parent
        # force a port binding spec if we are running a core
        if options.core:
            options.port = options.port or DEFAULT_MASTER_PORT
        p = roslaunch_parent.ROSLaunchParent(uuid, args, roslaunch_strs=roslaunch_strs,
                                             is_core=options.core, port=options.port, local_only=options.local_only,
                                             verbose=options.verbose, force_screen=options.force_screen,
                                             force_log=options.force_log,
                                             num_workers=options.num_workers, timeout=options.timeout,
                                             master_logger_level=options.master_logger_level,
                                             show_summary=not options.no_summary,
                                             force_required=options.force_required)
        p.start()
        p.spin()
```

`set_urdf.launch`文件

```text
<launch>
  <param name="robot_description" command="cat /config/robot_description.urdf" />
  <!-- 运行joint_state_publisher节点，发布机器人的关节状态  -->
  <node name="joint_state_publisher" pkg="joint_state_publisher" type="joint_state_publisher" ></node>

  <!-- 运行robot_state_publisher节点，发布tf  -->
  <node name="robot_state_publisher" pkg="robot_state_publisher" type="robot_state_publisher"  output="screen" >
      <param name="publish_frequency" type="double" value="50.0" />
  </node>
</launch>
```

### 2.2.2、joint_state_publisher启动流程

- 启动相关日志

```text
[roslaunch][INFO] 2023-01-16 14:40:51,168: launch_nodes: launching local nodes ...
[roslaunch][INFO] 2023-01-16 14:40:51,168: ... preparing to launch node of type [joint_state_publisher/joint_state_publisher]
[roslaunch][INFO] 2023-01-16 14:40:51,169: create_node_process: package[joint_state_publisher] type[joint_state_publisher] machine[Machine(name[] env_loader[None] address[localhost] ssh_port[22] user[None] assignable[True] timeout[10.0])] master_uri[http://192.168.111.111:11311]
[roslaunch][INFO] 2023-01-16 14:40:51,169: process[joint_state_publisher-1]: env[{'ROS_DISTRO': 'melodic', 'ROS_IP': '192.168.111.111', 'ROS_PACKAGE_PATH': '/opt/ros/melodic/share', 'PATH': '/opt/ros/melodic/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin', 'CMAKE_PREFIX_PATH': '/opt/ros/melodic', 'ROS_LOG_FILENAME': '/root/.ros/log/b598afc4-9568-11ed-b4d9-00044bde2742/roslaunch-tegra-ubuntu-30.log', 'LANG': 'C.UTF-8', 'TERM': 'xterm', 'SHLVL': '1', 'LD_LIBRARY_PATH': '/opt/ros/melodic/lib', 'ROS_MASTER_URI': 'http://192.168.111.111:11311', 'HOME': '/root', 'ROS_PYTHON_VERSION': '2', 'PYTHONPATH': '/opt/ros/melodic/lib/python2.7/dist-packages', 'ROS_ROOT': '/opt/ros/melodic/share/ros', 'PKG_CONFIG_PATH': '/opt/ros/melodic/lib/pkgconfig', 'LC_ALL': 'C.UTF-8', '_': '/usr/bin/nohup', 'HOSTNAME': 'tegra-ubuntu', 'ROSLISP_PACKAGE_DIRECTORIES': '', 'PWD': '/', 'ROS_ETC_DIR': '/opt/ros/melodic/etc/ros', 'ROS_VERSION': '1'}]
[roslaunch][INFO] 2023-01-16 14:40:51,203: process[joint_state_publisher-1]: args[[u'/opt/ros/melodic/lib/joint_state_publisher/joint_state_publisher', u'__name:=joint_state_publisher']]
[roslaunch][INFO] 2023-01-16 14:40:51,203: ... created process [joint_state_publisher-1]
[roslaunch.pmon][INFO] 2023-01-16 14:40:51,204: ProcessMonitor.register[joint_state_publisher-1]
[roslaunch.pmon][INFO] 2023-01-16 14:40:51,204: ProcessMonitor.register[joint_state_publisher-1] complete
[roslaunch][INFO] 2023-01-16 14:40:51,204: ... registered process [joint_state_publisher-1]
[roslaunch][INFO] 2023-01-16 14:40:51,205: process[joint_state_publisher-1]: starting os process
[roslaunch][INFO] 2023-01-16 14:40:51,205: process[joint_state_publisher-1]: start w/ args [[u'/opt/ros/melodic/lib/joint_state_publisher/joint_state_publisher', u'__name:=joint_state_publisher', u'__log:=/root/.ros/log/b598afc4-9568-11ed-b4d9-00044bde2742/joint_state_publisher-1.log']]
[roslaunch][INFO] 2023-01-16 14:40:51,205: process[joint_state_publisher-1]: cwd will be [/root/.ros]
[roslaunch][INFO] 2023-01-16 14:40:51,957: process[joint_state_publisher-1]: started with pid [78]
[roslaunch][INFO] 2023-01-16 14:40:51,958: ... successfully launched [joint_state_publisher-1]
[roslaunch][INFO] 2023-01-16 14:40:51,958: ... preparing to launch node of type [robot_state_publisher/robot_state_publisher]
......robot_state_publisher节点启动相关日志
[roslaunch][INFO] 2023-01-16 14:40:52,696: ... launch_nodes complete
```

- 前面相关启动跟rosmaster一样，实例化配置、进程管理、XML-RPC等等

```python
class ROSLaunchParent(object):
    def start(self, auto_terminate=True):
        # self._init_runner()
        self.runner = roslaunch.launch.ROSLaunchRunner(self.run_id, self.config, server_uri=self.server.uri,
                                                       pmon=self.pm, is_core=self.is_core,
                                                       remote_runner=self.remote_runner, is_rostest=self.is_rostest,
                                                       num_workers=self.num_workers, timeout=self.timeout,
                                                       master_logger_level=self.master_logger_level)

        self.runner.launch()
        # joint_state_publisher启动流程 启动流程
        # -> self._launch_nodes()
        # -> self.launch_node(node)  
        # -> p = create_node_process()  # 创建启动进程，p为LocalProcess实例
        # -> self.pm.register(p) 提交给进程管理
        # -> p.start()
        # -> subprocess.Popen()  # 执行joint_state_publisher脚本，脚本路径：/opt/ros/melodic/lib/joint_state_publisher/joint_state_publisher
```

```python
def launch_node(self, node, core=False):
    """
    Launch a single node locally. Remote launching is handled separately by the remote module.
    If node name is not assigned, one will be created for it.
    
    @param node Node: node to launch
    @param core bool: if True, core node
    @return obj, bool: Process handle, successful launch. If success, return actual Process instance. Otherwise return name.
    """
    self.logger.info("... preparing to launch node of type [%s/%s]", node.package, node.type)

    # TODO: should this always override, per spec?. I added this
    # so that this api can be called w/o having to go through an
    # extra assign machines step.
    if node.machine is None:
        node.machine = self.config.machines['']
    if node.name is None:
        node.name = rosgraph.names.anonymous_name(node.type)

    master = self.config.master
    import roslaunch.node_args
    try:
        process = create_node_process(self.run_id, node, master.uri)
    except roslaunch.node_args.NodeParamsException as e:
        self.logger.error(e)
        printerrlog("ERROR: cannot launch node of type [%s/%s]: %s" % (node.package, node.type, str(e)))
        if node.name:
            return node.name, False
        else:
            return "%s/%s" % (node.package, node.type), False

    self.logger.info("... created process [%s]", process.name)
    if core:
        self.pm.register_core_proc(process)
    else:
        self.pm.register(process)
    node.process_name = process.name  # store this in the node object for easy reference
    self.logger.info("... registered process [%s]", process.name)

    # note: this may raise FatalProcessLaunch, which aborts the entire launch
    success = process.start()
    if not success:
        if node.machine.name:
            printerrlog("launch of %s/%s on %s failed" % (node.package, node.type, node.machine.name))
        else:
            printerrlog("local launch of %s/%s failed" % (node.package, node.type))
    else:
        self.logger.info("... successfully launched [%s]", process.name)
    return process, success
```

```python
def create_node_process(run_id, node, master_uri):
    """
    Factory for generating processes for launching local ROS
    nodes. Also registers the process with the L{ProcessMonitor} so that
    events can be generated when the process dies.
    
    @param run_id: run_id of launch
    @type  run_id: str
    @param node: node to launch. Node name must be assigned.
    @type  node: L{Node}
    @param master_uri: API URI for master node
    @type  master_uri: str
    @return: local process instance
    @rtype: L{LocalProcess}
    @raise NodeParamsException: If the node's parameters are improperly specific
    """
    # [roslaunch][INFO] 2023-01-16 14:40:51,169: create_node_process: package[joint_state_publisher] 
    # type[joint_state_publisher] machine[Machine(name[] env_loader[None] address[localhost] ssh_port[22] user[None] assignable[True] timeout[10.0])] master_uri[http://192.168.111.111:11311]
    _logger.info("create_node_process: package[%s] type[%s] machine[%s] master_uri[%s]", node.package, node.type,
                 node.machine, master_uri)
    # check input args
    machine = node.machine
    if machine is None:
        raise RLException("Internal error: no machine selected for node of type [%s/%s]" % (node.package, node.type))
    if not node.name:
        raise ValueError("node name must be assigned")

    # - setup env for process (vars must be strings for os.environ)
    env = setup_env(node, machine, master_uri)

    if not node.name:
        raise ValueError("node name must be assigned")

    # we have to include the counter to prevent potential name
    # collisions between the two branches

    name = "%s-%s" % (rosgraph.names.ns_join(node.namespace, node.name), _next_counter())
    if name[0] == '/':
        name = name[1:]
    # [roslaunch][INFO] 2023-01-16 14:40:51,169: process[joint_state_publisher-1]: 
    # env[{'ROS_DISTRO': 'melodic', 'ROS_IP': '192.168.111.111', 'ROS_PACKAGE_PATH': '/opt/ros/melodic/share', 
    # 'PATH': '/opt/ros/melodic/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin', 
    # 'CMAKE_PREFIX_PATH': '/opt/ros/melodic', 'ROS_LOG_FILENAME': '/root/.ros/log/b598afc4-9568-11ed-b4d9-00044bde2742/roslaunch-tegra-ubuntu-30.log', 
    # 'LANG': 'C.UTF-8', 'TERM': 'xterm', 'SHLVL': '1', 'LD_LIBRARY_PATH': '/opt/ros/melodic/lib', 
    # 'ROS_MASTER_URI': 'http://192.168.111.111:11311', 'HOME': '/root', 'ROS_PYTHON_VERSION': '2',
    # 'PYTHONPATH': '/opt/ros/melodic/lib/python2.7/dist-packages', 'ROS_ROOT': '/opt/ros/melodic/share/ros', 
    # 'PKG_CONFIG_PATH': '/opt/ros/melodic/lib/pkgconfig', 'LC_ALL': 'C.UTF-8', '_': '/usr/bin/nohup', 
    # 'HOSTNAME': 'tegra-ubuntu', 'ROSLISP_PACKAGE_DIRECTORIES': '', 'PWD': '/', 'ROS_ETC_DIR': '/opt/ros/melodic/etc/ros', 
    # 'ROS_VERSION': '1'}]
    _logger.info('process[%s]: env[%s]', name, env)

    args = create_local_process_args(node, machine)
    # [roslaunch][INFO] 2023-01-16 14:40:51,203: process[joint_state_publisher-1]: 
    # args[[u'/opt/ros/melodic/lib/joint_state_publisher/joint_state_publisher', u'__name:=joint_state_publisher']]
    _logger.info('process[%s]: args[%s]', name, args)

    # default for node.output not set is 'log'
    log_output = node.output != 'screen'
    _logger.debug('process[%s]: returning LocalProcess wrapper')
    return LocalProcess(run_id, node.package, name, args, env, log_output,
                        respawn=node.respawn, respawn_delay=node.respawn_delay,
                        required=node.required, cwd=node.cwd)
```

```python
class LocalProcess(Process):
    """
    Process launched on local machine
    """

    def start(self):
        """
        Start the process.
        
        @raise FatalProcessLaunch: if process cannot be started and it
        is not likely to ever succeed
        """
        super(LocalProcess, self).start()
        try:
            self.lock.acquire()
            if self.started:
                _logger.info("process[%s]: restarting os process", self.name)
            else:
                _logger.info("process[%s]: starting os process", self.name)
            self.started = self.stopped = False

            full_env = self.env

            # _configure_logging() can mutate self.args
            try:
                logfileout, logfileerr = self._configure_logging()
            except Exception as e:
                _logger.error(traceback.format_exc())
                printerrlog("[%s] ERROR: unable to configure logging [%s]" % (self.name, str(e)))
                # it's not safe to inherit from this process as
                # rostest changes stdout to a StringIO, which is not a
                # proper file.
                logfileout, logfileerr = subprocess.PIPE, subprocess.PIPE

            if self.cwd == 'node':
                cwd = os.path.dirname(self.args[0])
            elif self.cwd == 'cwd':
                cwd = os.getcwd()
            elif self.cwd == 'ros-root':
                cwd = get_ros_root()
            else:
                cwd = rospkg.get_ros_home()
            if not os.path.exists(cwd):
                try:
                    os.makedirs(cwd)
                except OSError:
                    # exist_ok=True
                    pass
            # [roslaunch][INFO] 2023-01-16 14:40:51,205: process[joint_state_publisher-1]: start w/ 
            # args [[u'/opt/ros/melodic/lib/joint_state_publisher/joint_state_publisher', 
            # u'__name:=joint_state_publisher', u'__log:=/root/.ros/log/b598afc4-9568-11ed-b4d9-00044bde2742/joint_state_publisher-1.log']]
            # [roslaunch][INFO] 2023-01-16 14:40:51,205: process[joint_state_publisher-1]: cwd will be [/root/.ros]
            _logger.info("process[%s]: start w/ args [%s]", self.name, self.args)
            _logger.info("process[%s]: cwd will be [%s]", self.name, cwd)

            try:
                preexec_function = os.setsid
                close_file_descriptor = True
            except AttributeError:
                preexec_function = None
                close_file_descriptor = False

            try:
                self.popen = subprocess.Popen(self.args, cwd=cwd, stdout=logfileout, stderr=logfileerr, env=full_env,
                                              close_fds=close_file_descriptor, preexec_fn=preexec_function)
            except OSError as e:
                self.started = True  # must set so is_alive state is correct
                _logger.error("OSError(%d, %s)", e.errno, e.strerror)
                if e.errno == errno.ENOEXEC:  # Exec format error
                    raise FatalProcessLaunch(
                        "Unable to launch [%s]. \nIf it is a script, you may be missing a '#!' declaration at the top." % self.name)
                elif e.errno == errno.ENOENT:  # no such file or directory
                    raise FatalProcessLaunch("""Roslaunch got a '%s' error while attempting to run:

%s

Please make sure that all the executables in this command exist and have
executable permission. This is often caused by a bad launch-prefix.""" % (e.strerror, ' '.join(self.args)))
                else:
                    raise FatalProcessLaunch("unable to launch [%s]: %s" % (' '.join(self.args), e.strerror))

            self.started = True
            # Check that the process is either still running (poll returns
            # None) or that it completed successfully since when we
            # launched it above (poll returns the return code, 0).
            poll_result = self.popen.poll()
            if poll_result is None or poll_result == 0:
                self.pid = self.popen.pid
                # [roslaunch][INFO] 2023-01-16 14:40:51,957: process[joint_state_publisher-1]: started with pid [78]
                printlog_bold("process[%s]: started with pid [%s]" % (self.name, self.pid))
                return True
            else:
                printerrlog("failed to start local process: %s" % (' '.join(self.args)))
                return False
        finally:
            self.lock.release()
```

## 2.3、roslaunch 包分析

### 2.3.1、python distutils

distutils可以用来在Python环境中构建和安装额外的模块。新的模块可以是纯python的，也可以是用C/C++写的扩展模块，或者可以是Python包，包中包含了由C和Python编写的模块。
对于模块开发者以及需要安装模块的使用者来说，distutils的使用都很简单，作为一个开发者，除了编写源码之外，还需要：

- 编写setup脚本（一般是setup.py）；
- 编写一个setup配置文件（可选）；
- 创建一个源码发布；
- 创建一个或多个构建（二进制）发布（可选）;

一个setup.py的简单例子

```python
from distutils.core import setup

setup(name='Distutils',
      version='1.0',
      description='Python Distribution Utilities',
      author='Greg Ward',
      author_email='xxx@python.net',
      url='https://www.python.org/sigs/distutils-sig/',
      packages=['distutils', 'distutils.command'],
      )
```

关于distutils的具体用法，可以参考[官方文档](https://docs.python.org/2/distutils/setupscript.html)

### 2.3.2、roslaunch 包结构分析

roslaunch的setup.py

```python
from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# 参数收集，返回到d,dict
d = generate_distutils_setup(
    packages=['roslaunch'],
    package_dir={'': 'src'},
    scripts=['scripts/roscore',
             'scripts/roslaunch',
             'scripts/roslaunch-complete',
             'scripts/roslaunch-deps',
             'scripts/roslaunch-logs'],
    requires=['genmsg', 'genpy', 'roslib', 'rospkg']
)
# 序列解包
setup(**d)
```

而其中的catkin_pkg，其git地址为[https://github.com/ros-infrastructure/catkin_pkg.git](https://github.com/ros-infrastructure/catkin_pkg.git)
功能介绍如下

```text
catkin_pkg

Standalone Python library for the Catkin package system.
```

下面是`generate_distutils_setup()`函数的实现，这里用到了`**`在函数定义时的参数收集功能(dict)， 其核心功能就是将package.xml文件中的内容解析放到一个字典中，然后返回。(
而且还要加上输入参数kwargs，输入参数kwargs中收集的key如果在package.xml中有，则值必须一样，如果没有，则添加到返回值中)

```python
# catkin_pkg\src\catkin_pkg\python_setup.py

from .package import InvalidPackage, parse_package


def generate_distutils_setup(package_xml_path=os.path.curdir, **kwargs):
    """
    Extract the information relevant for distutils from the package
    manifest. The following keys will be set:

    The "name" and "version" are taken from the eponymous tags.

    A single maintainer will set the keys "maintainer" and
    "maintainer_email" while multiple maintainers are merged into the
    "maintainer" fields (including their emails). Authors are handled
    likewise.

    The first URL of type "website" (or without a type) is used for
    the "url" field.

    The "description" is taken from the eponymous tag if it does not
    exceed 200 characters. If it does "description" contains the
    truncated text while "description_long" contains the complete.

    All licenses are merged into the "license" field.

    :param kwargs: All keyword arguments are passed through. The above
        mentioned keys are verified to be identical if passed as a
        keyword argument

    :returns: return dict populated with parsed fields and passed
        keyword arguments
    :raises: :exc:`InvalidPackage`
    :raises: :exc:`IOError`
    """
    package = parse_package(package_xml_path)

    data = {}
    data['name'] = package.name
    data['version'] = package.version

    # either set one author with one email or join all in a single field
    if len(package.authors) == 1 and package.authors[0].email is not None:
        data['author'] = package.authors[0].name
        data['author_email'] = package.authors[0].email
    else:
        data['author'] = ', '.join(
            [('%s <%s>' % (a.name, a.email) if a.email is not None else a.name) for a in package.authors])

    # either set one maintainer with one email or join all in a single field
    if len(package.maintainers) == 1:
        data['maintainer'] = package.maintainers[0].name
        data['maintainer_email'] = package.maintainers[0].email
    else:
        data['maintainer'] = ', '.join(['%s <%s>' % (m.name, m.email) for m in package.maintainers])

    # either set the first URL with the type 'website' or the first URL of any type
    websites = [url.url for url in package.urls if url.type == 'website']
    if websites:
        data['url'] = websites[0]
    elif package.urls:
        data['url'] = package.urls[0].url

    if len(package.description) <= 200:
        data['description'] = package.description
    else:
        data['description'] = package.description[:197] + '...'
        data['long_description'] = package.description

    data['license'] = ', '.join(package.licenses)

    # 输入参数kwargs中收集的key如果在package.xml中有，则值必须一样；
    # 如果没有，则添加到返回值中。
    # pass keyword arguments and verify equality if generated and passed in
    for k, v in kwargs.items():
        if k in data:
            if v != data[k]:
                raise InvalidPackage(
                    'The keyword argument "%s" does not match the information from package.xml: "%s" != "%s"' % (
                        k, v, data[k]))
        else:
            data[k] = v

    return data
```

而，`package.xml`中都是一些distutils中setup()函数执行时需要的一些参数，用xml进行可配置化。

```xml
<?xml version="1.0"?>
<package>
    <name>roslaunch</name>
    <version>1.13.0</version>
    <description>
        roslaunch is a tool for easily launching multiple ROS <a
            href="http://ros.org/wiki/Nodes">nodes
    </a> locally and remotely
        via SSH, as well as setting parameters on the <a
            href="http://ros.org/wiki/Parameter Server">Parameter
        Server</a>. It includes options to automatically respawn processes
        that have already died. roslaunch takes in one or more XML
        configuration files (with the <tt>.launch</tt> extension) that
        specify the parameters to set and nodes to launch, as well as the
        machines that they should be run on.
    </description>
    <maintainer email="dthomas@osrfoundation.org">Dirk Thomas</maintainer>
    <license>BSD</license>

    <url>http://ros.org/wiki/roslaunch</url>
    <author>Ken Conley</author>

    <buildtool_depend version_gte="0.5.78">catkin</buildtool_depend>

    <run_depend>python-paramiko</run_depend>
    <run_depend version_gte="1.0.37">python-rospkg</run_depend>
    <run_depend>python-yaml</run_depend>
    <run_depend>rosclean</run_depend>
    <run_depend>rosgraph_msgs</run_depend>
    <run_depend>roslib</run_depend>
    <run_depend version_gte="1.11.16">rosmaster</run_depend>
    <run_depend>rosout</run_depend>
    <run_depend>rosparam</run_depend>
    <run_depend version_gte="1.13.3">rosunit</run_depend>

    <test_depend>rosbuild</test_depend>

    <export>
        <rosdoc config="rosdoc.yaml"/>
        <architecture_independent/>
    </export>
</package>
```

setup()函数的输入参数中，scripts的解释如下，

So far we have been dealing with pure and non-pure Python modules, which are usually not run by themselves but imported
by scripts. Scripts are files containing Python source code, intended to be started from the command line. Scripts don’t
require Distutils to do anything very complicated.

> 到目前为止，我们一直在处理纯和非纯Python模块，这些模块通常不是自己运行的，而是由脚本导入的。
> 脚本是包含python源代码的文件，旨在从命令行启动。脚本不需要Distutils做任何非常复杂的事情

所以，python 模块主要是用来被其他模块去import，而script是为了直接在命令行执行，类似于应用程序。

而roscore就是这样一个需要在命令行执行的脚本(程序)

```python
scripts = ['scripts/roscore',
           'scripts/roslaunch',
           'scripts/roslaunch-complete',
           'scripts/roslaunch-deps',
           'scripts/roslaunch-logs']
```

而roscore最终会去import roslaunch package，去调用其中的main函数。

```python
# ros_comm\tools\roslaunch\scripts\roscore
import roslaunch

roslaunch.main(['roscore', '--core'] + sys.argv[1:])
```

























