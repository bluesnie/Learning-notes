###### datetime:2023/01/30 11:41

###### author:nzb

# ROS 日志

## roscore 日志

```text
[roslaunch][INFO] 2023-01-16 14:40:47,556: Checking log directory for disk usage. This may take awhile.
Press Ctrl-C to interrupt
[roslaunch][INFO] 2023-01-16 14:40:47,601: Done checking log file disk usage. Usage is <1GB.
[roslaunch][INFO] 2023-01-16 14:40:47,602: roslaunch starting with args ['roscore', '--core']
[roslaunch][INFO] 2023-01-16 14:40:47,603: roslaunch env is {'ROS_DISTRO': 'melodic', 'ROSLISP_PACKAGE_DIRECTORIES': '', 'HOME': '/root', 'PATH': '/opt/ros/melodic/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin', 'ROS_PACKAGE_PATH': '/opt/ros/melodic/share', 'CMAKE_PREFIX_PATH': '/opt/ros/melodic', 'LD_LIBRARY_PATH': '/opt/ros/melodic/lib', 'LANG': 'C.UTF-8', 'TERM': 'xterm', 'SHLVL': '1', 'ROS_LOG_FILENAME': '/root/.ros/log/b598afc4-9568-11ed-b4d9-00044bde2742/roslaunch-tegra-ubuntu-29.log', 'ROS_MASTER_URI': 'http://192.168.111.111:11311', 'ROS_PYTHON_VERSION': '2', 'PYTHONPATH': '/opt/ros/melodic/lib/python2.7/dist-packages', 'ROS_ROOT': '/opt/ros/melodic/share/ros', 'PKG_CONFIG_PATH': '/opt/ros/melodic/lib/pkgconfig', 'LC_ALL': 'C.UTF-8', '_': '/usr/bin/nohup', 'HOSTNAME': 'tegra-ubuntu', 'ROS_IP': '192.168.111.111', 'PWD': '/', 'ROS_ETC_DIR': '/opt/ros/melodic/etc/ros', 'ROS_VERSION': '1'}
[roslaunch][INFO] 2023-01-16 14:40:47,603: starting in server mode
[roslaunch.parent][INFO] 2023-01-16 14:40:47,604: starting roslaunch parent run
[roslaunch][INFO] 2023-01-16 14:40:47,605: loading roscore config file /opt/ros/melodic/etc/ros/roscore.xml
[roslaunch][INFO] 2023-01-16 14:40:48,721: Added core node of type [rosout/rosout] in namespace [/]
[roslaunch.pmon][INFO] 2023-01-16 14:40:48,721: start_process_monitor: creating ProcessMonitor
[roslaunch.pmon][INFO] 2023-01-16 14:40:48,722: created process monitor <ProcessMonitor(ProcessMonitor-1, initial daemon)>
[roslaunch.pmon][INFO] 2023-01-16 14:40:48,723: start_process_monitor: ProcessMonitor started
[roslaunch.parent][INFO] 2023-01-16 14:40:48,723: starting parent XML-RPC server
[roslaunch.server][INFO] 2023-01-16 14:40:48,723: starting roslaunch XML-RPC server
[roslaunch.server][INFO] 2023-01-16 14:40:48,724: waiting for roslaunch XML-RPC server to initialize
[xmlrpc][INFO] 2023-01-16 14:40:48,724: XML-RPC server binding to 0.0.0.0:0
[xmlrpc][INFO] 2023-01-16 14:40:48,725: Started XML-RPC server [http://192.168.111.111:35647/]
[xmlrpc][INFO] 2023-01-16 14:40:48,726: xml rpc node: starting XML-RPC server
[roslaunch][INFO] 2023-01-16 14:40:48,737: started roslaunch server http://192.168.111.111:35647/
[roslaunch.parent][INFO] 2023-01-16 14:40:48,738: ... parent XML-RPC server started
[roslaunch][INFO] 2023-01-16 14:40:48,739: master.is_running[http://192.168.111.111:11311/]
[roslaunch][INFO] 2023-01-16 14:40:48,751: auto-starting new master
[roslaunch][INFO] 2023-01-16 14:40:48,752: create_master_process: rosmaster, /opt/ros/melodic/share/ros, 11311, 3, None, False
[roslaunch][INFO] 2023-01-16 14:40:48,752: process[master]: launching with args [['rosmaster', '--core', '-p', '11311', '-w', '3']]
[roslaunch.pmon][INFO] 2023-01-16 14:40:48,753: ProcessMonitor.register[master]
[roslaunch.pmon][INFO] 2023-01-16 14:40:48,753: ProcessMonitor.register[master] complete
[roslaunch][INFO] 2023-01-16 14:40:48,753: process[master]: starting os process
[roslaunch][INFO] 2023-01-16 14:40:48,753: process[master]: start w/ args [['rosmaster', '--core', '-p', '11311', '-w', '3', '__log:=/root/.ros/log/b598afc4-9568-11ed-b4d9-00044bde2742/master.log']]
[roslaunch][INFO] 2023-01-16 14:40:48,754: process[master]: cwd will be [/root/.ros]
[roslaunch][INFO] 2023-01-16 14:40:49,535: process[master]: started with pid [46]
[roslaunch][INFO] 2023-01-16 14:40:49,536: master.is_running[http://192.168.111.111:11311/]
[roslaunch][INFO] 2023-01-16 14:40:49,638: master.is_running[http://192.168.111.111:11311/]
[roslaunch][INFO] 2023-01-16 14:40:49,739: master.is_running[http://192.168.111.111:11311/]
[roslaunch][INFO] 2023-01-16 14:40:49,840: master.is_running[http://192.168.111.111:11311/]
[roslaunch][INFO] 2023-01-16 14:40:49,942: master.is_running[http://192.168.111.111:11311/]
[roslaunch][INFO] 2023-01-16 14:40:50,043: master.is_running[http://192.168.111.111:11311/]
[roslaunch][INFO] 2023-01-16 14:40:50,048: master.is_running[http://192.168.111.111:11311/]
[roslaunch][INFO] 2023-01-16 14:40:50,051: ROS_MASTER_URI=http://192.168.111.111:11311/
[roslaunch][INFO] 2023-01-16 14:40:50,054: setting /run_id to b598afc4-9568-11ed-b4d9-00044bde2742
[roslaunch][INFO] 2023-01-16 14:40:50,057: setting /roslaunch/uris/host_192_168_111_111__35647' to http://192.168.111.111:35647/
[roslaunch][INFO] 2023-01-16 14:40:50,064: ... preparing to launch node of type [rosout/rosout]
[roslaunch][INFO] 2023-01-16 14:40:50,064: create_node_process: package[rosout] type[rosout] machine[Machine(name[] env_loader[None] address[localhost] ssh_port[22] user[None] assignable[True] timeout[10.0])] master_uri[http://192.168.111.111:11311/]
[roslaunch][INFO] 2023-01-16 14:40:50,065: process[rosout-1]: env[{'ROS_DISTRO': 'melodic', 'ROS_IP': '192.168.111.111', 'ROS_PACKAGE_PATH': '/opt/ros/melodic/share', 'PATH': '/opt/ros/melodic/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin', 'CMAKE_PREFIX_PATH': '/opt/ros/melodic', 'ROS_LOG_FILENAME': '/root/.ros/log/b598afc4-9568-11ed-b4d9-00044bde2742/roslaunch-tegra-ubuntu-29.log', 'LANG': 'C.UTF-8', 'TERM': 'xterm', 'SHLVL': '1', 'LD_LIBRARY_PATH': '/opt/ros/melodic/lib', 'ROS_MASTER_URI': 'http://192.168.111.111:11311/', 'HOME': '/root', 'ROS_PYTHON_VERSION': '2', 'PYTHONPATH': '/opt/ros/melodic/lib/python2.7/dist-packages', 'ROS_ROOT': '/opt/ros/melodic/share/ros', 'PKG_CONFIG_PATH': '/opt/ros/melodic/lib/pkgconfig', 'LC_ALL': 'C.UTF-8', '_': '/usr/bin/nohup', 'HOSTNAME': 'tegra-ubuntu', 'ROSLISP_PACKAGE_DIRECTORIES': '', 'PWD': '/', 'ROS_ETC_DIR': '/opt/ros/melodic/etc/ros', 'ROS_VERSION': '1'}]
[roslaunch][INFO] 2023-01-16 14:40:50,095: process[rosout-1]: args[[u'/opt/ros/melodic/lib/rosout/rosout', u'__name:=rosout']]
[roslaunch][INFO] 2023-01-16 14:40:50,096: ... created process [rosout-1]
[roslaunch.pmon][INFO] 2023-01-16 14:40:50,096: ProcessMonitor.register[rosout-1]
[roslaunch.pmon][INFO] 2023-01-16 14:40:50,097: ProcessMonitor.register[rosout-1] complete
[roslaunch][INFO] 2023-01-16 14:40:50,097: ... registered process [rosout-1]
[roslaunch][INFO] 2023-01-16 14:40:50,097: process[rosout-1]: starting os process
[roslaunch][INFO] 2023-01-16 14:40:50,098: process[rosout-1]: start w/ args [[u'/opt/ros/melodic/lib/rosout/rosout', u'__name:=rosout', u'__log:=/root/.ros/log/b598afc4-9568-11ed-b4d9-00044bde2742/rosout-1.log']]
[roslaunch][INFO] 2023-01-16 14:40:50,098: process[rosout-1]: cwd will be [/root/.ros]
[roslaunch][INFO] 2023-01-16 14:40:50,847: process[rosout-1]: started with pid [59]
[roslaunch][INFO] 2023-01-16 14:40:50,848: ... successfully launched [rosout-1]
[roslaunch][INFO] 2023-01-16 14:40:50,849: load_parameters starting ...
[roslaunch][INFO] 2023-01-16 14:40:50,858: ... load_parameters complete
[roslaunch][INFO] 2023-01-16 14:40:50,858: launch_nodes: launching local nodes ...
[roslaunch][INFO] 2023-01-16 14:40:50,858: ... launch_nodes complete
[roslaunch.pmon][INFO] 2023-01-16 14:40:50,858: registrations completed <ProcessMonitor(ProcessMonitor-1, started daemon 548405932528)>
[roslaunch.parent][INFO] 2023-01-16 14:40:50,859: ... roslaunch parent running, waiting for process exit
[roslaunch][INFO] 2023-01-16 14:40:50,859: spin
```

## master 日志

```text
[rosmaster.main][INFO] 2023-01-16 14:40:49,996: initialization complete, waiting for shutdown
[rosmaster.main][INFO] 2023-01-16 14:40:49,996: Starting ROS Master Node
[xmlrpc][INFO] 2023-01-16 14:40:49,999: XML-RPC server binding to 0.0.0.0:11311
[rosmaster.master][INFO] 2023-01-16 14:40:50,000: Master initialized: port[11311], uri[http://192.168.111.111:11311/]
[xmlrpc][INFO] 2023-01-16 14:40:50,000: Started XML-RPC server [http://192.168.111.111:11311/]
[xmlrpc][INFO] 2023-01-16 14:40:50,001: xml rpc node: starting XML-RPC server
[rosmaster.master][INFO] 2023-01-16 14:40:50,056: +PARAM [/run_id] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:40:50,059: +PARAM [/roslaunch/uris/host_192_168_111_111__35647] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:40:50,856: +PARAM [/rosversion] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:40:50,856: +PARAM [/rosdistro] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:40:51,065: +SERVICE [/rosout/get_loggers] /rosout http://192.168.111.111:38431/
[rosmaster.master][INFO] 2023-01-16 14:40:51,067: +SERVICE [/rosout/set_logger_level] /rosout http://192.168.111.111:38431/
[rosmaster.master][INFO] 2023-01-16 14:40:51,072: +PUB [/rosout_agg] /rosout http://192.168.111.111:38431/
[rosmaster.master][INFO] 2023-01-16 14:40:51,078: +SUB [/rosout] /rosout http://192.168.111.111:38431/
[rosmaster.master][INFO] 2023-01-16 14:40:51,157: +PARAM [/roslaunch/uris/host_192_168_111_111__33083] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:40:51,165: +PARAM [/rosversion] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:40:51,165: +PARAM [/rosdistro] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:40:51,166: +PARAM [/robot_description] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:40:51,166: +PARAM [/robot_state_publisher/publish_frequency] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:40:52,534: +PUB [/rosout] /joint_state_publisher http://192.168.111.111:41129/
[rosmaster.master][INFO] 2023-01-16 14:40:52,542: +SERVICE [/joint_state_publisher/get_loggers] /joint_state_publisher http://192.168.111.111:41129/
[rosmaster.master][INFO] 2023-01-16 14:40:52,546: +SERVICE [/joint_state_publisher/set_logger_level] /joint_state_publisher http://192.168.111.111:41129/
[rosmaster.master][INFO] 2023-01-16 14:40:52,577: +PUB [/joint_states] /joint_state_publisher http://192.168.111.111:41129/
[rosmaster.master][INFO] 2023-01-16 14:40:52,604: publisherUpdate[/rosout] -> http://192.168.111.111:38431/ ['http://192.168.111.111:41129/']
[rosmaster.master][INFO] 2023-01-16 14:40:52,833: +PUB [/rosout] /robot_state_publisher http://192.168.111.111:43717/
[rosmaster.master][INFO] 2023-01-16 14:40:52,834: +SERVICE [/robot_state_publisher/get_loggers] /robot_state_publisher http://192.168.111.111:43717/
[rosmaster.master][INFO] 2023-01-16 14:40:52,835: +SERVICE [/robot_state_publisher/set_logger_level] /robot_state_publisher http://192.168.111.111:43717/
[rosmaster.master][INFO] 2023-01-16 14:40:52,844: +PUB [/tf] /robot_state_publisher http://192.168.111.111:43717/
[rosmaster.master][INFO] 2023-01-16 14:40:52,846: +PUB [/tf_static] /robot_state_publisher http://192.168.111.111:43717/
[rosmaster.master][INFO] 2023-01-16 14:40:52,857: +SUB [/joint_states] /robot_state_publisher http://192.168.111.111:43717/
[rosmaster.master][INFO] 2023-01-16 14:40:52,945: publisherUpdate[/rosout] -> http://192.168.111.111:38431/ ['http://192.168.111.111:41129/']: sec=0.34, result=[1, '', 0]
[rosmaster.master][INFO] 2023-01-16 14:40:52,945: publisherUpdate[/rosout] -> http://192.168.111.111:38431/ ['http://192.168.111.111:41129/', 'http://192.168.111.111:43717/']
[rosmaster.master][INFO] 2023-01-16 14:40:52,948: publisherUpdate[/rosout] -> http://192.168.111.111:38431/ ['http://192.168.111.111:41129/', 'http://192.168.111.111:43717/']: sec=0.00, result=[1, '', 0]
[rosmaster.master][INFO] 2023-01-16 14:41:20,906: +PARAM [/roslaunch/uris/host_192_168_111_111__45309] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:41:20,915: +PARAM [/rosversion] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:41:20,915: +PARAM [/scans_concat_filter/output_frame_id] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:41:20,916: +PARAM [/rosdistro] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:41:20,916: +PARAM [/scans_concat_filter/input_topics] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:41:21,195: +PARAM [/roslaunch/uris/host_192_168_111_111__35101] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:41:21,205: +PARAM [/rear/r2000_node/samples_per_scan] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:41:21,205: +PARAM [/rosdistro] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:41:21,205: +PARAM [/rear/r2000_node/frame_id] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:41:21,206: +PARAM [/front/r2000_node/scan_frequency] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:41:21,206: +PARAM [/rosversion] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:41:21,207: +PARAM [/rear/r2000_node/scan_frequency] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:41:21,207: +PARAM [/rear/r2000_node/scanner_ip] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:41:21,207: +PARAM [/front/r2000_node/frame_id] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:41:21,208: +PARAM [/front/r2000_node/scanner_ip] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:41:21,208: +PARAM [/front/r2000_node/samples_per_scan] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:41:22,026: +PUB [/rosout] /front/r2000_node http://192.168.111.111:45843/
[rosmaster.master][INFO] 2023-01-16 14:41:22,028: +SERVICE [/front/r2000_node/get_loggers] /front/r2000_node http://192.168.111.111:45843/
[rosmaster.master][INFO] 2023-01-16 14:41:22,029: +SERVICE [/front/r2000_node/set_logger_level] /front/r2000_node http://192.168.111.111:45843/
[rosmaster.master][INFO] 2023-01-16 14:41:22,057: publisherUpdate[/rosout] -> http://192.168.111.111:38431/ ['http://192.168.111.111:41129/', 'http://192.168.111.111:43717/', 'http://192.168.111.111:45843/']
[rosmaster.master][INFO] 2023-01-16 14:41:22,059: publisherUpdate[/rosout] -> http://192.168.111.111:38431/ ['http://192.168.111.111:41129/', 'http://192.168.111.111:43717/', 'http://192.168.111.111:45843/']: sec=0.00, result=[1, '', 0]
[rosmaster.master][INFO] 2023-01-16 14:41:22,154: +PUB [/rosout] /scans_concat_filter http://192.168.111.111:39961/
[rosmaster.master][INFO] 2023-01-16 14:41:22,155: +SERVICE [/scans_concat_filter/get_loggers] /scans_concat_filter http://192.168.111.111:39961/
[rosmaster.master][INFO] 2023-01-16 14:41:22,156: +SERVICE [/scans_concat_filter/set_logger_level] /scans_concat_filter http://192.168.111.111:39961/
[rosmaster.master][INFO] 2023-01-16 14:41:22,160: publisherUpdate[/rosout] -> http://192.168.111.111:38431/ ['http://192.168.111.111:41129/', 'http://192.168.111.111:43717/', 'http://192.168.111.111:45843/', 'http://192.168.111.111:39961/']
[rosmaster.master][INFO] 2023-01-16 14:41:22,162: publisherUpdate[/rosout] -> http://192.168.111.111:38431/ ['http://192.168.111.111:41129/', 'http://192.168.111.111:43717/', 'http://192.168.111.111:45843/', 'http://192.168.111.111:39961/']: sec=0.00, result=[1, '', 0]
[rosmaster.master][INFO] 2023-01-16 14:41:22,169: +SUB [/tf] /scans_concat_filter http://192.168.111.111:39961/
[rosmaster.master][INFO] 2023-01-16 14:41:22,207: +SUB [/tf_static] /scans_concat_filter http://192.168.111.111:39961/
[rosmaster.master][INFO] 2023-01-16 14:41:22,472: +PUB [/front/scan_dense] /front/r2000_node http://192.168.111.111:45843/
[rosmaster.master][INFO] 2023-01-16 14:41:22,475: +PUB [/front/scan] /front/r2000_node http://192.168.111.111:45843/
[rosmaster.master][INFO] 2023-01-16 14:41:22,483: +PUB [/front/r2000_node/diagno_msg] /front/r2000_node http://192.168.111.111:45843/
[rosmaster.master][INFO] 2023-01-16 14:41:22,509: +SUB [/front/r2000_node/control_command] /front/r2000_node http://192.168.111.111:45843/
[rosmaster.master][INFO] 2023-01-16 14:41:22,858: +PUB [/rosout] /rear/r2000_node http://192.168.111.111:39017/
[rosmaster.master][INFO] 2023-01-16 14:41:22,859: +SERVICE [/rear/r2000_node/get_loggers] /rear/r2000_node http://192.168.111.111:39017/
[rosmaster.master][INFO] 2023-01-16 14:41:22,861: +SERVICE [/rear/r2000_node/set_logger_level] /rear/r2000_node http://192.168.111.111:39017/
[rosmaster.master][INFO] 2023-01-16 14:41:22,864: publisherUpdate[/rosout] -> http://192.168.111.111:38431/ ['http://192.168.111.111:41129/', 'http://192.168.111.111:43717/', 'http://192.168.111.111:45843/', 'http://192.168.111.111:39961/', 'http://192.168.111.111:39017/']
[rosmaster.master][INFO] 2023-01-16 14:41:22,867: publisherUpdate[/rosout] -> http://192.168.111.111:38431/ ['http://192.168.111.111:41129/', 'http://192.168.111.111:43717/', 'http://192.168.111.111:45843/', 'http://192.168.111.111:39961/', 'http://192.168.111.111:39017/']: sec=0.00, result=[1, '', 0]
[rosmaster.master][INFO] 2023-01-16 14:41:23,240: +SUB [/front/scan] /scans_concat_filter http://192.168.111.111:39961/
[rosmaster.master][INFO] 2023-01-16 14:41:23,247: +SUB [/rear/scan] /scans_concat_filter http://192.168.111.111:39961/
[rosmaster.master][INFO] 2023-01-16 14:41:23,277: +PUB [/cloud] /scans_concat_filter http://192.168.111.111:39961/
[rosmaster.master][INFO] 2023-01-16 14:41:23,287: +SUB [/odom] /scans_concat_filter http://192.168.111.111:39961/
[rosmaster.master][INFO] 2023-01-16 14:41:23,289: +PUB [/front_t] /scans_concat_filter http://192.168.111.111:39961/
[rosmaster.master][INFO] 2023-01-16 14:41:23,292: +PUB [/rear_t] /scans_concat_filter http://192.168.111.111:39961/
[rosmaster.master][INFO] 2023-01-16 14:41:23,313: +PUB [/rear/scan_dense] /rear/r2000_node http://192.168.111.111:39017/
[rosmaster.master][INFO] 2023-01-16 14:41:23,316: +PUB [/rear/scan] /rear/r2000_node http://192.168.111.111:39017/
[rosmaster.master][INFO] 2023-01-16 14:41:23,318: +PUB [/rear/r2000_node/diagno_msg] /rear/r2000_node http://192.168.111.111:39017/
[rosmaster.master][INFO] 2023-01-16 14:41:23,330: +SUB [/rear/r2000_node/control_command] /rear/r2000_node http://192.168.111.111:39017/
[rosmaster.master][INFO] 2023-01-16 14:41:23,367: publisherUpdate[/rear/scan] -> http://192.168.111.111:39961/ ['http://192.168.111.111:39017/']
[rosmaster.master][INFO] 2023-01-16 14:41:23,384: +PARAM [/roslaunch/uris/host_192_168_111_111__45005] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:41:23,390: +PARAM [/rosversion] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:41:23,391: +PARAM [/rosdistro] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:41:23,391: +PARAM [/obstacle_detection/obstacle_source] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:41:23,490: +PUB [/rosout] /slam_hal http://192.168.111.111:41995/
[rosmaster.master][INFO] 2023-01-16 14:41:23,491: +SERVICE [/slam_hal/get_loggers] /slam_hal http://192.168.111.111:41995/
[rosmaster.master][INFO] 2023-01-16 14:41:23,494: +SERVICE [/slam_hal/set_logger_level] /slam_hal http://192.168.111.111:41995/
[rosmaster.master][INFO] 2023-01-16 14:41:23,500: publisherUpdate[/rosout] -> http://192.168.111.111:38431/ ['http://192.168.111.111:41129/', 'http://192.168.111.111:43717/', 'http://192.168.111.111:45843/', 'http://192.168.111.111:39961/', 'http://192.168.111.111:39017/', 'http://192.168.111.111:41995/']
[rosmaster.master][INFO] 2023-01-16 14:41:23,502: publisherUpdate[/rosout] -> http://192.168.111.111:38431/ ['http://192.168.111.111:41129/', 'http://192.168.111.111:43717/', 'http://192.168.111.111:45843/', 'http://192.168.111.111:39961/', 'http://192.168.111.111:39017/', 'http://192.168.111.111:41995/']: sec=0.00, result=[1, '', 0]
[rosmaster.master][INFO] 2023-01-16 14:41:23,504: +PUB [/odom] /slam_hal http://192.168.111.111:41995/
[rosmaster.master][INFO] 2023-01-16 14:41:23,512: +PUB [/dsp_pos] /slam_hal http://192.168.111.111:41995/
[rosmaster.master][INFO] 2023-01-16 14:41:23,526: +SUB [/cur_pose] /slam_hal http://192.168.111.111:41995/
[rosmaster.master][INFO] 2023-01-16 14:41:23,744: publisherUpdate[/rear/scan] -> http://192.168.111.111:39961/ ['http://192.168.111.111:39017/']: sec=0.38, result=[1, '', 0]
[rosmaster.master][INFO] 2023-01-16 14:41:23,745: publisherUpdate[/odom] -> http://192.168.111.111:39961/ ['http://192.168.111.111:41995/']
[rosmaster.master][INFO] 2023-01-16 14:41:23,747: publisherUpdate[/odom] -> http://192.168.111.111:39961/ ['http://192.168.111.111:41995/']: sec=0.00, result=[1, '', 0]
[rosmaster.master][INFO] 2023-01-16 14:41:34,941: +PARAM [/roslaunch/uris/host_192_168_111_111__35333] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:41:34,949: +PARAM [/rosversion] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:41:34,950: +PARAM [/rosdistro] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:41:38,019: +PUB [/rosout] /charging_pile_recognition http://192.168.111.111:42095/
[rosmaster.master][INFO] 2023-01-16 14:41:38,020: +SERVICE [/charging_pile_recognition/get_loggers] /charging_pile_recognition http://192.168.111.111:42095/
[rosmaster.master][INFO] 2023-01-16 14:41:38,021: +SERVICE [/charging_pile_recognition/set_logger_level] /charging_pile_recognition http://192.168.111.111:42095/
[rosmaster.master][INFO] 2023-01-16 14:41:38,037: +SUB [/tf] /charging_pile_recognition http://192.168.111.111:42095/
[rosmaster.master][INFO] 2023-01-16 14:41:38,043: publisherUpdate[/rosout] -> http://192.168.111.111:38431/ ['http://192.168.111.111:41129/', 'http://192.168.111.111:43717/', 'http://192.168.111.111:45843/', 'http://192.168.111.111:39961/', 'http://192.168.111.111:39017/', 'http://192.168.111.111:41995/', 'http://192.168.111.111:42095/']
[rosmaster.master][INFO] 2023-01-16 14:41:38,049: +SUB [/tf_static] /charging_pile_recognition http://192.168.111.111:42095/
[rosmaster.master][INFO] 2023-01-16 14:41:38,055: publisherUpdate[/rosout] -> http://192.168.111.111:38431/ ['http://192.168.111.111:41129/', 'http://192.168.111.111:43717/', 'http://192.168.111.111:45843/', 'http://192.168.111.111:39961/', 'http://192.168.111.111:39017/', 'http://192.168.111.111:41995/', 'http://192.168.111.111:42095/']: sec=0.01, result=[1, '', 0]
[rosmaster.master][INFO] 2023-01-16 14:41:38,060: +SUB [/cloud] /charging_pile_recognition http://192.168.111.111:42095/
[rosmaster.master][INFO] 2023-01-16 14:41:38,072: +SERVICE [/GetChargingPilePose] /charging_pile_recognition http://192.168.111.111:42095/
[rosmaster.master][INFO] 2023-01-16 14:41:46,985: +PARAM [/roslaunch/uris/host_tegra_ubuntu__43647] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:41:46,993: +PARAM [/rosversion] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:41:46,993: +PARAM [/rosdistro] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:41:46,994: +PARAM [/operate_qslam_c/config_file] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:41:48,255: +PUB [/rosout] /operate_qslam_c http://tegra-ubuntu:32835/
[rosmaster.master][INFO] 2023-01-16 14:41:48,262: +SERVICE [/operate_qslam_c/get_loggers] /operate_qslam_c http://tegra-ubuntu:32835/
[rosmaster.master][INFO] 2023-01-16 14:41:48,264: +SERVICE [/operate_qslam_c/set_logger_level] /operate_qslam_c http://tegra-ubuntu:32835/
[rosmaster.master][INFO] 2023-01-16 14:41:48,270: +SUB [/quicktron/switch_mode] /operate_qslam_c http://tegra-ubuntu:32835/
[rosmaster.master][INFO] 2023-01-16 14:41:48,273: +SERVICE [/quicktron/increment_mapping] /operate_qslam_c http://tegra-ubuntu:32835/
[rosmaster.master][INFO] 2023-01-16 14:41:48,275: +SERVICE [/convert_map] /operate_qslam_c http://tegra-ubuntu:32835/
[rosmaster.master][INFO] 2023-01-16 14:41:48,278: +SERVICE [/operate_qslam_c] /operate_qslam_c http://tegra-ubuntu:32835/
[rosmaster.master][INFO] 2023-01-16 14:41:48,303: publisherUpdate[/rosout] -> http://192.168.111.111:38431/ ['http://192.168.111.111:41129/', 'http://192.168.111.111:43717/', 'http://192.168.111.111:45843/', 'http://192.168.111.111:39961/', 'http://192.168.111.111:39017/', 'http://192.168.111.111:41995/', 'http://192.168.111.111:42095/', 'http://tegra-ubuntu:32835/']
[rosmaster.master][INFO] 2023-01-16 14:41:48,306: publisherUpdate[/rosout] -> http://192.168.111.111:38431/ ['http://192.168.111.111:41129/', 'http://192.168.111.111:43717/', 'http://192.168.111.111:45843/', 'http://192.168.111.111:39961/', 'http://192.168.111.111:39017/', 'http://192.168.111.111:41995/', 'http://192.168.111.111:42095/', 'http://tegra-ubuntu:32835/']: sec=0.00, result=[1, '', 0]
[rosmaster.master][INFO] 2023-01-16 14:41:58,455: +PARAM [/roslaunch/uris/host_192_168_111_111__46555] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:41:58,474: +PARAM [/rosversion] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:41:58,474: +PARAM [/rosdistro] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:41:58,503: +PARAM [/roslaunch/uris/host_192_168_111_111__44533] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:41:58,512: +PARAM [/rosversion] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:41:58,513: +PARAM [/rosdistro] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:41:58,529: +PARAM [/roslaunch/uris/host_192_168_111_111__35149] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:41:58,540: +PARAM [/rosversion] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:41:58,540: +PARAM [/rosdistro] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:41:59,926: +PUB [/rosout] /operate_qslam_m http://192.168.111.111:45739/
[rosmaster.master][INFO] 2023-01-16 14:41:59,936: +SERVICE [/operate_qslam_m/get_loggers] /operate_qslam_m http://192.168.111.111:45739/
[rosmaster.master][INFO] 2023-01-16 14:41:59,940: +SERVICE [/operate_qslam_m/set_logger_level] /operate_qslam_m http://192.168.111.111:45739/
[rosmaster.master][INFO] 2023-01-16 14:41:59,950: publisherUpdate[/rosout] -> http://192.168.111.111:38431/ ['http://192.168.111.111:41129/', 'http://192.168.111.111:43717/', 'http://192.168.111.111:45843/', 'http://192.168.111.111:39961/', 'http://192.168.111.111:39017/', 'http://192.168.111.111:41995/', 'http://192.168.111.111:42095/', 'http://tegra-ubuntu:32835/', 'http://192.168.111.111:45739/']
[rosmaster.master][INFO] 2023-01-16 14:41:59,952: publisherUpdate[/rosout] -> http://192.168.111.111:38431/ ['http://192.168.111.111:41129/', 'http://192.168.111.111:43717/', 'http://192.168.111.111:45843/', 'http://192.168.111.111:39961/', 'http://192.168.111.111:39017/', 'http://192.168.111.111:41995/', 'http://192.168.111.111:42095/', 'http://tegra-ubuntu:32835/', 'http://192.168.111.111:45739/']: sec=0.00, result=[1, '', 0]
[rosmaster.master][INFO] 2023-01-16 14:42:00,112: +PUB [/rosout] /operate_qslam_a http://192.168.111.111:41973/
[rosmaster.master][INFO] 2023-01-16 14:42:00,128: +SERVICE [/operate_qslam_a/get_loggers] /operate_qslam_a http://192.168.111.111:41973/
[rosmaster.master][INFO] 2023-01-16 14:42:00,137: +SERVICE [/operate_qslam_a/set_logger_level] /operate_qslam_a http://192.168.111.111:41973/
[rosmaster.master][INFO] 2023-01-16 14:42:00,153: publisherUpdate[/rosout] -> http://192.168.111.111:38431/ ['http://192.168.111.111:41129/', 'http://192.168.111.111:43717/', 'http://192.168.111.111:45843/', 'http://192.168.111.111:39961/', 'http://192.168.111.111:39017/', 'http://192.168.111.111:41995/', 'http://192.168.111.111:42095/', 'http://tegra-ubuntu:32835/', 'http://192.168.111.111:45739/', 'http://192.168.111.111:41973/']
[rosmaster.master][INFO] 2023-01-16 14:42:00,158: publisherUpdate[/rosout] -> http://192.168.111.111:38431/ ['http://192.168.111.111:41129/', 'http://192.168.111.111:43717/', 'http://192.168.111.111:45843/', 'http://192.168.111.111:39961/', 'http://192.168.111.111:39017/', 'http://192.168.111.111:41995/', 'http://192.168.111.111:42095/', 'http://tegra-ubuntu:32835/', 'http://192.168.111.111:45739/', 'http://192.168.111.111:41973/']: sec=0.00, result=[1, '', 0]
[rosmaster.master][INFO] 2023-01-16 14:42:00,278: +PUB [/rosout] /operate_cold_start http://192.168.111.111:40973/
[rosmaster.master][INFO] 2023-01-16 14:42:00,281: publisherUpdate[/rosout] -> http://192.168.111.111:38431/ ['http://192.168.111.111:41129/', 'http://192.168.111.111:43717/', 'http://192.168.111.111:45843/', 'http://192.168.111.111:39961/', 'http://192.168.111.111:39017/', 'http://192.168.111.111:41995/', 'http://192.168.111.111:42095/', 'http://tegra-ubuntu:32835/', 'http://192.168.111.111:45739/', 'http://192.168.111.111:41973/', 'http://192.168.111.111:40973/']
[rosmaster.master][INFO] 2023-01-16 14:42:00,284: publisherUpdate[/rosout] -> http://192.168.111.111:38431/ ['http://192.168.111.111:41129/', 'http://192.168.111.111:43717/', 'http://192.168.111.111:45843/', 'http://192.168.111.111:39961/', 'http://192.168.111.111:39017/', 'http://192.168.111.111:41995/', 'http://192.168.111.111:42095/', 'http://tegra-ubuntu:32835/', 'http://192.168.111.111:45739/', 'http://192.168.111.111:41973/', 'http://192.168.111.111:40973/']: sec=0.00, result=[1, '', 0]
[rosmaster.master][INFO] 2023-01-16 14:42:00,299: +SERVICE [/operate_cold_start/get_loggers] /operate_cold_start http://192.168.111.111:40973/
[rosmaster.master][INFO] 2023-01-16 14:42:00,306: +SERVICE [/operate_cold_start/set_logger_level] /operate_cold_start http://192.168.111.111:40973/
[rosmaster.master][INFO] 2023-01-16 14:42:00,309: +SERVICE [/operate_cold_start] /operate_cold_start http://192.168.111.111:40973/
[rosmaster.master][INFO] 2023-01-16 14:42:01,828: +PARAM [/roslaunch/uris/host_192_168_111_111__33225] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:42:01,842: +PARAM [/rosversion] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:42:01,842: +PARAM [/use_sim_time] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:42:01,842: +PARAM [/rosdistro] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:42:01,843: +PARAM [/mekf_node/cfg_file_path] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:42:02,102: +PARAM [/roslaunch/uris/host_192_168_111_111__42169] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:42:02,116: +PARAM [/rosversion] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:42:02,117: +PARAM [/use_sim_time] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:42:02,117: +PARAM [/rosdistro] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:42:02,118: +PARAM [/map_server_new/config_dir] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:42:02,584: +PARAM [/roslaunch/uris/host_192_168_111_111__43933] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:42:02,598: +PARAM [/use_sim_time] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:42:02,598: +PARAM [/rosdistro] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:42:02,599: +PARAM [/cold_start_node/load_state_filename] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:42:02,599: +PARAM [/cold_start_node/configuration_directory] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:42:02,600: +PARAM [/cold_start_node/barcode_priority] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:42:02,600: +PARAM [/rosversion] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:42:02,601: +PARAM [/cold_start_node/init_theta] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:42:02,601: +PARAM [/cold_start_node/configuration_basename] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:42:02,607: +PARAM [/cold_start_node/reflector_priority] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:42:02,607: +PARAM [/cold_start_node/init_y] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:42:02,608: +PARAM [/cold_start_node/init_x] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:42:02,671: +SERVICE [/operate_qslam_m_new] /operate_qslam_m http://192.168.111.111:45739/
[rosmaster.master][INFO] 2023-01-16 14:42:02,674: +SERVICE [/operate_qslam_m] /operate_qslam_m http://192.168.111.111:45739/
[rosmaster.master][INFO] 2023-01-16 14:42:03,309: +PUB [/rosout] /map_server_new http://192.168.111.111:34787/
[rosmaster.master][INFO] 2023-01-16 14:42:03,312: +SERVICE [/map_server_new/get_loggers] /map_server_new http://192.168.111.111:34787/
[rosmaster.master][INFO] 2023-01-16 14:42:03,314: +SERVICE [/map_server_new/set_logger_level] /map_server_new http://192.168.111.111:34787/
[rosmaster.master][INFO] 2023-01-16 14:42:03,331: +SERVICE [/reload_map_service_with_init_pose] /map_server_new http://192.168.111.111:34787/
[rosmaster.master][INFO] 2023-01-16 14:42:03,335: +SERVICE [/static_map] /map_server_new http://192.168.111.111:34787/
[rosmaster.master][INFO] 2023-01-16 14:42:03,336: +PUB [/map_metadata] /map_server_new http://192.168.111.111:34787/
[rosmaster.master][INFO] 2023-01-16 14:42:03,337: +PUB [/map] /map_server_new http://192.168.111.111:34787/
[rosmaster.master][INFO] 2023-01-16 14:42:03,338: publisherUpdate[/rosout] -> http://192.168.111.111:38431/ ['http://192.168.111.111:41129/', 'http://192.168.111.111:43717/', 'http://192.168.111.111:45843/', 'http://192.168.111.111:39961/', 'http://192.168.111.111:39017/', 'http://192.168.111.111:41995/', 'http://192.168.111.111:42095/', 'http://tegra-ubuntu:32835/', 'http://192.168.111.111:45739/', 'http://192.168.111.111:41973/', 'http://192.168.111.111:40973/', 'http://192.168.111.111:34787/']
[rosmaster.master][INFO] 2023-01-16 14:42:03,341: publisherUpdate[/rosout] -> http://192.168.111.111:38431/ ['http://192.168.111.111:41129/', 'http://192.168.111.111:43717/', 'http://192.168.111.111:45843/', 'http://192.168.111.111:39961/', 'http://192.168.111.111:39017/', 'http://192.168.111.111:41995/', 'http://192.168.111.111:42095/', 'http://tegra-ubuntu:32835/', 'http://192.168.111.111:45739/', 'http://192.168.111.111:41973/', 'http://192.168.111.111:40973/', 'http://192.168.111.111:34787/']: sec=0.00, result=[1, '', 0]
[rosmaster.master][INFO] 2023-01-16 14:42:04,282: +SERVICE [/operate_qslam_a] /operate_qslam_a http://192.168.111.111:41973/
[rosmaster.master][INFO] 2023-01-16 14:42:04,314: +PUB [/rosout] /odom_tf http://192.168.111.111:33181/
[rosmaster.master][INFO] 2023-01-16 14:42:04,316: +SERVICE [/odom_tf/get_loggers] /odom_tf http://192.168.111.111:33181/
[rosmaster.master][INFO] 2023-01-16 14:42:04,317: +SERVICE [/odom_tf/set_logger_level] /odom_tf http://192.168.111.111:33181/
[rosmaster.master][INFO] 2023-01-16 14:42:04,320: +PUB [/tf] /odom_tf http://192.168.111.111:33181/
[rosmaster.master][INFO] 2023-01-16 14:42:04,327: +SUB [/tf] /odom_tf http://192.168.111.111:33181/
[rosmaster.master][INFO] 2023-01-16 14:42:04,341: +SUB [/tf_static] /odom_tf http://192.168.111.111:33181/
[rosmaster.master][INFO] 2023-01-16 14:42:04,347: +SUB [/odom] /odom_tf http://192.168.111.111:33181/
[rosmaster.master][INFO] 2023-01-16 14:42:04,347: publisherUpdate[/rosout] -> http://192.168.111.111:38431/ ['http://192.168.111.111:41129/', 'http://192.168.111.111:43717/', 'http://192.168.111.111:45843/', 'http://192.168.111.111:39961/', 'http://192.168.111.111:39017/', 'http://192.168.111.111:41995/', 'http://192.168.111.111:42095/', 'http://tegra-ubuntu:32835/', 'http://192.168.111.111:45739/', 'http://192.168.111.111:41973/', 'http://192.168.111.111:40973/', 'http://192.168.111.111:34787/', 'http://192.168.111.111:33181/']
[rosmaster.master][INFO] 2023-01-16 14:42:04,355: publisherUpdate[/rosout] -> http://192.168.111.111:38431/ ['http://192.168.111.111:41129/', 'http://192.168.111.111:43717/', 'http://192.168.111.111:45843/', 'http://192.168.111.111:39961/', 'http://192.168.111.111:39017/', 'http://192.168.111.111:41995/', 'http://192.168.111.111:42095/', 'http://tegra-ubuntu:32835/', 'http://192.168.111.111:45739/', 'http://192.168.111.111:41973/', 'http://192.168.111.111:40973/', 'http://192.168.111.111:34787/', 'http://192.168.111.111:33181/']: sec=0.01, result=[1, '', 0]
[rosmaster.master][INFO] 2023-01-16 14:42:04,356: publisherUpdate[/tf] -> http://192.168.111.111:39961/ ['http://192.168.111.111:43717/', 'http://192.168.111.111:33181/']
[rosmaster.master][INFO] 2023-01-16 14:42:04,357: publisherUpdate[/tf] -> http://192.168.111.111:39961/ ['http://192.168.111.111:43717/', 'http://192.168.111.111:33181/']: sec=0.00, result=[1, '', 0]
[rosmaster.master][INFO] 2023-01-16 14:42:04,358: publisherUpdate[/tf] -> http://192.168.111.111:42095/ ['http://192.168.111.111:43717/', 'http://192.168.111.111:33181/']
[rosmaster.master][INFO] 2023-01-16 14:42:04,770: publisherUpdate[/tf] -> http://192.168.111.111:42095/ ['http://192.168.111.111:43717/', 'http://192.168.111.111:33181/']: sec=0.41, result=[1, '', 0]
[rosmaster.master][INFO] 2023-01-16 14:42:16,918: +SUB [/cur_pose] /StateCenter http://192.168.111.111:38243/
[rosmaster.master][INFO] 2023-01-16 14:42:16,922: +SUB [/robot_mode] /StateCenter http://192.168.111.111:38243/
[rosmaster.master][INFO] 2023-01-16 14:42:16,924: +SUB [/cold_start_status] /StateCenter http://192.168.111.111:38243/
[rosmaster.master][INFO] 2023-01-16 14:42:16,928: +SUB [/map_cloud] /StateCenter http://192.168.111.111:38243/
[rosmaster.master][INFO] 2023-01-16 14:42:16,931: +SUB [/scan_matched_points2] /StateCenter http://192.168.111.111:38243/
[rosmaster.master][INFO] 2023-01-16 14:42:16,934: +SUB [/barcode] /StateCenter http://192.168.111.111:38243/
[rosmaster.master][INFO] 2023-01-16 14:42:16,937: +SUB [/barcode_poses_list] /StateCenter http://192.168.111.111:38243/
[rosmaster.master][INFO] 2023-01-16 14:42:16,940: +SUB [/carto_pose_confidence] /StateCenter http://192.168.111.111:38243/
[rosmaster.master][INFO] 2023-01-16 14:42:16,943: +SUB [/ipu_pos] /StateCenter http://192.168.111.111:38243/
[rosmaster.master][INFO] 2023-01-16 14:42:16,946: +SUB [/landmarks] /StateCenter http://192.168.111.111:38243/
[rosmaster.master][INFO] 2023-01-16 14:42:16,949: +SUB [/map] /StateCenter http://192.168.111.111:38243/
[rosmaster.master][INFO] 2023-01-16 14:42:16,952: +SUB [/mapping_process] /StateCenter http://192.168.111.111:38243/
[rosmaster.master][INFO] 2023-01-16 14:42:16,955: +SUB [/dsp_pos] /StateCenter http://192.168.111.111:38243/
[rosmaster.master][INFO] 2023-01-16 14:42:16,957: +SUB [/reflector_pose_with_confidence] /StateCenter http://192.168.111.111:38243/
[rosmaster.master][INFO] 2023-01-16 14:42:16,961: +SUB [/landmark_poses_list] /StateCenter http://192.168.111.111:38243/
[rosmaster.master][INFO] 2023-01-16 14:42:17,015: +PUB [/rosout] /StateCenter http://192.168.111.111:38243/
[rosmaster.master][INFO] 2023-01-16 14:42:17,025: publisherUpdate[/rosout] -> http://192.168.111.111:38431/ ['http://192.168.111.111:41129/', 'http://192.168.111.111:43717/', 'http://192.168.111.111:45843/', 'http://192.168.111.111:39961/', 'http://192.168.111.111:39017/', 'http://192.168.111.111:41995/', 'http://192.168.111.111:42095/', 'http://tegra-ubuntu:32835/', 'http://192.168.111.111:45739/', 'http://192.168.111.111:41973/', 'http://192.168.111.111:40973/', 'http://192.168.111.111:34787/', 'http://192.168.111.111:33181/', 'http://192.168.111.111:38243/']
[rosmaster.master][INFO] 2023-01-16 14:42:17,028: publisherUpdate[/rosout] -> http://192.168.111.111:38431/ ['http://192.168.111.111:41129/', 'http://192.168.111.111:43717/', 'http://192.168.111.111:45843/', 'http://192.168.111.111:39961/', 'http://192.168.111.111:39017/', 'http://192.168.111.111:41995/', 'http://192.168.111.111:42095/', 'http://tegra-ubuntu:32835/', 'http://192.168.111.111:45739/', 'http://192.168.111.111:41973/', 'http://192.168.111.111:40973/', 'http://192.168.111.111:34787/', 'http://192.168.111.111:33181/', 'http://192.168.111.111:38243/']: sec=0.00, result=[1, '', 0]
[rosmaster.master][INFO] 2023-01-16 14:42:17,029: +SERVICE [/StateCenter/get_loggers] /StateCenter http://192.168.111.111:38243/
[rosmaster.master][INFO] 2023-01-16 14:42:17,032: +SERVICE [/StateCenter/set_logger_level] /StateCenter http://192.168.111.111:38243/
[rosmaster.master][INFO] 2023-01-16 14:42:18,699: +PUB [/rosout] /UdpBarrier http://192.168.111.111:40901/
[rosmaster.master][INFO] 2023-01-16 14:42:18,707: +SERVICE [/UdpBarrier/get_loggers] /UdpBarrier http://192.168.111.111:40901/
[rosmaster.master][INFO] 2023-01-16 14:42:18,710: +SERVICE [/UdpBarrier/set_logger_level] /UdpBarrier http://192.168.111.111:40901/
[rosmaster.master][INFO] 2023-01-16 14:42:18,719: +SUB [/new_obstacles] /UdpBarrier http://192.168.111.111:40901/
[rosmaster.master][INFO] 2023-01-16 14:42:18,724: +SUB [/obstacle_detection_all_sensors/sensor_states] /UdpBarrier http://192.168.111.111:40901/
[rosmaster.master][INFO] 2023-01-16 14:42:18,743: publisherUpdate[/rosout] -> http://192.168.111.111:38431/ ['http://192.168.111.111:41129/', 'http://192.168.111.111:43717/', 'http://192.168.111.111:45843/', 'http://192.168.111.111:39961/', 'http://192.168.111.111:39017/', 'http://192.168.111.111:41995/', 'http://192.168.111.111:42095/', 'http://tegra-ubuntu:32835/', 'http://192.168.111.111:45739/', 'http://192.168.111.111:41973/', 'http://192.168.111.111:40973/', 'http://192.168.111.111:34787/', 'http://192.168.111.111:33181/', 'http://192.168.111.111:38243/', 'http://192.168.111.111:40901/']
[rosmaster.master][INFO] 2023-01-16 14:42:18,745: publisherUpdate[/rosout] -> http://192.168.111.111:38431/ ['http://192.168.111.111:41129/', 'http://192.168.111.111:43717/', 'http://192.168.111.111:45843/', 'http://192.168.111.111:39961/', 'http://192.168.111.111:39017/', 'http://192.168.111.111:41995/', 'http://192.168.111.111:42095/', 'http://tegra-ubuntu:32835/', 'http://192.168.111.111:45739/', 'http://192.168.111.111:41973/', 'http://192.168.111.111:40973/', 'http://192.168.111.111:34787/', 'http://192.168.111.111:33181/', 'http://192.168.111.111:38243/', 'http://192.168.111.111:40901/']: sec=0.00, result=[1, '', 0]
[rosmaster.master][INFO] 2023-01-16 14:42:25,158: +SUB [/error_status] /upper_computer_withWebUI http://192.168.111.111:42321/
[rosmaster.master][INFO] 2023-01-16 14:42:25,170: +SUB [/cold_start_status] /upper_computer_withWebUI http://192.168.111.111:42321/
[rosmaster.master][INFO] 2023-01-16 14:42:25,258: +PUB [/rosout] /upper_computer_withWebUI http://192.168.111.111:42321/
[rosmaster.master][INFO] 2023-01-16 14:42:25,278: publisherUpdate[/rosout] -> http://192.168.111.111:38431/ ['http://192.168.111.111:41129/', 'http://192.168.111.111:43717/', 'http://192.168.111.111:45843/', 'http://192.168.111.111:39961/', 'http://192.168.111.111:39017/', 'http://192.168.111.111:41995/', 'http://192.168.111.111:42095/', 'http://tegra-ubuntu:32835/', 'http://192.168.111.111:45739/', 'http://192.168.111.111:41973/', 'http://192.168.111.111:40973/', 'http://192.168.111.111:34787/', 'http://192.168.111.111:33181/', 'http://192.168.111.111:38243/', 'http://192.168.111.111:40901/', 'http://192.168.111.111:42321/']
[rosmaster.master][INFO] 2023-01-16 14:42:25,280: publisherUpdate[/rosout] -> http://192.168.111.111:38431/ ['http://192.168.111.111:41129/', 'http://192.168.111.111:43717/', 'http://192.168.111.111:45843/', 'http://192.168.111.111:39961/', 'http://192.168.111.111:39017/', 'http://192.168.111.111:41995/', 'http://192.168.111.111:42095/', 'http://tegra-ubuntu:32835/', 'http://192.168.111.111:45739/', 'http://192.168.111.111:41973/', 'http://192.168.111.111:40973/', 'http://192.168.111.111:34787/', 'http://192.168.111.111:33181/', 'http://192.168.111.111:38243/', 'http://192.168.111.111:40901/', 'http://192.168.111.111:42321/']: sec=0.00, result=[1, '', 0]
[rosmaster.master][INFO] 2023-01-16 14:42:25,301: +SERVICE [/upper_computer_withWebUI/get_loggers] /upper_computer_withWebUI http://192.168.111.111:42321/
[rosmaster.master][INFO] 2023-01-16 14:42:25,313: +SERVICE [/upper_computer_withWebUI/set_logger_level] /upper_computer_withWebUI http://192.168.111.111:42321/
[rosmaster.master][INFO] 2023-01-16 14:42:25,318: +SUB [/error_msg] /upper_computer_withWebUI http://192.168.111.111:42321/
[rosmaster.master][INFO] 2023-01-16 14:42:28,033: +PUB [/rosout] /Communication http://192.168.111.111:42289/
[rosmaster.master][INFO] 2023-01-16 14:42:28,039: +SERVICE [/Communication/get_loggers] /Communication http://192.168.111.111:42289/
[rosmaster.master][INFO] 2023-01-16 14:42:28,042: +SERVICE [/Communication/set_logger_level] /Communication http://192.168.111.111:42289/
[rosmaster.master][INFO] 2023-01-16 14:42:28,043: publisherUpdate[/rosout] -> http://192.168.111.111:38431/ ['http://192.168.111.111:41129/', 'http://192.168.111.111:43717/', 'http://192.168.111.111:45843/', 'http://192.168.111.111:39961/', 'http://192.168.111.111:39017/', 'http://192.168.111.111:41995/', 'http://192.168.111.111:42095/', 'http://tegra-ubuntu:32835/', 'http://192.168.111.111:45739/', 'http://192.168.111.111:41973/', 'http://192.168.111.111:40973/', 'http://192.168.111.111:34787/', 'http://192.168.111.111:33181/', 'http://192.168.111.111:38243/', 'http://192.168.111.111:40901/', 'http://192.168.111.111:42321/', 'http://192.168.111.111:42289/']
[rosmaster.master][INFO] 2023-01-16 14:42:28,045: publisherUpdate[/rosout] -> http://192.168.111.111:38431/ ['http://192.168.111.111:41129/', 'http://192.168.111.111:43717/', 'http://192.168.111.111:45843/', 'http://192.168.111.111:39961/', 'http://192.168.111.111:39017/', 'http://192.168.111.111:41995/', 'http://192.168.111.111:42095/', 'http://tegra-ubuntu:32835/', 'http://192.168.111.111:45739/', 'http://192.168.111.111:41973/', 'http://192.168.111.111:40973/', 'http://192.168.111.111:34787/', 'http://192.168.111.111:33181/', 'http://192.168.111.111:38243/', 'http://192.168.111.111:40901/', 'http://192.168.111.111:42321/', 'http://192.168.111.111:42289/']: sec=0.00, result=[1, '', 0]
[rosmaster.master][INFO] 2023-01-16 14:42:28,219: +SERVICE [/RcsOnline] /Communication http://192.168.111.111:42289/
[rosmaster.master][INFO] 2023-01-16 14:42:28,221: +SERVICE [/RcsOffline] /Communication http://192.168.111.111:42289/
[rosmaster.master][INFO] 2023-01-16 14:59:59,187: +PARAM [/roslaunch/uris/host_172_31_242_250__35161] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:59:59,194: +PARAM [/rosversion] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:59:59,194: +PARAM [/rosdistro] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:59:59,195: +PARAM [/obstacle_detection_all_sensors/use_slam_pos] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 14:59:59,195: +PARAM [/obstacle_detection_all_sensors/filter_distance] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 15:00:00,624: +SERVICE [/obstacle_detection_all_sensors/get_loggers] /obstacle_detection_all_sensors http://172.31.242.250:38867/
[rosmaster.master][INFO] 2023-01-16 15:00:00,626: +SERVICE [/obstacle_detection_all_sensors/set_logger_level] /obstacle_detection_all_sensors http://172.31.242.250:38867/
[rosmaster.master][INFO] 2023-01-16 15:00:00,628: +SERVICE [/obstacle_detection_all_sensors/barrier_operation] /obstacle_detection_all_sensors http://172.31.242.250:38867/
[rosmaster.master][INFO] 2023-01-16 15:00:00,630: +PUB [/obstacle_detection_all_sensors/sensor_states] /obstacle_detection_all_sensors http://172.31.242.250:38867/
[rosmaster.master][INFO] 2023-01-16 15:00:00,633: publisherUpdate[/obstacle_detection_all_sensors/sensor_states] -> http://192.168.111.111:40901/ ['http://172.31.242.250:38867/']
[rosmaster.master][INFO] 2023-01-16 15:00:00,637: publisherUpdate[/obstacle_detection_all_sensors/sensor_states] -> http://192.168.111.111:40901/ ['http://172.31.242.250:38867/']: sec=0.00, result=[1, '', 0]
[rosmaster.master][INFO] 2023-01-16 15:00:19,342: +PUB [/rosout] /barrier_operation_client http://172.31.242.250:42379/
[rosmaster.master][INFO] 2023-01-16 15:00:19,344: +SERVICE [/barrier_operation_client/get_loggers] /barrier_operation_client http://172.31.242.250:42379/
[rosmaster.master][INFO] 2023-01-16 15:00:19,346: +SERVICE [/barrier_operation_client/set_logger_level] /barrier_operation_client http://172.31.242.250:42379/
[rosmaster.master][INFO] 2023-01-16 15:00:19,359: publisherUpdate[/rosout] -> http://192.168.111.111:38431/ ['http://192.168.111.111:41129/', 'http://192.168.111.111:43717/', 'http://192.168.111.111:45843/', 'http://192.168.111.111:39961/', 'http://192.168.111.111:39017/', 'http://192.168.111.111:41995/', 'http://192.168.111.111:42095/', 'http://tegra-ubuntu:32835/', 'http://192.168.111.111:45739/', 'http://192.168.111.111:41973/', 'http://192.168.111.111:40973/', 'http://192.168.111.111:34787/', 'http://192.168.111.111:33181/', 'http://192.168.111.111:38243/', 'http://192.168.111.111:40901/', 'http://192.168.111.111:42321/', 'http://192.168.111.111:42289/', 'http://172.31.242.250:42379/']
[rosmaster.master][INFO] 2023-01-16 15:00:19,359: +PUB [/new_obstacles] /obstacle_detection_all_sensors http://172.31.242.250:38867/
[rosmaster.master][INFO] 2023-01-16 15:00:19,362: publisherUpdate[/rosout] -> http://192.168.111.111:38431/ ['http://192.168.111.111:41129/', 'http://192.168.111.111:43717/', 'http://192.168.111.111:45843/', 'http://192.168.111.111:39961/', 'http://192.168.111.111:39017/', 'http://192.168.111.111:41995/', 'http://192.168.111.111:42095/', 'http://tegra-ubuntu:32835/', 'http://192.168.111.111:45739/', 'http://192.168.111.111:41973/', 'http://192.168.111.111:40973/', 'http://192.168.111.111:34787/', 'http://192.168.111.111:33181/', 'http://192.168.111.111:38243/', 'http://192.168.111.111:40901/', 'http://192.168.111.111:42321/', 'http://192.168.111.111:42289/', 'http://172.31.242.250:42379/']: sec=0.00, result=[1, '', 0]
[rosmaster.master][INFO] 2023-01-16 15:00:19,362: +PUB [/oa_box] /obstacle_detection_all_sensors http://172.31.242.250:38867/
[rosmaster.master][INFO] 2023-01-16 15:00:19,363: publisherUpdate[/new_obstacles] -> http://192.168.111.111:40901/ ['http://172.31.242.250:38867/']
[rosmaster.master][INFO] 2023-01-16 15:00:19,364: +PUB [/oa_concave] /obstacle_detection_all_sensors http://172.31.242.250:38867/
[rosmaster.master][INFO] 2023-01-16 15:00:19,366: publisherUpdate[/new_obstacles] -> http://192.168.111.111:40901/ ['http://172.31.242.250:38867/']: sec=0.00, result=[1, '', 0]
[rosmaster.master][INFO] 2023-01-16 15:00:19,367: +PUB [/error_msg] /obstacle_detection_all_sensors http://172.31.242.250:38867/
[rosmaster.master][INFO] 2023-01-16 15:00:19,367: publisherUpdate[/error_msg] -> http://192.168.111.111:42321/ ['http://172.31.242.250:38867/']
[rosmaster.master][INFO] 2023-01-16 15:00:19,369: +PUB [/obstacle_detection_all_sensors/raw_cloud] /obstacle_detection_all_sensors http://172.31.242.250:38867/
[rosmaster.master][INFO] 2023-01-16 15:00:19,370: +PUB [/obstacle_detection_all_sensors/area_filter_cloud] /obstacle_detection_all_sensors http://172.31.242.250:38867/
[rosmaster.master][INFO] 2023-01-16 15:00:19,371: publisherUpdate[/error_msg] -> http://192.168.111.111:42321/ ['http://172.31.242.250:38867/']: sec=0.00, result=[1, '', 0]
[rosmaster.master][INFO] 2023-01-16 15:00:19,372: +PUB [/obstacles_list] /obstacle_detection_all_sensors http://172.31.242.250:38867/
[rosmaster.master][INFO] 2023-01-16 15:00:19,379: +SUB [/front/scan] /obstacle_detection_all_sensors http://172.31.242.250:38867/
[rosmaster.master][INFO] 2023-01-16 15:00:21,567: -PUB [/rosout] /barrier_operation_client http://172.31.242.250:42379/
[rosmaster.master][INFO] 2023-01-16 15:00:21,569: -SERVICE [/barrier_operation_client/get_loggers] /barrier_operation_client rosrpc://172.31.242.250:48941
[rosmaster.master][INFO] 2023-01-16 15:00:21,570: -SERVICE [/barrier_operation_client/set_logger_level] /barrier_operation_client rosrpc://172.31.242.250:48941
[rosmaster.master][INFO] 2023-01-16 15:00:21,577: publisherUpdate[/rosout] -> http://192.168.111.111:38431/ ['http://192.168.111.111:41129/', 'http://192.168.111.111:43717/', 'http://192.168.111.111:45843/', 'http://192.168.111.111:39961/', 'http://192.168.111.111:39017/', 'http://192.168.111.111:41995/', 'http://192.168.111.111:42095/', 'http://tegra-ubuntu:32835/', 'http://192.168.111.111:45739/', 'http://192.168.111.111:41973/', 'http://192.168.111.111:40973/', 'http://192.168.111.111:34787/', 'http://192.168.111.111:33181/', 'http://192.168.111.111:38243/', 'http://192.168.111.111:40901/', 'http://192.168.111.111:42321/', 'http://192.168.111.111:42289/']
[rosmaster.master][INFO] 2023-01-16 15:00:21,579: publisherUpdate[/rosout] -> http://192.168.111.111:38431/ ['http://192.168.111.111:41129/', 'http://192.168.111.111:43717/', 'http://192.168.111.111:45843/', 'http://192.168.111.111:39961/', 'http://192.168.111.111:39017/', 'http://192.168.111.111:41995/', 'http://192.168.111.111:42095/', 'http://tegra-ubuntu:32835/', 'http://192.168.111.111:45739/', 'http://192.168.111.111:41973/', 'http://192.168.111.111:40973/', 'http://192.168.111.111:34787/', 'http://192.168.111.111:33181/', 'http://192.168.111.111:38243/', 'http://192.168.111.111:40901/', 'http://192.168.111.111:42321/', 'http://192.168.111.111:42289/']: sec=0.00, result=[1, '', 0]
[rosmaster.master][INFO] 2023-01-16 15:19:44,197: +PARAM [/roslaunch/uris/host_172_31_242_250__42507] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 15:19:44,205: +PARAM [/rosversion] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 15:19:44,205: +PARAM [/rosdistro] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 15:19:44,206: +PARAM [/obstacle_detection_all_sensors/use_slam_pos] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 15:19:44,206: +PARAM [/obstacle_detection_all_sensors/filter_distance] by /roslaunch
[rosmaster.master][INFO] 2023-01-16 15:19:45,090: +SERVICE [/obstacle_detection_all_sensors/get_loggers] /obstacle_detection_all_sensors http://172.31.242.250:38283/
[rosmaster.master][INFO] 2023-01-16 15:19:45,092: +SERVICE [/obstacle_detection_all_sensors/set_logger_level] /obstacle_detection_all_sensors http://172.31.242.250:38283/
[rosmaster.master][INFO] 2023-01-16 15:19:45,094: +SERVICE [/obstacle_detection_all_sensors/barrier_operation] /obstacle_detection_all_sensors http://172.31.242.250:38283/
[rosmaster.master][INFO] 2023-01-16 15:19:45,096: +PUB [/obstacle_detection_all_sensors/sensor_states] /obstacle_detection_all_sensors http://172.31.242.250:38283/
[rosmaster.master][INFO] 2023-01-16 15:19:45,108: publisherUpdate[/obstacle_detection_all_sensors/sensor_states] -> http://192.168.111.111:40901/ ['http://172.31.242.250:38283/']
[rosmaster.master][INFO] 2023-01-16 15:19:45,111: publisherUpdate[/obstacle_detection_all_sensors/sensor_states] -> http://192.168.111.111:40901/ ['http://172.31.242.250:38283/']: sec=0.00, result=[1, '', 0]
[rosmaster.master][INFO] 2023-01-16 15:19:50,471: +PUB [/rosout] /barrier_operation_client http://172.31.242.250:32811/
[rosmaster.master][INFO] 2023-01-16 15:19:50,474: +SERVICE [/barrier_operation_client/get_loggers] /barrier_operation_client http://172.31.242.250:32811/
[rosmaster.master][INFO] 2023-01-16 15:19:50,475: +SERVICE [/barrier_operation_client/set_logger_level] /barrier_operation_client http://172.31.242.250:32811/
[rosmaster.master][INFO] 2023-01-16 15:19:50,486: +PUB [/new_obstacles] /obstacle_detection_all_sensors http://172.31.242.250:38283/
[rosmaster.master][INFO] 2023-01-16 15:19:50,488: +PUB [/oa_box] /obstacle_detection_all_sensors http://172.31.242.250:38283/
[rosmaster.master][INFO] 2023-01-16 15:19:50,489: +PUB [/oa_concave] /obstacle_detection_all_sensors http://172.31.242.250:38283/
[rosmaster.master][INFO] 2023-01-16 15:19:50,490: +PUB [/error_msg] /obstacle_detection_all_sensors http://172.31.242.250:38283/
[rosmaster.master][INFO] 2023-01-16 15:19:50,491: +PUB [/obstacle_detection_all_sensors/raw_cloud] /obstacle_detection_all_sensors http://172.31.242.250:38283/
[rosmaster.master][INFO] 2023-01-16 15:19:50,492: +PUB [/obstacle_detection_all_sensors/area_filter_cloud] /obstacle_detection_all_sensors http://172.31.242.250:38283/
[rosmaster.master][INFO] 2023-01-16 15:19:50,493: +PUB [/obstacles_list] /obstacle_detection_all_sensors http://172.31.242.250:38283/
[rosmaster.master][INFO] 2023-01-16 15:19:50,501: +SUB [/front/scan] /obstacle_detection_all_sensors http://172.31.242.250:38283/
[rosmaster.master][INFO] 2023-01-16 15:19:50,522: publisherUpdate[/rosout] -> http://192.168.111.111:38431/ ['http://192.168.111.111:41129/', 'http://192.168.111.111:43717/', 'http://192.168.111.111:45843/', 'http://192.168.111.111:39961/', 'http://192.168.111.111:39017/', 'http://192.168.111.111:41995/', 'http://192.168.111.111:42095/', 'http://tegra-ubuntu:32835/', 'http://192.168.111.111:45739/', 'http://192.168.111.111:41973/', 'http://192.168.111.111:40973/', 'http://192.168.111.111:34787/', 'http://192.168.111.111:33181/', 'http://192.168.111.111:38243/', 'http://192.168.111.111:40901/', 'http://192.168.111.111:42321/', 'http://192.168.111.111:42289/', 'http://172.31.242.250:32811/']
[rosmaster.master][INFO] 2023-01-16 15:19:50,525: publisherUpdate[/rosout] -> http://192.168.111.111:38431/ ['http://192.168.111.111:41129/', 'http://192.168.111.111:43717/', 'http://192.168.111.111:45843/', 'http://192.168.111.111:39961/', 'http://192.168.111.111:39017/', 'http://192.168.111.111:41995/', 'http://192.168.111.111:42095/', 'http://tegra-ubuntu:32835/', 'http://192.168.111.111:45739/', 'http://192.168.111.111:41973/', 'http://192.168.111.111:40973/', 'http://192.168.111.111:34787/', 'http://192.168.111.111:33181/', 'http://192.168.111.111:38243/', 'http://192.168.111.111:40901/', 'http://192.168.111.111:42321/', 'http://192.168.111.111:42289/', 'http://172.31.242.250:32811/']: sec=0.00, result=[1, '', 0]
[rosmaster.master][INFO] 2023-01-16 15:19:50,525: publisherUpdate[/new_obstacles] -> http://192.168.111.111:40901/ ['http://172.31.242.250:38283/']
[rosmaster.master][INFO] 2023-01-16 15:19:50,528: publisherUpdate[/new_obstacles] -> http://192.168.111.111:40901/ ['http://172.31.242.250:38283/']: sec=0.00, result=[1, '', 0]
[rosmaster.master][INFO] 2023-01-16 15:19:50,528: publisherUpdate[/error_msg] -> http://192.168.111.111:42321/ ['http://172.31.242.250:38283/']
[rosmaster.master][INFO] 2023-01-16 15:19:50,531: publisherUpdate[/error_msg] -> http://192.168.111.111:42321/ ['http://172.31.242.250:38283/']: sec=0.00, result=[1, '', 0]
[rosmaster.master][INFO] 2023-01-16 15:19:52,633: -PUB [/rosout] /barrier_operation_client http://172.31.242.250:32811/
[rosmaster.master][INFO] 2023-01-16 15:19:52,635: -SERVICE [/barrier_operation_client/get_loggers] /barrier_operation_client rosrpc://172.31.242.250:41817
[rosmaster.master][INFO] 2023-01-16 15:19:52,636: -SERVICE [/barrier_operation_client/set_logger_level] /barrier_operation_client rosrpc://172.31.242.250:41817
[rosmaster.master][INFO] 2023-01-16 15:19:52,639: publisherUpdate[/rosout] -> http://192.168.111.111:38431/ ['http://192.168.111.111:41129/', 'http://192.168.111.111:43717/', 'http://192.168.111.111:45843/', 'http://192.168.111.111:39961/', 'http://192.168.111.111:39017/', 'http://192.168.111.111:41995/', 'http://192.168.111.111:42095/', 'http://tegra-ubuntu:32835/', 'http://192.168.111.111:45739/', 'http://192.168.111.111:41973/', 'http://192.168.111.111:40973/', 'http://192.168.111.111:34787/', 'http://192.168.111.111:33181/', 'http://192.168.111.111:38243/', 'http://192.168.111.111:40901/', 'http://192.168.111.111:42321/', 'http://192.168.111.111:42289/']
[rosmaster.master][INFO] 2023-01-16 15:19:52,641: publisherUpdate[/rosout] -> http://192.168.111.111:38431/ ['http://192.168.111.111:41129/', 'http://192.168.111.111:43717/', 'http://192.168.111.111:45843/', 'http://192.168.111.111:39961/', 'http://192.168.111.111:39017/', 'http://192.168.111.111:41995/', 'http://192.168.111.111:42095/', 'http://tegra-ubuntu:32835/', 'http://192.168.111.111:45739/', 'http://192.168.111.111:41973/', 'http://192.168.111.111:40973/', 'http://192.168.111.111:34787/', 'http://192.168.111.111:33181/', 'http://192.168.111.111:38243/', 'http://192.168.111.111:40901/', 'http://192.168.111.111:42321/', 'http://192.168.111.111:42289/']: sec=0.00, result=[1, '', 0]
```

## roslaunch 日志

```text
[roslaunch][INFO] 2023-01-16 14:40:50,148: Checking log directory for disk usage. This may take awhile.
Press Ctrl-C to interrupt
[roslaunch][INFO] 2023-01-16 14:40:50,170: Done checking log file disk usage. Usage is <1GB.
[roslaunch][INFO] 2023-01-16 14:40:50,171: roslaunch starting with args ['/opt/ros/melodic/bin/roslaunch', '/set_urdf.launch', '--wait']
[roslaunch][INFO] 2023-01-16 14:40:50,171: roslaunch env is {'ROS_DISTRO': 'melodic', 'ROSLISP_PACKAGE_DIRECTORIES': '', 'HOME': '/root', 'PATH': '/opt/ros/melodic/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin', 'ROS_PACKAGE_PATH': '/opt/ros/melodic/share', 'CMAKE_PREFIX_PATH': '/opt/ros/melodic', 'LD_LIBRARY_PATH': '/opt/ros/melodic/lib', 'LANG': 'C.UTF-8', 'TERM': 'xterm', 'SHLVL': '1', 'ROS_LOG_FILENAME': '/root/.ros/log/b598afc4-9568-11ed-b4d9-00044bde2742/roslaunch-tegra-ubuntu-30.log', 'ROS_MASTER_URI': 'http://192.168.111.111:11311', 'ROS_PYTHON_VERSION': '2', 'PYTHONPATH': '/opt/ros/melodic/lib/python2.7/dist-packages', 'ROS_ROOT': '/opt/ros/melodic/share/ros', 'PKG_CONFIG_PATH': '/opt/ros/melodic/lib/pkgconfig', 'LC_ALL': 'C.UTF-8', '_': '/usr/bin/nohup', 'HOSTNAME': 'tegra-ubuntu', 'ROS_IP': '192.168.111.111', 'PWD': '/', 'ROS_ETC_DIR': '/opt/ros/melodic/etc/ros', 'ROS_VERSION': '1'}
[roslaunch][INFO] 2023-01-16 14:40:50,171: starting in server mode
[roslaunch.parent][INFO] 2023-01-16 14:40:50,172: starting roslaunch parent run
[roslaunch][INFO] 2023-01-16 14:40:50,172: loading roscore config file /opt/ros/melodic/etc/ros/roscore.xml
[roslaunch][INFO] 2023-01-16 14:40:51,093: Added core node of type [rosout/rosout] in namespace [/]
[roslaunch.config][INFO] 2023-01-16 14:40:51,093: loading config file /set_urdf.launch
[roslaunch][INFO] 2023-01-16 14:40:51,118: Added node of type [joint_state_publisher/joint_state_publisher] in namespace [/]
[roslaunch][INFO] 2023-01-16 14:40:51,119: Added node of type [robot_state_publisher/robot_state_publisher] in namespace [/]
[roslaunch][INFO] 2023-01-16 14:40:51,120: ... selected machine [] for node of type [joint_state_publisher/joint_state_publisher]
[roslaunch][INFO] 2023-01-16 14:40:51,120: ... selected machine [] for node of type [robot_state_publisher/robot_state_publisher]
[roslaunch.pmon][INFO] 2023-01-16 14:40:51,126: start_process_monitor: creating ProcessMonitor
[roslaunch.pmon][INFO] 2023-01-16 14:40:51,126: created process monitor <ProcessMonitor(ProcessMonitor-1, initial daemon)>
[roslaunch.pmon][INFO] 2023-01-16 14:40:51,128: start_process_monitor: ProcessMonitor started
[roslaunch.parent][INFO] 2023-01-16 14:40:51,128: starting parent XML-RPC server
[roslaunch.server][INFO] 2023-01-16 14:40:51,128: starting roslaunch XML-RPC server
[roslaunch.server][INFO] 2023-01-16 14:40:51,129: waiting for roslaunch XML-RPC server to initialize
[xmlrpc][INFO] 2023-01-16 14:40:51,129: XML-RPC server binding to 0.0.0.0:0
[xmlrpc][INFO] 2023-01-16 14:40:51,130: Started XML-RPC server [http://192.168.111.111:33083/]
[xmlrpc][INFO] 2023-01-16 14:40:51,131: xml rpc node: starting XML-RPC server
[roslaunch][INFO] 2023-01-16 14:40:51,143: started roslaunch server http://192.168.111.111:33083/
[roslaunch.parent][INFO] 2023-01-16 14:40:51,144: ... parent XML-RPC server started
[roslaunch][INFO] 2023-01-16 14:40:51,145: master.is_running[http://192.168.111.111:11311]
[roslaunch][INFO] 2023-01-16 14:40:51,148: master.is_running[http://192.168.111.111:11311]
[roslaunch][INFO] 2023-01-16 14:40:51,151: ROS_MASTER_URI=http://192.168.111.111:11311
[roslaunch][INFO] 2023-01-16 14:40:51,156: setting /roslaunch/uris/host_192_168_111_111__33083' to http://192.168.111.111:33083/
[roslaunch][INFO] 2023-01-16 14:40:51,159: load_parameters starting ...
[roslaunch][INFO] 2023-01-16 14:40:51,167: ... load_parameters complete
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
[roslaunch][INFO] 2023-01-16 14:40:51,959: create_node_process: package[robot_state_publisher] type[robot_state_publisher] machine[Machine(name[] env_loader[None] address[localhost] ssh_port[22] user[None] assignable[True] timeout[10.0])] master_uri[http://192.168.111.111:11311]
[roslaunch][INFO] 2023-01-16 14:40:51,959: process[robot_state_publisher-2]: env[{'ROS_DISTRO': 'melodic', 'ROS_IP': '192.168.111.111', 'ROS_PACKAGE_PATH': '/opt/ros/melodic/share', 'PATH': '/opt/ros/melodic/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin', 'CMAKE_PREFIX_PATH': '/opt/ros/melodic', 'ROS_LOG_FILENAME': '/root/.ros/log/b598afc4-9568-11ed-b4d9-00044bde2742/roslaunch-tegra-ubuntu-30.log', 'LANG': 'C.UTF-8', 'TERM': 'xterm', 'SHLVL': '1', 'LD_LIBRARY_PATH': '/opt/ros/melodic/lib', 'ROS_MASTER_URI': 'http://192.168.111.111:11311', 'HOME': '/root', 'ROS_PYTHON_VERSION': '2', 'PYTHONPATH': '/opt/ros/melodic/lib/python2.7/dist-packages', 'ROS_ROOT': '/opt/ros/melodic/share/ros', 'PKG_CONFIG_PATH': '/opt/ros/melodic/lib/pkgconfig', 'LC_ALL': 'C.UTF-8', '_': '/usr/bin/nohup', 'HOSTNAME': 'tegra-ubuntu', 'ROSLISP_PACKAGE_DIRECTORIES': '', 'PWD': '/', 'ROS_ETC_DIR': '/opt/ros/melodic/etc/ros', 'ROS_VERSION': '1'}]
[roslaunch][INFO] 2023-01-16 14:40:51,963: process[robot_state_publisher-2]: args[[u'/opt/ros/melodic/lib/robot_state_publisher/robot_state_publisher', u'__name:=robot_state_publisher']]
[roslaunch][INFO] 2023-01-16 14:40:51,964: ... created process [robot_state_publisher-2]
[roslaunch.pmon][INFO] 2023-01-16 14:40:51,964: ProcessMonitor.register[robot_state_publisher-2]
[roslaunch.pmon][INFO] 2023-01-16 14:40:51,964: ProcessMonitor.register[robot_state_publisher-2] complete
[roslaunch][INFO] 2023-01-16 14:40:51,965: ... registered process [robot_state_publisher-2]
[roslaunch][INFO] 2023-01-16 14:40:51,965: process[robot_state_publisher-2]: starting os process
[roslaunch][INFO] 2023-01-16 14:40:51,965: process[robot_state_publisher-2]: start w/ args [[u'/opt/ros/melodic/lib/robot_state_publisher/robot_state_publisher', u'__name:=robot_state_publisher', u'__log:=/root/.ros/log/b598afc4-9568-11ed-b4d9-00044bde2742/robot_state_publisher-2.log']]
[roslaunch][INFO] 2023-01-16 14:40:51,966: process[robot_state_publisher-2]: cwd will be [/root/.ros]
[roslaunch][INFO] 2023-01-16 14:40:52,695: process[robot_state_publisher-2]: started with pid [79]
[roslaunch][INFO] 2023-01-16 14:40:52,695: ... successfully launched [robot_state_publisher-2]
[roslaunch][INFO] 2023-01-16 14:40:52,696: ... launch_nodes complete
[roslaunch.pmon][INFO] 2023-01-16 14:40:52,696: registrations completed <ProcessMonitor(ProcessMonitor-1, started daemon 548206273008)>
[roslaunch.parent][INFO] 2023-01-16 14:40:52,696: ... roslaunch parent running, waiting for process exit
[roslaunch][INFO] 2023-01-16 14:40:52,696: spin
```

## joint_state_publisher 节点日志

```text
[rospy.client][INFO] 2023-01-16 14:40:52,427: init_node, name[/joint_state_publisher], pid[78]
[xmlrpc][INFO] 2023-01-16 14:40:52,428: XML-RPC server binding to 0.0.0.0:0
[xmlrpc][INFO] 2023-01-16 14:40:52,429: Started XML-RPC server [http://192.168.111.111:41129/]
[rospy.init][INFO] 2023-01-16 14:40:52,429: ROS Slave URI: [http://192.168.111.111:41129/]
[rospy.impl.masterslave][INFO] 2023-01-16 14:40:52,430: _ready: http://192.168.111.111:41129/
[xmlrpc][INFO] 2023-01-16 14:40:52,432: xml rpc node: starting XML-RPC server
[rospy.registration][INFO] 2023-01-16 14:40:52,431: Registering with master node http://192.168.111.111:11311
[rospy.init][INFO] 2023-01-16 14:40:52,530: registered with master
[rospy.rosout][INFO] 2023-01-16 14:40:52,531: initializing /rosout core topic
[rospy.rosout][INFO] 2023-01-16 14:40:52,535: connected to core topic /rosout
[rospy.simtime][INFO] 2023-01-16 14:40:52,539: /use_sim_time is not set, will not subscribe to simulated time [/clock] topic
[rospy.internal][INFO] 2023-01-16 14:40:53,150: topic[/rosout] adding connection to [/rosout], count 0
[rospy.internal][INFO] 2023-01-16 14:40:53,274: topic[/joint_states] adding connection to [/robot_state_publisher], count 0
```