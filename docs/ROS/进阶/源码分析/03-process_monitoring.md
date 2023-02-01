###### datetime:2023/01/30 16:48

###### author:nzb

# 3、进程监控管理(process monitoring)

roslaunch中有个小功能类似于android
init进程中的service重启功能，如果该进程在创建时有respawn属性，则在该进程dead后需要将其重启起来，起到一个进程监控的作用，相关源码位于`ros\melodic\lib\python2.7\dist-packages\roslaunch\pmon.py`
。

下面分析下其主要功能

## 3.1、进程类

```python
class Process(object):
    """
    Basic process representation for L{ProcessMonitor}. Must be subclassed
    to provide actual start()/stop() implementations.

    Constructor *must* be called from the Python Main thread in order
    for signal handlers to register properly.
    """

    def __init__(self, package, name, args, env,
                 respawn=False, respawn_delay=0.0, required=False):
        # ①进程的属性，respawn为是否需要重启
        self.respawn = respawn
        self.respawn_delay = respawn_delay
        self.required = required
```

`Process()`类就是`ProcessMonitor()`所监控的进程需要去继承的基类，可以设置dead后是否需要重启属性。

通过调用`start_process_monitor()`函数可以启动一个`ProcessMonitor`线程

子类：`LocalProcess`、`DeadProcess`、`ChildROSLaunchProcess`

## 3.2、进程监控类

```python
class ProcessMonitor(Thread):
    def __init__(self, name="ProcessMonitor"):
        Thread.__init__(self, name=name)
        # ①所监控的进程
        self.procs = []
        # #885: ensure core procs
        self.core_procs = []

    def register(self, p):
        """
        Register process with L{ProcessMonitor}
        @param p: Process
        @type  p: L{Process}
        @raise RLException: if process with same name is already registered
        """
        logger.info("ProcessMonitor.register[%s]" % p.name)
        e = None
        with self.plock:
            if self.has_process(p.name):
                e = RLException("cannot add process with duplicate name '%s'" % p.name)
            elif self.is_shutdown:
                e = RLException("cannot add process [%s] after process monitor has been shut down" % p.name)
            else:  # ② 将进程注册到ProcessMonitor，即添加到procs 
                self.procs.append(p)

    # ③ProcessMonitor线程的线程函数
    def run(self):
        """
        thread routine of the process monitor. NOTE: you must still
        call mainthread_spin or mainthread_spin_once() from the main
        thread in order to pick up main thread work from the process
        monitor.
        """
        try:
            # don't let exceptions bomb thread, interferes with exit
            try:
                self._run()
            except:
                pass
        finally:
            self._post_run()

    # ④ProcessMonitor线程的线程函数的主体    
    def _run(self):
        """
        Internal run loop of ProcessMonitor
        """
        plock = self.plock
        dead = []
        respawn = []
        while not self.is_shutdown:  # while循环，pmon关闭开关
            with plock:  # copy self.procs
                procs = self.procs[:]
            # ...
            # 监控中的进程
            for p in procs:
                try:
                    if not p.is_alive():
                        exit_code_str = p.get_exit_description()
                        # ⑤ 这个进程是必须的，如果这个必须的进程dead掉了，pmon自己也关闭
                        # 将self.is_shutdown 设置为 True
                        # ros master 的就是必须的进程
                        if p.required:
                            self.is_shutdown = True
                        elif not p in respawn:
                            # ...
                            dead.append(p)
                        # ...

                except Exception as e:
                    # ...
                    dead.append(p)
            for d in dead:
                try:
                    # when should_respawn() returns 0.0, bool(0.0) evaluates to False
                    # work around this by checking if the return value is False
                    if d.should_respawn() is not False:
                        respawn.append(d)  # 添加到需要重启的列表
                    else:
                        self.unregister(d)
                        # stop process, don't accumulate errors
                        d.stop([])
                        # save process data to dead list
                        with plock:
                            self.dead_list.append(DeadProcess(d))
                except:
                    pass

            # dead check is to make sure that ProcessMonitor at least
            # waits until its had at least one process before exiting
            if self._registrations_complete and dead and not self.procs and not respawn:
                printlog("all processes on machine have died, roslaunch will exit")
                self.is_shutdown = True
            del dead[:]
            _respawn = []
            for r in respawn:
                try:
                    if self.is_shutdown:
                        break
                    if r.should_respawn() <= 0.0:
                        # stop process, don't accumulate errors
                        r.stop([])
                        # ⑥ 重启需要重启的进程，起到进程监控的作用。
                        r.start()
                    else:
                        # not ready yet, keep it around
                        _respawn.append(r)
                except:
                    pass
            respawn = _respawn
            time.sleep(0.1)  # yield thread
        # moved this to finally block of _post_run
        # self._post_run() #kill all processes

    # 通过上面代码发现，self.is_shutdown是pmon的关闭开关，当is_shutdown为True，则while循环退出，将会继续执行_post_run()，会杀掉所有的监控进程，不过有顺序，最后杀掉核心进程(core_procs)。
    def _post_run(self):
        logger.info("ProcessMonitor._post_run %s" % self)
        # this is already true entering, but go ahead and make sure
        self.is_shutdown = True
        # killall processes on run exit

        q = Queue()
        q.join()

        with self.plock:
            # make copy of core_procs for threadsafe usage
            core_procs = self.core_procs[:]
            logger.info("ProcessMonitor._post_run %s: remaining procs are %s" % (self, self.procs))

            # enqueue all non-core procs in reverse order for parallel kill
            # #526/885: ignore core procs
            [q.put(p) for p in reversed(self.procs) if not p in core_procs]

        # use 10 workers
        killers = []
        for i in range(10):
            t = _ProcessKiller(q, i)
            killers.append(t)
            t.start()

        # wait for workers to finish
        q.join()
        shutdown_errors = []

        # accumulate all the shutdown errors
        for t in killers:
            shutdown_errors.extend(t.errors)
        del killers[:]

        # #526/885: kill core procs last
        # we don't want to parallelize this as the master has to be last
        for p in reversed(core_procs):
            _kill_process(p, shutdown_errors)

        # delete everything except dead_list
        logger.info("ProcessMonitor exit: cleaning up data structures and signals")
        with self.plock:
            del core_procs[:]
            del self.procs[:]
            del self.core_procs[:]

        reacquire_signals = self.reacquire_signals
        if reacquire_signals:
            reacquire_signals.clear()
        logger.info("ProcessMonitor exit: pmon has shutdown")
        self.done = True

        if shutdown_errors:
            printerrlog("Shutdown errors:\n" + '\n'.join([" * %s" % e for e in shutdown_errors]))


```

## 3.3、启动流程

开启函数

```python
_pmons = []
_pmon_counter = 0
def start_process_monitor():
    global _pmon_counter
    if _shutting_down:
        #logger.error("start_process_monitor: cannot start new ProcessMonitor (shutdown initiated)")
        return None
    _pmon_counter += 1
    name = "ProcessMonitor-%s"%_pmon_counter
    logger.info("start_process_monitor: creating ProcessMonitor")
    process_monitor = ProcessMonitor(name)
    try:
        # prevent race condition with pmon_shutdown() being triggered
        # as we are starting a ProcessMonitor (i.e. user hits ctrl-C
        # during startup)
        _shutdown_lock.acquire()
        _pmons.append(process_monitor)
        process_monitor.start()
        logger.info("start_process_monitor: ProcessMonitor started")
    finally:
        _shutdown_lock.release()

    return process_monitor
```

- 脚本(`roscore`、`roslaunch`)执行，导入`roslaunch`，运行`main`函数
- 实例化`ROSLaunchParent()`
- 执行实例`start()` -> `self._start_infrastructure()` -> `self._start_pm()`
- 执行函数`start_process_monitor()`，返回`ProcessMonitor`的实例，并执行`start()`，该类继承`Thread`
    - `ProcessMonitor`的`run`方法注释说明了，我们必须在主线程中调用`ProcessMonitor`的`mainthread_spin()`或`mainthread_spin_once()`方法，才能让进程监控功能开启

- 执行`ROSLaunchParent()`实例的`spin()`方法
    - 执行`self.runner.spin()`，`runner`是`ROSLaunchRunner`的实例
    - `spin()`方法里面执行了`self.pm.mainthread_spin()`, `pm`为`ProcessMonitor`实例，对应了上面必须主线程启动
    
> 通过`pmon.py`的代码分析，`pmon.py`肯定是在一个进程的主线程中去`import`，调用`start_process_monitor()`函数就会产生一个`pmon`，然后把需要监控的进程(线程)
注册到`pmon`中，主线程会有多个`pmon`保存在全局`_pmons = []`中。
> 

> **答疑**：为什么必须调用`ProcessMonitor`的`mainthread_spin()`或`mainthread_spin_once()`方法？
> mainthread_spin()函数注释：run() occurs in a separate thread and cannot do certain signal-related
        work. The main thread of the application must call mainthread_spin()
        or mainthread_spin_once() in order to perform these jobs. mainthread_spin()
        blocks until the process monitor is complete.  
> 
> 因为`run()`函数是一个单独的线程，用于监控进程，不能执行信号相关的操作，所以需要再主线中调用这2个方法，来执行信号相关的操作