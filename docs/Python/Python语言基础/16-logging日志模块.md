###### datetime:2021/11/14 17:55

###### author:nzb

## logging 模块

基于 logging 实现日志，每天的日志记录到一个文件中。 例如：开发一个网站，4CPU 开 4个 进程(或 8个)
假设：

- 启动网站，创建 4个 进程，每个进程中都打开文件 a1.log， 每个进程中都有

```python
# 每个进程都有各自文件对象
file_object = open("a1.log", 'a', encoding='utf-8')
```

- 用户访问

```python
# 其中 1个 进程接收并加以处理，执行自己的 write
file_object.write('日志...')
```

- 同时来 4个

```python
# 其中 4个 进程接收并处理，执行自己的 write
file_object.write('日志...')
```

### logging 模块实现机制(自动切割日志的功能)

- 启动网站，创建 4个 进程，每个进程中都打开文件 a1.log， 每个进程中都有

    ```python
    # 每个进程都有各自文件对象
    file_object = open("a1.log", 'a', encoding='utf-8')
    ```

- 用户访问

    ```python
    # 其中 1个 进程接收并加以处理，执行自己的 write
    file_object.write('日志...')
    ```

- 同时来 4个

    ```python
    # 其中 4个 进程接收并处理，执行自己的 write
    file_object.write('日志...')
    ```

- 问题1：多进程写日志会导致删除

    - 例如 11-18日
        ```text
        默认都会记录到 a1.log 文件中
        到了 11-19日
        判断 a1-11-19.log 文件是否存在，如果存在，就删除(导致多进程会一直删除文件)(优化，文件不存在，可以重命名；文件存在，继续在 a1.log 中写入日志)
        a1.log -> a1-11-19.log  # 到 11.19 后 a1.log 重命名为 a1-11-19.log
        然后再写入 a1.log
        ```

    - 源码找原因

    ```python
    import time
    import os
    from logging.handlers import BaseRotatingHandler
    
    
    class TimedRotatingFileHandler(BaseRotatingHandler):
        """
        Handler for logging to a file, rotating the log file at certain timed
        intervals.
      
        If backupCount is > 0, when rollover is done, no more than backupCount
        files are kept - the oldest ones are deleted.
        """
    
        def doRollover(self):
            """
            do a rollover; in this case, a date/time stamp is appended to the filename
            when the rollover happens.  However, you want the file to be named for the
            start of the interval, not the current time.  If there is a backup count,
            then we have to get a list of matching filenames, sort them and remove
            the one with the oldest suffix.
            """
            if self.stream:
                self.stream.close()
                self.stream = None
            # get the time that this sequence started at and make it a TimeTuple
            currentTime = int(time.time())
            dstNow = time.localtime(currentTime)[-1]
            t = self.rolloverAt - self.interval
            if self.utc:
                timeTuple = time.gmtime(t)
            else:
                timeTuple = time.localtime(t)
                dstThen = timeTuple[-1]
                if dstNow != dstThen:
                    if dstNow:
                        addend = 3600
                    else:
                        addend = -3600
                    timeTuple = time.localtime(t + addend)
    
            # # 新文件名，a1-11-19.log
            # dfn = self.rotation_filename(self.baseFilename + "." +
            #                              time.strftime(self.suffix, timeTuple))
            # # 判断是否存在，存在删除，问题就出在这，多进程时会删除其他进程创建备份的
            # if os.path.exists(dfn):
            #     os.remove(dfn)
            # # 重命名
            # self.rotate(self.baseFilename, dfn)
    
            # 修复
            dfn = self.rotation_filename(self.baseFilename + "." +
                                         time.strftime(self.suffix, timeTuple))
            # 判断是否存在，不存在重命名
            if not os.path.exists(dfn):
                self.rotate(self.baseFilename, dfn)
    
            if self.backupCount > 0:
                for s in self.getFilesToDelete():
                    os.remove(s)
            if not self.delay:
                self.stream = self._open()
            newRolloverAt = self.computeRollover(currentTime)
            while newRolloverAt <= currentTime:
                newRolloverAt = newRolloverAt + self.interval
            # If DST changes and midnight or weekly rollover, adjust for this.
            if (self.when == 'MIDNIGHT' or self.when.startswith('W')) and not self.utc:
                dstAtRollover = time.localtime(newRolloverAt)[-1]
                if dstNow != dstAtRollover:
                    if not dstNow:  # DST kicks in before next rollover, so we need to deduct an hour
                        addend = -3600
                    else:  # DST bows out before next rollover, so we need to add an hour
                        addend = 3600
                    newRolloverAt += addend
            self.rolloverAt = newRolloverAt
    ```

- 问题2：日志文件不能做相关操作(比如：删除或修改日志文件，日志就不创建文件继续打印了，需要重新执行该程序，才会重新创建文件写入日志)

  ```python
  # 类比 WatchedFileHandler 提供解决思路
  import os
  import time
  from stat import ST_INO, ST_DEV
  
  file_obj = open('xxx.log', 'a', encoding='utf-8')
  sres = os.fstat(file_obj.fileno())
  dev, ino = sres[ST_DEV], sres[ST_INO]
  
  while True:
      # WatchedFileHandler 处理 handler 的处理机制
      # 删除文件之后报错
      try:
          # stat the file by path, checking for existence
          new_sres = os.stat("xxx.log")
      except FileNotFoundError:
          sres = None
      if not new_sres or new_sres[ST_DEV] != dev or new_sres[ST_INO] != ino:
          print("文件被删除或修改了")
          # 重新打开，获取标志数据
          file_obj = open('xxx.log', 'a', encoding='utf-8')
          sres = os.fstat(file_obj.fileno())
          dev, ino = sres[ST_DEV], sres[ST_INO]
  
      file_obj.write("111\n")
      file_obj.flush()
      time.sleep(1)
  
  ```

### 多进程和文件修改(删除)后 2 个结合重写 handler

#### logging Handler 源码解析(TimedRotatingFileHandler为例)

```python

import os
import time
from _stat import ST_INO
from logging.handlers import TimedRotatingFileHandler

import logging
from stat import ST_DEV

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {
        "standard": {
            "format": '[%(levelname)s][%(asctime)s][%(filename)s:%(lineno)d]%(message)s',
            "style": "%"
        }
    },
    "handlers": {
        "error_file": {
            "class": "logging.handlers.CustomizeHandler",  # 自定义处理类
            "formatter": "standard",
            "filename": "a1.log",
            "when": "S",  # 根据天拆分日志,
            "interval": 10,  # 1天
            "backupCount": 2,  # 备份数
            "encoding": "utf-8"
        }
    },
    "loggers": {
        "": {
            "handlers": ["error_file"],
            "level": "ERROR",
            "propagate": True
        },
        "error": {
            "handlers": ["error_file"],
            "level": "ERROR",
            "propagate": True
        }
    }
}

"""
继承从下往上
1、实例化 CustomizeHandler 对象
    logging.Filterer.__init__()
    logging.Handler.__init__()
    logging.StreamHandler.__init__()
    logging.FileHandler.__init__()
    BaseRotatingHandler.__init__()
    TimedRotatingFileHandler.__init__() # 接受了很多关键字参数，这些都是配置字典里面 handlers 里面的值
    CustomizeHandler.__init__()
    # 看2、3点，发现关键点在于
        self.baseFilename      # 日志文件的绝对路径
        self.stream = stream   # 打开的文件对象
    # 看第 4 点，需要写日志时
        handler对象.emit("日志内容")
        把检测文件标识的代码移植到 emit 里面
    

2、FileHandler.__init__()
class FileHandler(StreamHandler):

    def __init__(self, filename, mode='a', encoding=None, delay=False, errors=None):
        filename = os.fspath(filename)
        #keep the absolute path, otherwise derived classes which use this
        #may come a cropper when the current directory changes
        # 日志文件的绝对路径
        self.baseFilename = os.path.abspath(filename)
        self.mode = mode
        self.encoding = encoding
        self.errors = errors
        self.delay = delay
        if delay:
            #We don't open the stream, but we still need to call the
            #Handler constructor to set level, formatter, lock etc.
            Handler.__init__(self)
            self.stream = None
        else:
            # self._open() 是打开文件对象返回的句柄
            StreamHandler.__init__(self, self._open())
            
    def _open(self):
        return open(self.baseFilename, self.mode, encoding=self.encoding,
                    errors=self.errors)

3、StreamHandler.__init__()
class StreamHandler(Handler):

    terminator = '\n'

    def __init__(self, stream=None):
        Handler.__init__(self)
        if stream is None:
            stream = sys.stderr
        self.stream = stream  # 关键点，这里赋值了 self.stream = 文件句柄

4、开始写日志：handler对象.emit("日志内容")

class BaseRotatingHandler(logging.FileHandler):
    def emit(self, record):
        try:
            # 判断是否已经过了设置的时间(比如第二天)，就重命名
            if self.shouldRollover(record):
                self.doRollover()
            # 执行父类的写日志
            logging.FileHandler.emit(self, record)
        except Exception:
            self.handleError(record)
            
    
class FileHandler(StreamHandler):
    def emit(self, record):
    # 如果文件句柄为空，重新打开文件赋值 self.stream
    if self.stream is None:
        self.stream = self._open()
    # 执行父类的写日志 
    StreamHandler.emit(self, record)
    
class StreamHandler(Handler):
    def emit(self, record):
        try:
            msg = self.format(record)  # 按你设置的 formatter 格式化日志
            stream = self.stream
            stream.write(msg + self.terminator)  # 文件写入日志
            self.flush()
        except RecursionError:  # See issue 36272
            raise
        except Exception:
            self.handleError(record)
"""


class CustomizeHandler(TimedRotatingFileHandler):

    def __init__(self, *args, **kwargs):
        """
        借鉴 WatchedFileHandler 的机制检测文件
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.dev, self.ino = -1, -1
        self._statstream()

    # WatchedFileHandler 方法
    def _statstream(self):
        if self.stream:
            sres = os.fstat(self.stream.fileno())
            self.dev, self.ino = sres[ST_DEV], sres[ST_INO]

    # WatchedFileHandler 方法
    def reopenIfNeeded(self):
        """
        Reopen log file if needed.

        Checks if the underlying file has changed, and if it
        has, close the old stream and reopen the file to get the
        current stream.
        """
        # Reduce the chance of race conditions by stat'ing by path only
        # once and then fstat'ing our new fd if we opened a new log stream.
        # See issue #14632: Thanks to John Mulligan for the problem report
        # and patch.
        try:
            # stat the file by path, checking for existence
            sres = os.stat(self.baseFilename)
        except FileNotFoundError:
            sres = None
        # compare file system stat with that of our stream file handle
        if not sres or sres[ST_DEV] != self.dev or sres[ST_INO] != self.ino:
            if self.stream is not None:
                # we have an open file handle, clean it up
                self.stream.flush()
                self.stream.close()
                self.stream = None  # See Issue #21742: _open () might fail.
                # open a new file handle and get new stat info from that fd
                self.stream = self._open()
                self._statstream()

    def doRollover(self):
        """
        do a rollover; in this case, a date/time stamp is appended to the filename
        when the rollover happens.  However, you want the file to be named for the
        start of the interval, not the current time.  If there is a backup count,
        then we have to get a list of matching filenames, sort them and remove
        the one with the oldest suffix.
        """
        if self.stream:
            self.stream.close()
            self.stream = None
        # get the time that this sequence started at and make it a TimeTuple
        currentTime = int(time.time())
        dstNow = time.localtime(currentTime)[-1]
        t = self.rolloverAt - self.interval
        if self.utc:
            timeTuple = time.gmtime(t)
        else:
            timeTuple = time.localtime(t)
            dstThen = timeTuple[-1]
            if dstNow != dstThen:
                if dstNow:
                    addend = 3600
                else:
                    addend = -3600
                timeTuple = time.localtime(t + addend)
        dfn = self.rotation_filename(self.baseFilename + "." +
                                     time.strftime(self.suffix, timeTuple))
        if not os.path.exists(dfn):  # 修复多进程删除日志文件，只有不存在再重命名
            self.rotate(self.baseFilename, dfn)
        if self.backupCount > 0:
            for s in self.getFilesToDelete():
                os.remove(s)
        if not self.delay:
            self.stream = self._open()
        newRolloverAt = self.computeRollover(currentTime)
        while newRolloverAt <= currentTime:
            newRolloverAt = newRolloverAt + self.interval
        # If DST changes and midnight or weekly rollover, adjust for this.
        if (self.when == 'MIDNIGHT' or self.when.startswith('W')) and not self.utc:
            dstAtRollover = time.localtime(newRolloverAt)[-1]
            if dstNow != dstAtRollover:
                if not dstNow:  # DST kicks in before next rollover, so we need to deduct an hour
                    addend = -3600
                else:  # DST bows out before next rollover, so we need to add an hour
                    addend = 3600
                newRolloverAt += addend
        self.rolloverAt = newRolloverAt

    def emit(self, record):
        try:
            # 判断是否已经过了设置的时间(比如第二天)，就重命名
            if self.shouldRollover(record):  # (TimedRotatingFileHandler 检测机制)
                self.doRollover()
            # 判断文件是否存在或修改(WatchedFileHandler的 检测机制)
            self.reopenIfNeeded()  # 文件删除后，函数里面报错，会重新打开赋值
            # 执行父类的写日志
            logging.FileHandler.emit(self, record)
        except Exception:
            self.handleError(record)
```

> 思考：上面重写的 handler 类，会不会出现问题？
>
> 答案：会的，比如多进程，第二天的时候所有的进程卡在时间特别短的时候，一起重命名，就会出现问题
>
> 解决：为什么不每天得日志都写入一个当天的日志，而不是重命名

### 每天写入当天日志

```python

"""
继承从下往上
1、实例化 CustomizeHandler 对象
    logging.Filterer.__init__()
    logging.Handler.__init__()
    logging.StreamHandler.__init__()
    logging.FileHandler.__init__()
    WatchedFileHandler.__init__() # 检测文件是否修改，删除的处理类
    CustomizeOneDayOneLogHandler.__init__()
"""

from logging.handlers import WatchedFileHandler

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {
        "standard": {
            "format": '[%(levelname)s][%(asctime)s][%(filename)s:%(lineno)d]%(message)s',
            "style": "%"
        }
    },
    "handlers": {
        "error_file": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "formatter": "standard",
            "filepath": "logs",
            "encoding": "utf-8"
        }
    },
    "loggers": {
        "": {
            "handlers": ["error_file"],
            "level": "ERROR",
            "propagate": True
        },
        "error": {
            "handlers": ["error_file"],
            "level": "ERROR",
            "propagate": True
        }
    }
}


class CustomizeOneDayOneLogHandler(WatchedFileHandler):
    """
    每天创建一个日志，每天的日志都打入当天的日志文件
    文件名：a-2021-10-20.log, 不会出现 a-2021-10-20.log.2021-10-21 这样的
    """

    def __init__(self, file_path, file_name_prefix, mode='a', encoding=None, delay=False,
                 errors=None):
        """
        :param file_path: 日志文件路径
        :param file_name_prefix: 日志文件前缀，就是 logging.getLogger(__name__) 获取，只是重下了 log 类继承 Logger，详细看下面
        :param mode:
        :param encoding:
        :param delay:
        :param errors:
        """
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        self.file_path = file_path
        self.file_name_prefix = file_name_prefix

        self.file_name = self.get_file_name()
        filename = os.path.join(file_path, self.file_name)

        # errors py3.9 有这个参数，3.8、3.7没有(更早版本也可能是)
        # super().__init__(filename=filename, mode=mode, encoding=encoding, delay=delay, errors=errors)
        super(CustomizeOneDayOneLogHandler, self).__init__(filename=filename, mode=mode, encoding=encoding, delay=delay)

    def get_file_name(self):
        """
        TODO 怎么切分可以这里实现，做成那个时间切分的参数配置
        :return: 日志名称
        """
        # 一天分一次
        return "{}-{}.log".format(self.file_name_prefix, datetime.datetime.now().strftime("%Y-%m-%d"))
        # 一分钟分一次
        # return "{}-{}.log".format(self.file_name_prefix, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))

    def emit(self, record):
        """
        Emit a record.

        If underlying file has changed, reopen the file before emitting the
        record to it.
        """

        current_file_name = self.get_file_name()
        # 文件不一致，新建文件 + 重新打开 + 重新获取 os.stat
        if current_file_name != self.file_name:
            self.file_name = current_file_name
            # 重新赋值，当前的文件名应该是最新的日期
            self.baseFilename = os.path.abspath(os.path.join(self.file_path, current_file_name))

            if self.stream:
                self.stream.flush()
                self.stream.close()
            self.stream = self._open()
            self._statstream()
        super(CustomizeOneDayOneLogHandler, self).emit(record)
```

## 项目中使用

```python
#!/usr/bin/env python
# -*- coding:utf8 -*-
__date__ = "2021/10/20 11:20"
__doc__ = """"""

# 定义三种日志输出格式 开始
import datetime
import glob
import os
import sys
import time
import traceback
from logging import Logger
from logging.handlers import WatchedFileHandler
import logging.config

standard_format = '[%(asctime)s][%(threadName)s:%(thread)d][task_id:%(name)s][%(filename)s:%(lineno)d]'
                  '[%(levelname)s][%(message)s]'  # 其中 name 为 getlogger 指定的名字

simple_format = '[%(levelname)s][%(asctime)s][%(filename)s:%(lineno)d]%(message)s'

id_simple_format = '[%(levelname)s][%(asctime)s] %(message)s'


# 第一种：每天一个日志文件

class CustomizeOneDayOneLogHandler(WatchedFileHandler):
    """
    每天创建一个日志，每天的日志都打入当天的日志文件
    文件名：a-2021-10-20.log, 不会出现 a-2021-10-20.log.2021-10-21 这样的
    """

    def __init__(self, file_path, file_name_prefix, backup_count: int = 5, mode='a', encoding=None, delay=False,
                 errors=None):
        """
        :param file_path: 日志文件路径
        :param file_name_prefix: 日志文件前缀
        :param backup_count: 日志备份数量
        :param mode:
        :param encoding:
        :param delay:
        :param errors:
        """
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        self.file_path = file_path
        self.file_name_prefix = file_name_prefix
        self.backup_count = backup_count

        self.file_name = self.get_file_name()
        filename = os.path.join(file_path, self.file_name)

        # errors py3.9 有这个参数，3.8、3.7没有(更早版本也可能是)
        # super().__init__(filename=filename, mode=mode, encoding=encoding, delay=delay, errors=errors)
        super().__init__(filename=filename, mode=mode, encoding=encoding, delay=delay)
        # 初始化的时候清理下，防止创建了文件不写入，导致空日志文件过多
        self.auto_clear()

    def get_file_name(self) -> str:
        """
        TODO 怎么切分可以这里实现，做成那个时间切分的参数配置
        :return: 日志名称
        """
        # 一天分一次
        return "{}-{}.log".format(self.file_name_prefix, datetime.datetime.now().strftime("%F"))
        # 一分钟分一次
        # return "{}-{}.log".format(self.file_name_prefix, datetime.datetime.now().strftime('%F-%H-%M'))

    def emit(self, record):
        """
        Emit a record.

        If underlying file has changed, reopen the file before emitting the
        record to it.
        """

        current_file_name = self.get_file_name()

        if current_file_name != self.file_name:
            self.file_name = current_file_name
            self.baseFilename = os.path.abspath(os.path.join(self.file_path, current_file_name))

            if self.stream:
                self.stream.flush()
                self.stream.close()
            self.stream = self._open()
            self._statstream()
            self.auto_clear()
        super().emit(record)

    def auto_clear(self):
        """
        自动清理 log 文件
        :return:
        """
        file_list = sorted(glob.glob(os.path.join(self.file_path, self.file_name_prefix + '*')),
                           key=lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getctime(x))),
                           reverse=True)
        for file_ in file_list[self.backup_count:]:
            os.remove(file_)


class MyLog(Logger):
    def error(self, msg, exc_info=True, extra=None, *args):
        """
        重写父类方法,exc_info默认为True
        """
        if self.isEnabledFor(level=40):  # 父类抄来,使用默认值
            self._log(40, msg, args, exc_info=True, extra=None)

    def info(self, msg, *args, **kwargs):
        """
        防止在 exception里写 log.info(error) 抓取一切报错堆栈
        Log 'msg % args' with severity 'INFO'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.info("Houston, we have a %s", "interesting problem", exc_info=1)
        """
        error_msg = traceback.format_exc()
        # Windows 和 Linux 不一样？
        # if not error_msg == 'None\n':
        if not error_msg == 'NoneType: None\n':
            error_msg = str(msg) + error_msg
            self._log(20, error_msg, args, **kwargs)
        if self.isEnabledFor(20):  # 父类抄来,使用默认值
            self._log(20, msg, args, **kwargs)


class LogUtil(object):

    def __init__(self, file_name_prefix, backup_count: int = 5, console_out: bool = False):
        """
        :param file_name_prefix: 日志名称前缀
        :param backup_count: 备份数量
        :param console_out: 是否在控制台输出
        """
        self.file_name_prefix = file_name_prefix
        if sys.platform.startswith("linux"):
            self.base_dir = '/app/logs'
        else:
            self.base_dir = './logs'  # 本地调试

        formatter = logging.Formatter('[%(levelname)s] [%(asctime)s] [%(filename)s-line:%(lineno)d] %(message)s')
        # 按天存放 同类型log最多保留5个
        self.log_file_handler = CustomizeOneDayOneLogHandler(self.base_dir, self.file_name_prefix,
                                                             backup_count=backup_count, encoding='utf-8')
        self.log_file_handler.setFormatter(formatter)
        self.logger = MyLog(name=self.log_file_handler.file_name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(self.log_file_handler)  # 输出到文件

        if console_out:  # 往屏幕上输出
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)  # 设置屏幕上显示的格式
            self.logger.addHandler(console_handler)  # 输出到控制台


log_test1 = LogUtil("test1", console_out=True).logger
log_test2 = LogUtil("test2", console_out=True).logger

if __name__ == '__main__':
    while True:
        time.sleep(5)
        log_test1.info("aaaaaaaaaaa")
        # time.sleep(0.1)
        # log_test2.info("bbbbbbbbbbb")

        # try:
        #     a = 1/ 0
        # except Exception as e:
        #     # log_test1.error("error")
        #     log_test1.info("info")

```

