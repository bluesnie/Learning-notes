###### datetime:2020/1/3 14:24
###### author:nzb

## 使用Docker安装FastDFS分布式文件系统

#### 拉取镜像

```text
    docker image pull delron/fastdfs
```

#### 运行tracker

```text
docker run -itd --network=host --name fastdfs-tracker -v /var/fdfs/tracker:/var/fdfs delron/fastdfs
```
将fastDFS tracker运行目录映射到本机的 /var/fdfs/tracker目录中。

- 查看是否允许起来
`docker ls`

- 停止运行
`docker stop fastdfs-tracker`

- 重新运行
`docker start fastdfs-tracker`

#### 运行storage

```text
docker run -itd --network=host --name fastdfs-storage -e TRACKER_SERVER=192.168.1.218:22122 -v /var/fdfs/storage:/var/fdfs delron/fastdfs storage
```

- TRACKER_SERVER=本机的ip地址:22122 本机ip地址不要使用127.0.0.1
- 将fastDFS storage运行目录映射到本机的/var/fdfs/storage目录中

- 查看是否允许起来
`docker ls`

- 停止运行
`docker stop fastdfs-storage`

- 重新运行
`docker start fastdfs-storage`


**注意**：如果无法重新运行，可以删除/var/fdfs/storage/data目录下的fdfs_storaged.pid 文件，然后重新运行storage。

## Django项目之FastDFS文件存储系统

#### FastDFS的Python客户端

python版本的FastDFS客户端使用说明参考：https://github.com/jefforeilly/fdfs_client-py

- 安装

    安装fdfs_client-py-master.zip到虚拟环境中

    ```text
    pip install fdfs_client-py-master.zip
    pip install mutagen
    pip install requests
    ```

- 配置

    在项目/utils目录下新建fastdfs目录，新建client.conf配置文件
    ```text
    # connect timeout in seconds
    # default value is 30s
    connect_timeout=30
    
    # network timeout in seconds
    # default value is 30s
    network_timeout=60
    
    # the base path to store log files
    base_path=FastDFS客户端存放日志文件的目录
    
    # tracker_server can ocur more than once, and tracker_server format is
    #  "host:port", host can be hostname or ip address
    tracker_server=172.17.0.1:22122
    
    #standard log level as syslog, case insensitive, value list:
    ### emerg for emergency
    ### alert
    ### crit for critical
    ### error
    ### warn for warning
    ### notice
    ### info
    ### debug
    log_level=info
    
    # if use connection pool
    # default value is false
    # since V4.05
    use_connection_pool = false
    
    # connections whose the idle time exceeds this time will be closed
    # unit: second
    # default value is 3600
    # since V4.05
    connection_pool_max_idle_time = 3600
    
    # if load FastDFS parameters from tracker server
    # since V4.05
    # default value is false
    load_fdfs_parameters_from_tracker=false
    
    # if use storage ID instead of IP address
    # same as tracker.conf
    # valid only when load_fdfs_parameters_from_tracker is false
    # default value is false
    # since V4.05
    use_storage_id = false
    
    # specify storage ids filename, can use relative or absolute path
    # same as tracker.conf
    # valid only when load_fdfs_parameters_from_tracker is false
    # since V4.05
    storage_ids_filename = storage_ids.conf
    
    #HTTP settings
    http.tracker_server_port=80
    
    #use "#include" directive to include HTTP other settiongs
    ##include http.conf
    ```

**注意**：需要修改一下client.conf配置文件

```text
    # FastDFS客户端存放日志文件的目录
    base_path=
    # 运行tracker服务的机器ip
    tracker_server=172.17.0.1:22122 
```

#### 自定义Django文件存储系统

Django自带文件存储系统，但是默认文件存储在本地，将文件保存到FastDFS服务器上，所以需要自定义文件存储系统。

在项目/utils/fastdfs目录中创建fdfs_storage.py文件，实现可以使用FastDFS存储文件的存储类如下

```text
    from django.conf import settings
    from django.core.files.storage import Storage
    from django.utils.deconstruct import deconstructible
    from fdfs_client.client import Fdfs_client
    
    
    @deconstructible
    class FastDFSStorage(Storage):
        def __init__(self, base_url=None, client_conf=None):
            """
            初始化
            :param base_url: 用于构造图片完整路径使用，图片服务器的域名
            :param client_conf: FastDFS客户端配置文件的路径
            """
            if base_url is None:
                base_url = settings.FDFS_URL
            self.base_url = base_url
            if client_conf is None:
                client_conf = settings.FDFS_CLIENT_CONF
            self.client_conf = client_conf
    
        def _open(self, name, mode='rb'):
            """
            用不到打开文件，所以省略
            """
            pass
    
        def _save(self, name, content):
            """
            在FastDFS中保存文件
            :param name: 传入的文件名
            :param content: 文件内容
            :return: 保存到数据库中的FastDFS的文件名
            """
            client = Fdfs_client(self.client_conf)
            ret = client.upload_by_buffer(content.read())
            if ret.get("Status") != "Upload successed.":
                raise Exception("upload file failed")
            file_name = ret.get("Remote file_id")
            return file_name
    
        def url(self, name):
            """
            返回文件的完整URL路径
            :param name: 数据库中保存的文件名
            :return: 完整的URL
            """
            return self.base_url + name
    
        def exists(self, name):
            """
            判断文件是否存在，FastDFS可以自行解决文件的重名问题
            所以此处返回False，告诉Django上传的都是新文件
            :param name:  文件名
            :return: False
            """
            return False
```

说明:

自定义文件存储系统的方法如下：

- 1）需要继承自django.core.files.storage.Storage，如

```python
    from django.core.files.storage import Storage
    
    class FastDFSStorage(Storage):
        ...
```

- 2）支持Django不带任何参数来实例化存储类，也就是说任何设置都应该从django.conf.settings中获取

```python
    from django.conf import settings
    from django.core.files.storage import Storage
    
    class FastDFSStorage(Storage):
        def __init__(self, base_url=None, client_conf=None):
            if base_url is None:
                base_url = settings.FDFS_URL
            self.base_url = base_url
            if client_conf is None:
                client_conf = settings.FDFS_CLIENT_CONF
            self.client_conf = client_conf
```

- 3）存储类中必须实现_open()和_save()方法，以及任何后续使用中可能用到的其他方法。

    - _open(name, mode='rb')

        被Storage.open()调用，在打开文件时被使用。

    - _save(name, content)

        被Storage.save()调用，name是传入的文件名，content是Django接收到的文件内容，该方法需要将content文件内容保存。

        Django会将该方法的返回值保存到数据库中对应的文件字段，也就是说该方法应该返回要保存在数据库中的文件名称信息。

    - exists(name)

        如果名为name的文件在文件系统中存在，则返回True，否则返回False。

    - url(name)

        返回文件的完整访问URL

    - delete(name)

        删除name的文件

    - listdir(path)

        列出指定路径的内容

    - size(name)

        返回name文件的总大小

    **注意**，并不是这些方法全部都要实现，可以省略用不到的方法。

- 4）需要为存储类添加django.utils.deconstruct.deconstructible装饰器

#### 在Django配置中设置自定义文件存储类

在settings.py文件中添加设置

```python
# django文件存储
DEFAULT_FILE_STORAGE = '项目名.utils.fastdfs.fdfs_storage.FastDFSStorage'

# FastDFS
FDFS_URL = 'http://域名:端口'  
FDFS_CLIENT_CONF = os.path.join(BASE_DIR, 'utils/fastdfs/client.conf')
```

#### 添加image域名

在/etc/hosts中添加访问FastDFS storage服务器的域名

```text
127.0.0.1   xx域名
```



