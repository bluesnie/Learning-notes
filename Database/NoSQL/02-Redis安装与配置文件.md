###### datetime:2019/10/30 16:26
###### author:nzb

## Redis安装与配置文件

### 安装

### Linux安装

可以使用Linux系统的包管理工具（如yum）来安装Redis，也可以通过在Redis的[官方网站](https://redis.io/)下载Redis的源代码，解压缩解归档之后通过make工具对源代码进行构建并安装，在更新这篇文档时，Redis官方提供的最新稳定版本是[Redis 5.0.4](http://download.redis.io/releases/redis-5.0.4.tar.gz)。

```Shell
wget http://download.redis.io/releases/redis-5.0.4.tar.gz
gunzip redis-5.0.4.tar.gz
tar -xvf redis-5.0.4.tar
cd redis-5.0.4
make && make install
```

### Docker安装

- 搜索镜像
    
    `docker search redis`

- 拉取镜像

    `docker pull redis`

- 创建redis容器

    `docker run -d --name redis --restart always -p 6379:6379 -v /usr/local/redis/data:/data redis --requirepass "123456" --appendonly yes` 

- 创建redis容器（指定配置文件）

    `docker run -d --name redis --restart always -p 6379:6379 -v /usr/local/redis/config:/usr/local/redis/conf/redis.conf -v /usr/local/redis/data:/data redis redis-server /usr/local/redis/conf/redis.conf --requirepass "123456" --appendonly yes`
    
    `docker run -d --name redis --restart always -p 6379:6379 -v /usr/local/redis/data:/data redis --requirepass "123456" --appendonly yes`

- 参数说明：
    - -p 6379:6379　　                                 
    //容器redis端口6379映射宿主主机6379
    - --name redis　　                            
    //容器名字为redis
    - -v /usr/local/redis/conf:/usr/local/redis/conf/redis.conf     
    //docker镜像redis默认无配置文件，在宿主主机/usr/local/redis/conf下创建redis.conf配置文件，会将宿主机的配置文件复制到docker中(加上这参数会报错)
    - -v /root/redis/redis01/data:/data　　      
    //容器/data映射到宿主机 /usr/local/redis/data下
    - -d redis 　　                              
    //后台模式启动redis
    - redis-server /usr/local/redis/conf/redis.conf         
    //redis将以/usr/local/redis/conf/redis.conf为配置文件启动(加上这参数会报错)
    - --appendonly yes　　                       
    //开启redis的AOF持久化，默认为false，不持久化