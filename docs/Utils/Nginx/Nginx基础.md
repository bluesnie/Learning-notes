###### datetime:2019/9/5 16:14
###### author:nzb

## Nginx的配置

1. 安装Nginx。

    ```Shell
    yum -y install nginx
    ```

2. 修改全局配置文件（`/etc/nginx/nginx.conf`）。

    ```Nginx
    # 配置用户
    user root;
    # 工作进程数(建议跟CPU的核数量一致)
    worker_processes auto;
    # 错误日志
    error_log /var/log/nginx/error.log;
    # 进程文件
    pid /run/nginx.pid;
    # 包含其他的配置
    include /usr/share/nginx/modules/*.conf;
    # 工作模式(多路IO复用方式)和连接上限
    events {
        use epoll;
        worker_connections 1024;(单进程的并发量, 总并发=进程数*单个进程的并发量)
    }
    # HTTP服务器相关配置
    http {
        # 日志格式
        log_format  main  '$remote_addr - $remote_user [$time_local] "$request" '
                          '$status $body_bytes_sent "$http_referer" '
                          '"$http_user_agent" "$http_x_forwarded_for"';
        # 访问日志
        access_log  /var/log/nginx/access.log  main;
        # 开启高效文件传输模式
        sendfile            on;
        # 用sendfile传输文件时有利于改善性能
        tcp_nopush          on;
        # 禁用Nagle来解决交互性问题
        tcp_nodelay         on;
        # 客户端保持连接时间
        keepalive_timeout   30;
        types_hash_max_size 2048;
        # 包含MIME类型的配置
        include             /etc/nginx/mime.types;
        # 默认使用二进制流格式
        default_type        application/octet-stream;
        # 包含其他配置文件
        include /etc/nginx/conf.d/*.conf;
        # 包含项目的Nginx配置文件
        include /root/project/conf/*.conf;
    }
    ```

3. 编辑局部配置文件（`/root/project/conf/nginx.conf`）。

    ```Nginx
    server {
        # 默认端口
        listen      80;
        # 域名解析
        server_name _;
        # 网站根目录
        root /root/project/www;
        # 缓存图片文件
        location ~ \.(jpeg|jpg|png)${       # 缓存文件类型
            # 缓存时间为1day
            expires 1d;     # h:小时, d:天
        }
        access_log /root/project/logs/access.log;
        error_log /root/project/logs/error.log;
        # 默认访问页
        location / {
            include uwsgi_params;
            uwsgi_pass 172.18.61.250:8000;
            index index.html index.htm;
        }
        location /static/ {
            alias /root/project/stat/;
            expires 30d;
        }
    }
    server {
        listen      443;
        server_name _;
        ssl         on;
        access_log /root/project/logs/access.log;
        error_log /root/project/logs/error.log;
        ssl_certificate     /root/project/conf/cert/214915882850706.pem;
        ssl_certificate_key /root/project/conf/cert/214915882850706.key;
        ssl_session_timeout 5m;
        ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE:ECDH:AES:HIGH:!NULL:!aNULL:!MD5:!ADH:!RC4;
        ssl_protocols TLSv1 TLSv1.1 TLSv1.2;
        ssl_prefer_server_ciphers on;
        location / {
            include uwsgi_params;
            uwsgi_pass 172.18.61.250:8000;
        }
        location /static/ {
            alias /root/project/static/;
            expires 30d;
        }
    }
    ```

    到此为止，我们可以启动Nginx来访问我们的应用程序，HTTP和HTTPS都是没有问题的，如果Nginx已经运行，在修改配置文件后，我们可以用下面的命令重新启动Nginx。

4. 重启Nginx服务器。

    ```Shell
    nginx -s reload
    ```

    或

    ```Shell
    systemctl restart nginx
    ```

> 说明：可以对Django项目使用`python manage.py collectstatic`命令将静态资源收集到指定目录下，要做到这点只需要在项目的配置文件`settings.py`中添加`STATIC_ROOT`配置即可。

#### 负载均衡配置

下面的配置中我们使用Nginx实现负载均衡，为另外的三个Nginx服务器（通过Docker创建）提供反向代理服务。

```Shell
docker run -d -p 801:80 --name nginx1 nginx:latest
docker run -d -p 802:80 --name nginx2 nginx:latest
docker run -d -p 803:80 --name nginx3 nginx:latest
```

```Nginx
user root;
worker_processes auto;
error_log /var/log/nginx/error.log;
pid /run/nginx.pid;

include /usr/share/nginx/modules/*.conf;

events {
    worker_connections 1024;
}

# 为HTTP服务配置负载均衡
http {   
	upstream fangtx {
	    # 分发：IP：端口 weight权重 max_fails失败次数 fail_timeout分发失败超时时间
		server 172.18.61.250:801 weight=4;
		server 172.18.61.250:802 weight=2;
		server 172.18.61.250:803 weight=2;
		# 配置同一用户访问同一个web服务器(解决session丢失问题导致无法登陆和验证码验证(生成和验证不在同一台服务器))
		ip_hash;
    }

	server {
		listen       80 default_server;
		listen       [::]:80 default_server;
		listen       443 ssl;
		listen       [::]:443 ssl;

        ssl on;
		access_log /root/project/logs/access.log;
		error_log /root/project/logs/error.log;
		ssl_certificate /root/project/conf/cert/214915882850706.pem;
		ssl_certificate_key /root/project/conf/cert/214915882850706.key;
		ssl_session_timeout 5m;
		ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE:ECDH:AES:HIGH:!NULL:!aNULL:!MD5:!ADH:!RC4;
		ssl_protocols TLSv1 TLSv1.1 TLSv1.2;
		ssl_prefer_server_ciphers on;

		location / {
			proxy_set_header Host $host;
			proxy_set_header X-Real-IP $remote_addr;
			proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
			proxy_buffering off;
			proxy_pass http://fangtx;  # 跟上面的upstream后的值一样
		}
	}
}
```

> 说明：Nginx在配置负载均衡时，默认使用WRR（加权轮询算法），除此之外还支持ip_hash、fair（需要安装upstream_fair模块）和url_hash算法。此外，在配置upstream模块时可以指定服务器的状态值，包括：backup（备份机器，其他服务器不可用时才将请求分配到该机器）、down、fail_timeout（请求失败达到max_fails后的暂停服务时间）、max_fails（允许请求失败的次数）和weight（轮询的权重）。

### Keepalived

当使用Nginx进行负载均衡配置时，要考虑负载均衡服务器宕机的情况。为此可以使用Keepalived来实现负载均衡主机和备机的热切换，从而保证系统的高可用性。Keepalived的配置还是比较复杂，通常由专门做运维的人进行配置，一个基本的配置可以参照[《Keepalived的配置和使用》](https://www.jianshu.com/p/dd93bc6d45f5)。