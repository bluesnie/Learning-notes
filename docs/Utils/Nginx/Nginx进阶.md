###### datetime:2022-01-08 14:34:00

###### author:nzb

# Nginx

> 该文档由 `html2text` 生成，二次编辑的

## 简介

* 高性能

* web服务器

* 反向代理服务器

* 占用内存少

* 高并发处理強

## 正向代理和反向代理

* **正向代理**

  以代理服务器来接受 Internet 上的连接请求，然后将请求转发给内部网络上的服务器，并将从服务器上得到的结果返回给 Internet 上请求连接的客户端

* **反向代理**

  正向代理类似一个跳板机，代理访问外部资源，比如：我是一个用户，我访问不了某网站，但是我能访问一个代理服务器，这个代理服务器，他能访问那个我不能访问的网站
  ，于是我先连上代理服务器，告诉它我需要那个无法访问网站的内容，代理服务器去取回来，然后返回给我。例子：VPN

## 配置文件详解

> `#` ： 表示配置文件中默认关闭

### `#` user nobody;

配置worker进程用户，主进程master是root，nobody也是一个Linux用户，一般用于启动程序，没有密码

### worker_processes auto;

工作进程数，根据硬件调整，通常等于CPU数量或者2倍于CPU数量(建议跟CPU的核数量一致)

### error_log /var/log/nginx/error.log;

  ```text
  配置全局错误日志级类型，【debug | info | notice | warn | error | crit 】，默认是error 
  # error_log /var/log/nginx/error.log notice; 
  # error_log /var/log/nginx/error.log info;
  ```

### pid /run/[nginx.pid](http://nginx.pid/);

配置进程pid文件

### events

  ```text
  events { 
        use epoll;                    # 配置工作模式(多路IO复用方式)和连接上限
        worker_connections 1024;      # 单进程的并发量，最大：65535 ​总并发=进程数*单个进程的并发量 单个进程的并发量：655535
  }
  ```

### http

HTTP服务器相关配置，利用它的反向代理功能提供负载均衡支持

```text
http {
    include /etc/nginx/mime.types;                                                # 配置Nginx支持哪些多媒体类型，可以在conf/mime.types查看支持哪些多媒体文件
    default_type application/octet-stream;                                        # 默认使用二进制流格式，流类型，可以理解为支持任意类型
    # 配置日志格式，main是一个变量名
    log_format main '$remote_addr - $remote_user [$time_local] "$request" ' '$status $body_bytes_sent "$http_referer" ' '"$http_user_agent" "$http_x_forwarded_for"'; 
    access_log /var/log/nginx/access.log main;                                    # 配置access.log日志及存放路径，并使用上面定义的main日志格式
    sendfile on;                                                                  # 开启高效文件传输模式
    tcp_nopush on;                                                                # 用sendfile传输文件时有利于改善性能
    tcp_nodelay on;                                                               # 禁用Nagle来解决交互性问题
    keepalive_timeout 30;                                                         # 客户端保持连接时间，单位秒
    gzip on;                                                                      # 开启gzip压缩输出
    include /etc/nginx/conf.d/*.conf;                                             # 包含其他配置文件，里面文件包含server虚拟主机
    include /root/project/conf/*.conf;                                            # 包含项目的Nginx配置文件
    
    server{
        # 配置虚拟主机，一个http里面可以有多个(server_name和listen不能完全一样)，可以写在conf.d目录下，包含进来
        listen 80;                                                                # 配置监听端口，默认端口80
        server_name localhost;                                                    # 配置服务名，域名解析
        root /root/project/www;                                                   # 网站根目录
        charset koi8-r;                                                           # 配置字符集
        access_log /root/project/logs/access.log main;                            # 配置本虚拟主机的访问日志
        error_log /root/project/logs/error.log main;
      
        location / { 
            # 默认访问页，默认的匹配斜杠“/”（根路径）的请求，当访问路径中有斜杠/，会被location匹配到并进行处理
            include uwsgi_params; 
            uwsgi_pass 172.18.61.250:8000; 
            index index.html index.htm; 
        }
      
        location / test {                                                         # test会拼接到root路径之后
            root /opt/www;                                                        # root后面的值就是：/test中的“/”（根路径）
            index index.html index.htm; 
        }
        
        error_page 404;                                                           # 配置404页面
      
        # error_page 500 502 503 504 /50x.html;                                   # 配置50x错误页面
        
        location /50x.html {                                                      # 精准匹配
            root html;
        }
    }
}
```

## 主要应用

### 静态网站部署

包括HTML，js，css，图片等

### 负载均衡

#### 硬件负载均衡

比如：F5、深信服、Array等

* 优点是有厂商专业的技术服务团队提供支持，性能稳定

* 缺点是费用昂贵，对于规模较小的网络应用成本太高

#### 软件负载均衡

比如：Nginx、LVS、HAProxy等

* 优点是免费开源，成本低廉

##### 主配置

```text
http {
    # 为HTTP服务配置负载均衡
    upstream www.example.com {
        # 分发：IP：端口 weight 权重 max_fails失败次数 fail_timeout分发失败超时时间
        server 172.18.61.250:801 weight=4;
        server 172.18.61.250:802 weight=2;
        server 172.18.61.250:803 weight=2;
        ip_hash; 
    }

    # 配置同一用户访问同一个web服务器(解决session丢失问题导致无法登陆和验证码验证(生成和验证不在同一台服务器))
    server {
        listen 80 default_server;
        listen [::]:80 default_server;
        listen 443 ssl;
        listen [::]:443 ssl;
        ssl on;
        access_log /root/project/logs/access.log;
        error_log /root/project/logs/error.log;
        ssl_certificate root/project/conf/cert/214915882850706.pem;
        ssl_certificate_key /root/project/conf/cert/214915882850706.key;
        ssl_session_timeout 5m;
        ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE:ECDH:AES:HIGH:!NULL:!aNULL:!MD5:!ADH:!RC4;
        ssl_protocols TLSv1 TLSv1.1 TLSv1.2;
        ssl_prefer_server_ciphers on;
    
        location / {
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for; proxy_buffering off;
            proxy_pass http://www.example.com;  # 跟上面的upstream后的值一样
        }
    }
}
```

##### 其他配置

- 配置1：备份服务器

```text
# 可以用于更新代码
upstream www.example.com {
    server 172.18.61.250:801;
    server 172.18.61.250:803 backup;  # 其他所有的非backup服务器down掉的时候，才请求backup服务器
} 

```

- 配置2

```text
upstream www.example.com {
    server 172.18.61.250:801;
    server 172.18.61.250:803 down;   # down表示当前的服务器是down状态，不参与负载均衡，基本没什么用
}
```

##### 负载均衡策略

- 轮询（默认）
```text
    # 注意：这里的轮询并不是每个请求轮流分配到不同的后端服务器，与ip_hash类似，但是按照访问url的hash结果来分配请求，
    # 使得每个url定向到同一个后端服务器，主要应用与后端服务器为缓存时的场景下，如果后端服务器down掉，将自动删除。
    # 每台服务器交替访问，需要保证服务器的性能一样，否则会出现请求堆积导致宕机。
    upstream www.example.com {
        server 172.18.61.250:801;
        server 172.18.61.250:802;
        server 172.18.61.250:803;
    }
```

- 权重
```text
    # 每个请求按一定比例分发到不同的后端服务器，weight值越大访问的比例越大，用于后端服务器性能不均的情况。权重按服务器性能给。
    upstream www.example.com {
        # 分发：IP：端口 weight 权重 max_fails失败次数 fail_timeout分发失败超时时间
        server 172.18.61.250:801 weight=4;
        server 172.18.61.250:802 weight=2;
        server 172.18.61.250:803 weight=2;
        ip_hash; 
    }
    # 配置同一用户访问同一个web服务器(解决session丢失问题导致无法登陆和验证码验证(生成和验证不在同一台服务器))
    # 注意：不是说4个请求一次性给第一个，而是给一个后第二个请求给第二个，第三个给第三个，等等
```

- 最少连接数
```text
# web请求会被转移到连接数最少的服务器上，当不知道服务器性能时，不过可能导致请求堆积，因为最少连接的应该性能差。
    upstream www.example.com {
        least_conn;
        server 172.18.61.250:801;
        server 172.18.61.250:802;
        server 172.18.61.250:803;
    }
```

- ip_hash
```text
    # ip_hash也叫ip绑定，每个请求按访问ip的hash值分配，这样每个访问客户端会固定访问一个后端服务器，可以解决会话session丢失的问题。但是模完的数相同（hash碰撞），也会导致请求堆积。
    # 算法：hash（"124.207.55.82"）% 3
      # 客户端ip 
      # 3：3台服务器
    upstream www.example.com {
        server 172.18.61.250:801 weight=4;
        server 172.18.61.250:802 weight=2;
        server 172.18.61.250:803 weight=2;
        ip_hash; 
    }
    # 配置同一用户访问同一个web服务器(解决session丢失问题导致无法登陆和验证码验证(生成和验证不在同一台服务器))
```


### 静态代理

* 图片、css、html、js等交给Nginx处理

* 实现

    * 方式一：在nginx.conf的location中配置静态资源的后缀，进行拦截
        * 例如：当访问静态资源，则从Linux服务器/opt/static目录下获取（举例）
        ```text
        location ~.*\\.(gif|jpg|png|js|css)$ {
            root /opt/static;
        }
        ~：正则匹配开始 
        .：任意字符 
        *：​任意次数一个或多个 
        \：转义字符 
        $：匹配结尾​
        ```

    * 方式二：在nginx.conf的location中配置静态资源所在目录，进行拦截  
        * 例如：当访问静态资源，则从Linux服务器/opt/static目录下获取（举例）常用
        ```text
            location ~.*/(css|js|img|images) {              # 不匹配以什么结尾，匹配目录
              root /opt/static;
            }
        ```


### 动静分离

  * 动态资源：如Django项目

  * 静态资源：如图片、css、js等由Nginx服务器完成，选择Nginx是因为Nginx效率高

### 虚拟主机

例如：58同城

  * 虚拟主机，就是把一台物理服务器划分成多个“虚拟”的服务器，这样我们的一台物理服务器就可以做多个服务器来使用，从而可以配置多个网站。

  * 实现

      * 方法一：基于端口的虚拟主机（一般不用，了解）
          * 基于端口的虚拟主机配置，使用端口来区分
          * 浏览器使用同一个域名 + 端口 或 同一个 ip地址 + 端口访问
          ```text
              server {
              
                  listen 8080;
                  server_name www.example1.com;
                  location / {
                      proxy_pass http://www.myweb.com;
                  }
              }
              
              server {
                  listen 9090;
                  server_name www.example1.com;
                  location / {
                      proxy_pass http://www.myweb1.com;
                  }
              }
          ```
      * 方法二：基于域名的虚拟主机（掌握）
          * 基于域名的虚拟主机是最常见的一种虚拟主机
          ```text
          server {
              listen 80;
              server_name www.example1.com;
              location / {
                  proxy_pass http://www.myweb.com;
              }
              }
          
          server {
              listen 80;
              server_name www.example2.com;
              location / {
                  proxy_pass http://www.myweb2.com;
              }
          }
          ```