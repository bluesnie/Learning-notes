###### datetime:2019/10/31 16:02
###### author:nzb

## Redis基础

### 应用场景

- EXPIRE key seconds
    
    - 限时的优惠活动信息
    - 网站数据缓存(对于一些需要定时更新的数据,例如:积分排行榜)
    - 手机验证码
    - 限制网站访客访问频率(例如：1分钟最多访问10次)

### Redis键(key)

- `DEL key`
该命令用于在 key 存在时删除 key。

- `DUMP key`
序列化给定 key ，并返回被序列化的值。

- `EXISTS key`
检查给定 key 是否存在。

- `EXPIRE key seconds`
为给定 key 设置过期时间，以秒计。

- `EXPIREAT key timestamp`
EXPIREAT 的作用和 EXPIRE 类似，都用于为 key 设置过期时间。 不同在于 EXPIREAT 命令接受的时间参数是 UNIX 时间戳(unix timestamp)。

- `PEXPIRE key milliseconds`
设置 key 的过期时间以毫秒计。

- `PEXPIREAT key milliseconds-timestamp`
设置 key 过期时间的时间戳(unix timestamp) 以毫秒计

- `KEYS pattern`
查找所有符合给定模式( pattern)的 key 。

- `MOVE key db`
将当前数据库的 key 移动到给定的数据库 db 当中。

    示例：
    ```text
        # key 存在于当前数据库
        
        redis> SELECT 0                             # redis默认使用数据库 0，为了清晰起见，这里再显式指定一次。
        OK
        
        redis> SET song "secret base - Zone"
        OK
        
        redis> MOVE song 1                          # 将 song 移动到数据库 1
        (integer) 1
        
        redis> EXISTS song                          # song 已经被移走
        (integer) 0
        
        redis> SELECT 1                             # 使用数据库 1
        OK
        
        redis:1> EXISTS song                        # 证实 song 被移到了数据库 1 (注意命令提示符变成了"redis:1"，表明正在使用数据库 1)
        (integer) 1
        
        
        # 当 key 不存在的时候
        
        redis:1> EXISTS fake_key
        (integer) 0
        
        redis:1> MOVE fake_key 0                    # 试图从数据库 1 移动一个不存在的 key 到数据库 0，失败
        (integer) 0
        
        redis:1> select 0                           # 使用数据库0
        OK
        
        redis> EXISTS fake_key                      # 证实 fake_key 不存在
        (integer) 0
        
        
        # 当源数据库和目标数据库有相同的 key 时
        
        redis> SELECT 0                             # 使用数据库0
        OK
        redis> SET favorite_fruit "banana"
        OK
        
        redis> SELECT 1                             # 使用数据库1
        OK
        redis:1> SET favorite_fruit "apple"
        OK
        
        redis:1> SELECT 0                           # 使用数据库0，并试图将 favorite_fruit 移动到数据库 1
        OK
        
        redis> MOVE favorite_fruit 1                # 因为两个数据库有相同的 key，MOVE 失败
        (integer) 0
        
        redis> GET favorite_fruit                   # 数据库 0 的 favorite_fruit 没变
        "banana"
        
        redis> SELECT 1
        OK
        
        redis:1> GET favorite_fruit                 # 数据库 1 的 favorite_fruit 也是
        "apple"
    ```

- `PERSIST key`
移除 key 的过期时间，key 将持久保持。

    示例
    ```text
        redis> SET mykey "Hello"
        OK
        
        redis> EXPIRE mykey 10  # 为 key 设置生存时间
        (integer) 1
        
        redis> TTL mykey
        (integer) 10
        
        redis> PERSIST mykey    # 移除 key 的生存时间
        (integer) 1
        
        redis> TTL mykey
        (integer) -1
    ```

- `PTTL key`
以毫秒为单位返回 key 的剩余的过期时间。

- `TTL key`
以秒为单位，返回给定 key 的剩余生存时间(TTL, time to live)。

    示例
    ```text
        # 不存在的 key
        
        redis> FLUSHDB
        OK
        
        redis> TTL key
        (integer) -2
        
        
        # key 存在，但没有设置剩余生存时间
        
        redis> SET key value
        OK
        
        redis> TTL key
        (integer) -1
        
        
        # 有剩余生存时间的 key
        
        redis> EXPIRE key 10086
        (integer) 1
        
        redis> TTL key
        (integer) 10084
    ```

- `RANDOMKEY`
从当前数据库中随机返回一个 key 。

- `RENAME key newkey`
修改 key 的名称

- `RENAMENX key newkey`
仅当 newkey 不存在时，将 key 改名为 newkey 。

- `TYPE key`
返回 key 所储存的值的类型。     

- key的命名规范

    redis 单个key存入512M大小
    
    - key不要太长，尽量不要超过1024字节，这不仅消耗内存，而且降低查找的效率
    - key也不要太短，太短的话，key的可读性会降低
    - 在一个项目中，key最好使用统一的命名模式，例如：user:123:password(推荐":"，不建议"_"，因为程序里面有的变量是以下划线连接的)


























