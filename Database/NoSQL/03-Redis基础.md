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
该命令用于在 key 存在时删除 key(所有类型都可以使用)。

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

### [Redis 字符串(String)](https://www.runoob.com/redis/redis-strings.html)

- `SET key value`
SET 命令用于设置给定 key 的值。如果 key 已经存储其他值， SET 就覆写旧值，且无视类型。

- `GET key`
Get 命令用于获取指定 key 的值。如果 key 不存在，返回 nil 。如果key 储存的值不是字符串类型，返回一个错误。

- `SETNX key value`  
只有在 key 不存在时设置 key 的值。应用于解决分布式锁方案之一

- `INCR key`
将 key 中储存的数字值增一。

- `INCRBY key increment`
将 key 所储存的值加上给定的增量值（increment） 。

- `DECR key`
将 key 中储存的数字值减一。

- `DECRBY key decrement`
key 所储存的值减去给定的减量值（decrement） 。

- `INCRBYFLOAT key increment`
将 key 所储存的值加上给定的浮点增量值（increment） 。

- `APPEND key value`
如果 key 已经存在并且是一个字符串， APPEND 命令将指定的 value 追加到该 key 原来值（value）的末尾。

- `GETRANGE key start end`
返回 key 中字符串值的子字符

- `GETSET key value`
将给定 key 的值设为 value ，并返回 key 的旧值(old value)。

- `GETBIT key offset`
对 key 所储存的字符串值，获取指定偏移量上的位(bit)。

- `MGET key1 [key2..]`
获取所有(一个或多个)给定 key 的值。

- `SETBIT key offset value`
对 key 所储存的字符串值，设置或清除指定偏移量上的位(bit)。

- `SETEX key seconds value`
将值 value 关联到 key ，并将 key 的过期时间设为 seconds (以秒为单位)。

- `SETRANGE key offset value`
用 value 参数覆写给定 key 所储存的字符串值，从偏移量 offset 开始。

- `STRLEN key`
返回 key 所储存的字符串值的长度。

- `MSET key value [key value ...]`
同时设置一个或多个 key-value 对。

- `MSETNX key value [key value ...]`
同时设置一个或多个 key-value 对，当且仅当所有给定 key 都不存在。

- `PSETEX key milliseconds value`
这个命令和 SETEX 命令相似，但它以毫秒为单位设置 key 的生存时间，而不是像 SETEX 命令那样，以秒为单位。

- 应用场景

    - String通常应用于保存单个字符串或json字符串数据
    - 因String是二进制安全的，所有完全可以把一个图片文件的内容作为字符串来存储
    - 计算器(通常key-value缓存一样，常规计数：微博数，粉丝数) INCR等指令就具有原子操作的特性，所有完全可以利用redis的INCR、INCRBY、DECR、DECRBY等指令来实现原子计数的效果。

### [Redis 哈希(Hash)](https://www.runoob.com/redis/redis-hashes.html)

```text
    Redis hash 是一个 string 类型的 field 和 value 的映射表，hash 特别适合用于存储对象。
    
    Redis 中每个 hash 可以存储 232 - 1 键值对（40多亿）。
```

- `HDEL key field1 [field2]`
删除一个或多个哈希表字段

- `HEXISTS key field`
查看哈希表 key 中，指定的字段是否存在。

- `HGET key field`
获取存储在哈希表中指定字段的值。

- `HGETALL key`
获取在哈希表中指定 key 的所有字段和值

- `HINCRBY key field increment`
为哈希表 key 中的指定字段的整数值加上增量 increment 。

- `HINCRBYFLOAT key field increment`
为哈希表 key 中的指定字段的浮点数值加上增量 increment 。

- `HKEYS key`
获取所有哈希表中的字段

- `HLEN key`
获取哈希表中字段的数量

- `HMGET key field1 [field2]`
获取所有给定字段的值

- `HMSET key field1 value1 [field2 value2 ]`
同时将多个 field-value (域-值)对设置到哈希表 key 中。

- `HSET key field value`
将哈希表 key 中的字段 field 的值设为 value 。

- `HSETNX key field value`
只有在字段 field 不存在时，设置哈希表字段的值。

- `HVALS key`
获取哈希表中所有值

- `HSCAN key cursor [MATCH pattern] [COUNT count]`
迭代哈希表中的键值对。

- 应用场景
    
    - 常用与存储一个对象
    - 为什么不用String存储一个对象？
    
        因为hash是最接近关系数据库结果的数据类型，可以将数据库一条记录或程序中一个对象转换成hashmap存放在redis中

###[Redis 列表(List)](https://www.runoob.com/redis/redis-lists.html)

```text
    Redis列表是简单的字符串列表，按照插入顺序排序。你可以添加一个元素到列表的头部（左边）或者尾部（右边）
    
    一个列表最多可以包含 232 - 1 个元素 (4294967295, 每个列表超过40亿个元素)。
```
- `BLPOP key1 [key2 ] timeout`
移出并获取列表的第一个元素， 如果列表没有元素会阻塞列表直到等待超时或发现可弹出元素为止。

- `BRPOP key1 [key2 ] timeout`
移出并获取列表的最后一个元素， 如果列表没有元素会阻塞列表直到等待超时或发现可弹出元素为止。

- `BRPOPLPUSH source destination timeout`
从列表中弹出一个值，将弹出的元素插入到另外一个列表中并返回它； 如果列表没有元素会阻塞列表直到等待超时或发现可弹出元素为止。

- `LINDEX key index`
通过索引获取列表中的元素

- `LINSERT key BEFORE|AFTER pivot value`
在列表的元素前或者后插入元素

- `LLEN key`
获取列表长度

- `LPOP key`
移出并获取列表的第一个元素

- `LPUSH key value1 [value2]`
将一个或多个值插入到列表头部

- `LPUSHX key value`
将一个值插入到已存在的列表头部

- `LRANGE key start stop`
获取列表指定范围内的元素

- `LREM key count value`
移除列表元素

- `LSET key index value`
通过索引设置列表元素的值

- `LTRIM key start stop`
对一个列表进行修剪(trim)，就是说，让列表只保留指定区间内的元素，不在指定区间之内的元素都将被删除。

- `RPOP key`
移除列表的最后一个元素，返回值为移除的元素。

- `RPOPLPUSH source destination`
移除列表的最后一个元素，并将该元素添加到另一个列表并返回

- `RPUSH key value1 [value2]`
在列表中添加一个或多个值

- `RPUSHX key value`
为已存在的列表添加值

- 应用场景

    - 对数据量大的集合数据删减：列表数据显示、关注列表、粉丝列表、留言评价等...分页、热点新闻(top)等。利用LANGE还可以很方便的实现分页的功能，在博客系统中，每片博文的评论也可以存入一个单独的list中。










