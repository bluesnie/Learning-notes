# 一、数据库基本概念
## 1、数据库设计的步骤
        六个阶段：需求分析、概念结构设计、逻辑结构设计、物理结构设计、数据库实施、数据库运行与维护
# 二、MySQL编程语言
## 1、MySQL函数
        1.1、聚合函数
            count()：计数（对于除“*”以外的任何参数，返回所选集合中非null值的数目）
            sun()：求和
            avg()：求平均数
            max()：求最大值
            min()：求最小值
        1.2、数学函数
            abs()：求绝对值
            floor()：返回小于或等于参数的最大整数
            rand()：返回0~1之间的随机数
            truncate(x,y)：返回x保留到小数点后y为的值
            sort(): 求参数的平方根
        1.3、字符串函数
            upper()和ucase()：把字符串所有字母变成大写字母
            left(s,n)：返回字符串s的前n个字符
            substring(s,n,len)：从字符串s的第n个位置开始获取长度为len的字符串
        1.4、日期和时间函数
            curdate()和current_date()：返回当前日期
            curtime()和current_time()：获取当前时间
            now()：获取当前日期和时间，current_timestamp()、localtime()、sysdate()、localtimestamp()同样可以获取当前日期和时间
        1.5、其他函数
            if(expr,v1,v2)：条件判断函数，如果表达式expr成立，则执行v1,否则执行v2
            ifnull(v1,v2)：条件判断函数，如果表达式v1不为空，则显示v1的值,否则显示v2的值
            version()：获取数据库的版本号
# 三、数据定义
## 1、定义数据库
        1.1、创建数据库
            create {database | schema} [if not exists] db_name [[default] character set [=] charset_name [[default] collate [=] collation_name];
        1.2、选择和查看数据库
            use da_name;
            show {databases | schemas};
        1.3、修改数据库
            alter {database | schema} [db_name] [[default] character set [=] charset_name [[default] collate [=] collation_name];
            数据库名可省略，表示修改当前数据库
        1.4、删除数据库
            drop {database | schema} [if exists] db_name;
## 2、定义表
        2.1数据类型
            2.1.1、数值类型
                bit
                tinyint
                bool,boolean
                smallint
                mediumint
                int,integer
                bigint
                double
                decimal(m.d)
            2.1.2、日期和时间类型
                date:日期型，MySQL以“YYYY-MM-DD”格式显示date值
                datetime:日期和时间类型，MySQL以“YYYY-MM-DD HH:MM:SS”格式显示datetime值
                timestamp:时间戳
                time:时间型,MySQL以“HH:MM:SS”格式显示time值
                year两位或四位格式的年
            2.1.3、字符串类型
                char：定长字符串
                varchar：可变长字符串
                tinytext
                text
        2.2、创建表
            create table tbl_name (
                    字段名1 数据类型 [列级完整性约束条件] [默认值]
                    [,字段名2 数据类型 [列级完整性约束条件] [默认值]
                    [,... ...]
                    [,表级完整性约束条件]
                    ) [engine=引擎类型];
        2.3、查看表
            2.3.1、查看表名称
                show tables [{from | in } db_name];
            2.3.2、查看数据表的基本结构
                show columns {from | in } tb_name [from | in } db_name];
                或
                desc tb_name;
            2.3.3、查看数据表的详细结构
                show create table tb_name;
                