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
        2.4、修改表
            2.4.1、添加字段
                alter table tb_name add [column] 新字段名 数据类型 [约束条件] [first | after 已有字段名];
            2.4.2、修改字段
                # 可改指定列的名称和数据类型，修改多个彼此用逗号分隔
                alter table tb_name change [column] 原字段名 新字段名 数据类型 [约束条件];
                # 修改或删除指定列的默认值
                alter table tb_name alter [column] 字段名 {set | drop} default;
                # 修改列的数据类型，而不改名
                alter table tb_name modify [column] 字段名 数据类型 [约束条件] [first | after 已有字段名];
            2.4.3、删除字段
                alter table tb_name drop [column] 字段名;
                
        2.5、重命名表
            alter table 原表名 rename [to] 新表名;
            或
            rename table 原表名1 to 新表名1 [,原表名2 to 新表名2]......;
        2.6、删除表
            drop table [if exists] tb_name1 [,tb_name2]......;
## 3、数据的完整性约束
    3.1、定义实体完整性
        3.1.1、主键约束
            一个表必须要有一个主键，且唯一不为空
        3.2.2、完整性约束的命名
            constraint <约束名> {primary key(主键字段列表) | unique(候选键字段列表) |foreign key(外键字段列表) references tb_被参照的关系(表) (主键字段列表) | check(约束条件表达式)};
    3.2、定义参照完整性
        外键需存在或为空
    3.3、用户定义完整性
        MySQL支持的几种用户定义完整性约束：非空约束，check约束和触发器
        check约束：
            check(expr);
    3.4、更新完整性约束
        3.4.1、删除约束
            alter table <表名> drop foreign key <外键约束名>;
            alter table <表名> drop primary key;
            alter table <表名> drop {约束名 | 候选键字段名};
        3.4.2、添加约束
            alter table <表名> add [constraint <约束名>] primary key(主键字段);
            alter table <表名> add [constraint <约束名>] foreign key(外键字段名) references 被参照表(主键字段名);
            alter table <表名> add [constraint <约束名>] unique key(字段名);
# 四、数据查询
## 1、select语句
        select [all | distinct | distinctrow] <目标表达式1>[,目标表达式2]...
        form <表名1 或 视图1>[,<表名2 或视图2]...
        [where <条件表达式>]
        [group by <列名1> [having <条件表达式>]]
        [order by <列名2> [asc | desc]]
        [limit[m,]n]
## 2、单表查询
        2.1、选择字段
            select 目标表达式1, 目标表达式2,...,目标表达式n from 表名;
            select * form 表名;
            # 定义字段的别名
            select 字段名 as 字段别名 from 表名;
        2.2、选择指定的字段
            select 目标表达式1, 目标表达式2, ... , 目标表达式n  from 表名 where 查询条件;
            # 带between ...and...
            select 目标表达式1, 目标表达式2, ... , 目标表达式n  from 表名 where expression [not] between expr1 and expr2;
            #带like关键字,换码字符也叫转义字符，如果字符串本身含有通配符_和%,就需要换码字符
            select 目标表达式1, 目标表达式2, ... , 目标表达式n  from 表名 where 字段名 [not] like '<匹配字符串>'[escape '<换码字符>'];
            #使用正则表达式查询
            select 目标表达式1, 目标表达式2, ... , 目标表达式n  from 表名 where 字段名 [not] [regexp | rlike] <正则表达式>;
            # 限制查询结果数目
            limit 行数 offset 位置偏移数;
## 3、分组聚合查询
        3.1、使用聚合函数
            group by 字段列表 having <条件表达式>;
## 4、连接查询
        4.1、交叉查询（笛卡尔积）:用得极少
            select * from tb_name1 cross join tb_name2;
            或
            select * from tb_name1, tb_name2;
        4.2、内连接
            select 目标表达式1, 目标表达式2, ... , 目标表达式n from table1 [as] 别名1 [inner] join table2 [as] 别名2 on 连接条件 [where 过滤条件];
            select 目标表达式1, 目标表达式2, ... , 目标表达式n from table1, table2 where 连接条件 [and 过滤条件];
            4.2.1、等值于非等值连接
                [<table1>.]<字段名1> <比较运算符> [<table2>.]<字段名2>;
            4.2.2、自连接
                使用自连接时，需要指定多个不同的别名，查询字段都有别名来限定
            4.2.3、自然连接
                两张表中的字段名都相同才可以使用，否则放回笛卡尔积结果
                select 目标表达式1, 目标表达式2, ... , 目标表达式n from table1 [as] 别名1 natural join table2 [as] 别名2;
        4.3、外连接
            # 左外连接
            select 目标表达式1, 目标表达式2, ... , 目标表达式n from table1 [as] 别名1 left [outer] join table2 [as] 别名2 on 连接条件 [where 过滤条件];
            # 右外连接
            select 目标表达式1, 目标表达式2, ... , 目标表达式n from table1 [as] 别名1 right [outer] join table2 [as] 别名2 on 连接条件 [where 过滤条件];
## 5、子查询
        子查询关键字：in, any, all, [not]exists, 
        必要时为表名加上别名         
## 6、联合查询
        select -from-where
        union [all]
        select -from-where
        [...union [all]
        select -from-where]
        多个表查询联合起来，不使用all关键字，执行的时候去重，返回的行都是唯一的，使用all则不去重。
# 五、数据更新

## 1、插入数据
    
        1.1、插入一条或多条
        insert into table(字段名列表) values(值列表1), [,值列表2], ... ,[值列表n];
        replace into table(字段名列表) values(值列表1), [,值列表2], ... ,[值列表n];
        1.2、插入查询结果
        insert into table1(字段名列表) select (字段名列表) from table2 where(conditions);
## 2、修改数据记录
    
        update table set 字段名1=值1, 字段名2=值2, ... , 字段名n=值n [where <conditions>];
## 3、删除数据记录
    
        delete from table [where<conditions>];
        truncate [table] tb_name;
# 六、索引

## 1、查看数据表上的索引
    
        show {index | indexes | keys} {fron | in } tb_name [{from | in } db_name];
## 2、创建索引
    
        2.1、建表时创建
            create table(字段名, 字段类型...) [constrint index_name] [unique] [index | key] [index_name](字段列名[长度]) [asc | aesc];
        2.2、使用create index
            create [unique] index index_name on tb_name(字段名[(长度)] [asc | desc] , ...);
        2.3、使用alter table
            alter table tb_name add [unique | fulltext] [index|key] index_name(字段名[(长度)][asc|desc],...)
## 3、删除索引
    
        drop index index_name on tb_name;
        alter table tb_name drop index index_name;
# 七、视图

## 1、创建视图
    
        create [or replace] view view_name[(column_list)] as select_statement [with [cascaded | local] check option];
## 2、删除视图
    
        drop view [if exists] view_name[,view_name]...;
## 3、修改视图
    
        alter view view_name [(column_list)] as select_statement [with [cascaded | local] check option];
## 4、查看视图定义
    
        show create view view_name\G
# 八、触发器

## 1、创建触发器
    
        create trigger trigger_name trigger_time(before|after) trigger_event(insert|update|delete) on tb_name for each row trigger_body;
        # 查看已有的触发器
            show triggers [{from | in} db_name]; 
## 2、删除触发器
    
        drop trigger [if exists] [schema_name.]trigger_name;
## 3、使用触发器
    
        在insert触发器中可以引用名为new的虚拟表来访问被插入的行
        在delete触发器中可以引用名为old的虚拟表来访问被删除的行
        在delete触发器中可以引用名为new的虚拟表来访问更新后的值
        在delete触发器中可以引用名为old的虚拟表来访问更新前的值
        例子：
            create trigger tg1 after update on tb_name1 for each row set new.col1 = old.col2; 
# 九、事件(临时触发器)

## 1、创建事件
    
        create event [if not exists] event_name on schedule 时间调度 [enable|disable|disable on slave] do event_body;
        时间调度语法格式：
            at timestamp [+ interval interval]... 
            | every interval 
            [starts timestamp [+ interval interval]...]
            [ends timestamp [+ interval interval]...]    
        interval语法格式
            quantity {year | quarter | month | day | hour | minute | week | second | year_month | day_hour | day_minute| day_second | hour_minute | hour_second | minute_second}
        例子：
            每个月向表tb_1插入一条数据，该事件开始于下个月并且结束于2019年12月31日。
                首先改变结束符：delimiter $$
                create event if not exists event_insert
                    on schedule every 1 month
                        starts curdate() + interval 1 month
                        ends  '2019-12-31'
                        do
                        begin
                            if year(curdate()) < 2013 then
                                insert into tb_1 values('aa','aa','b');
                            endif;
                        end$$       
## 2、修改事件
    
        alter event event_name 
            [on schedule 时间调度]
            [rename to new_name]
            [enable | disable | disable on slave]
            [do event_body];
## 3、删除事件
    
        drop event [if exists] event_name;    
# 十、存储过程和存储函数

## 1、存储过程
    
        1.1、创建存储过程
            create procedure sp_name ([proc_parameter[,...]]) [characteristic...]routine_body
            其中proc_parameter格式为
                [in | out | inout]param_name type
                分别对应输入、输出和输入/输出参数
            其中routine_body为存储过程主体：
                也称存储过程体，其中包含了在过程调用的时候必须执行的sql语句，这个部分以关键字begin开始，以关键字end结束。如存储过程体中只有一条sql语句，可以省略begin-end标志，另外begin-end可以嵌套。
        1.2、存储过程体
            1.2.1、局部变量
                声明：
                    declare var_name [,...] type [defautl value]                
                ps:局部变量不同意用户变量，区别：局部变量声明时，在其前面没有使用“@”符号，并且它只能在声明它的begin-end语句块中使用，而用户变量前面使用“@”符号，存在整个会话中。
            1.2.2、赋值
                set var_name = expr[, var_name = expr]...
                或
                select 字段名 into var_name[,...] 查询源及条件;
            1.2.3、流程控制语句
                条件
                    if search_condition then statement_list
                        [elseif search_condition then statement_list]...
                        [else statement_list]
                    endif;
                    或
                    case case_vale
                        when when_value then statement_list
                        [when when_value then statement_list]
                        [else statement_list]
                    endcase;
                    或
                    case
                        when search_condition then statement_list
                        [when search_condition then statement_list]
                        [else statement_list]
                    endcase;
                循环
                    [begin_label:]while search_condition do
                        statement_list
                    end while[end_label]
                    或
                    [begin_label:]repeat
                        statement_list
                    until search_condition
                    end repeat[end_label]
                    或
                    [begin_label:]loop
                        statement_list
                    end loop[end_label]
            1.2.4、游标
                # 声明游标
                    declare cursor_name cursor for select_statement;
                    # select_statement为一条select语句注意不能有into子句。
                # 打开游标
                    open cursor_name
                # 读取游标
                    fetch cursor_name into var_name[,var_name]...
                # 关闭游标
                    close cursor_name
                # 例子:统计行数
                    delimiter $$
                    create procedure sp_sum(out rows int)
                        begin
                            declare sno char;
                            declare found boolean default true;
                            declare cur cursor for select studentNo from tb_students;
                            declare continue handler for not found
                                set found = false;
                            set rows = 0;
                            open cur;
                            fetch cur into sno;
                            while found do
                                set rows = rows +1;
                                fetch cur into sno;
                            end while;
                            close cur;
                        end$$
        1.3、调用存储过程
            call sp_name([parameter[,...]]);
            call sp_name[()];
        1.4、删除存储过程
            drop procedure function [if exists] sp_name;
## 2、存储函数
    
        2.1、与存储过程的区别
            # 存储函数不能有输出参数，因为存储函数本身就是输出参数，而存储过程可以有输出参数
            # 可以直接对存储函数进行调用，不需要使用call
            # 存储函数必须包含一条return语句，而这条语句不允许包含于存储过程中
        2.1、创建存储函数
            create function sp_name ([func_parameter[,...]]) 
                returns type
                routine_body
            # 例子：根据给定的学号返回学生性别，如果没有则返回“没有该学生”
                delimiter $$
                create function fn_student(sno char(10))
                    returns char(2)
                    begin
                        declare ssex char(2);
                        select sex into ssex from student where studentNo = sno;
                        if ssex is null then
                            return (select "没有该学生");
                        else if ssex="女" then
                            return (select "女");
                            else return (select "男");
                        end if;
                    end$$
        2.2、调用存储函数
            select sp_name([func_parameter[,...]]);
        2.3、删除存储函数
            drop function [if exists] sp_name;                
            