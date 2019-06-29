###### datetime:2019/6/10 15:35
###### author:nzb

## Django模板系统

[官方文档](https://docs.djangoproject.com/en/1.11/ref/templates/builtins/#std:templatetag-for)

### 常见语法

只需要记两种特殊符号：

{{  }}和 {% %}

变量相关的用{{}}，逻辑相关的用{%%}。

#### 变量

在Django的模板语言中按此语法使用：{{ 变量名 }}。

当模版引擎遇到一个变量，它将计算这个变量，然后用结果替换掉它本身。 变量的命名包括任何字母数字以及下划线 ("_")的组合。 变量名称中不能有空格或标点符号。

点（.）在模板语言中有特殊的含义。当模版系统遇到点(".")，它将以这样的顺序查询：

字典查询（Dictionary lookup）
属性或方法查询（Attribute or method lookup）
数字索引查询（Numeric index lookup）

注意事项：

如果计算结果的值是可调用的，它将被无参数的调用。 调用的结果将成为模版的值。
如果使用的变量不存在， 模版系统将插入 string_if_invalid 选项的值， 它被默认设置为'' (空字符串) 。

```python
    # 几个例子：
    
    # view中代码：
        def template_test(request):
        l = [11, 22, 33]
        d = {"name": "alex"}
    
        class Person(object):
            def __init__(self, name, age):
                self.name = name
                self.age = age
    
            def dream(self):
                return "{} is dream...".format(self.name)
    
        Alex = Person(name="Alex", age=34)
        Egon = Person(name="Egon", age=9000)
        Eva_J = Person(name="Eva_J", age=18)
    
        person_list = [Alex, Egon, Eva_J]
        return render(request, "template_test.html", {"l": l, "d": d, "person_list": person_list})

    # 模板中支持的写法：
        {# 取l中的第一个参数 #}
        {{ l.0 }}
        {# 取字典中key的值 #}
        {{ d.name }}
        {# 取对象的name属性 #}
        {{ person_list.0.name }}
        {# .操作只能调用不带参数的方法 #}
        {{ person_list.0.dream }}
```

#### Filters(过滤器)

在Django的模板语言中，通过使用 过滤器 来改变变量的显示。

过滤器的语法： {{ value|filter_name:参数 }}

使用管道符"|"来应用过滤器。

例如：{{ name|lower }}会将name变量应用lower过滤器之后再显示它的值。lower在这里的作用是将文本全都变成小写。

注意事项：

过滤器支持“链式”操作。即一个过滤器的输出作为另一个过滤器的输入。
过滤器可以接受参数，例如：{{ sss|truncatewords:30 }}，这将显示sss的前30个词。
过滤器参数包含空格的话，必须用引号包裹起来。比如使用逗号和空格去连接一个列表中的元素，如：{{ list|join:', ' }}
'|'左右没有空格没有空格没有空格
 

Django的模板语言中提供了大约六十个内置过滤器。

##### default

如果一个变量是false或者为空，使用给定的默认值。 否则，使用变量的值。

{{ value|default:"nothing"}}

##### length

返回值的长度，作用于字符串和列表。

{{ value|length }}

返回value的长度，如 value=['a', 'b', 'c', 'd']的话，就显示4.

##### filesizeformat

将值格式化为一个 “人类可读的” 文件尺寸 （例如 '13 KB', '4.1 MB', '102 bytes', 等等）。例如：

{{ value|filesizeformat }}
如果 value 是 123456789，输出将会是 117.7 MB。

##### slice

切片

{{value|slice:"2:-1"}}

##### date

格式化

{{ value|date:"Y-m-d H:i:s"}}
 可用的参数：

| 格式化字符|描述|示例输出|
|---|-----|-----|
| a |	'a.m.'或'p.m.'（请注意，这与PHP的输出略有不同，因为这包括符合Associated Press风格的期间）|	'a.m.'|
| A	|'AM'或'PM'。	|'AM'|
| b	|月，文字，3个字母，小写。	|'jan'|
| B	|未实现。|	 |
| c	|ISO 8601格式。 （注意：与其他格式化程序不同，例如“Z”，“O”或“r”，如果值为naive datetime，则“c”格式化程序不会添加时区偏移量（请参阅datetime.tzinfo） 。	|2008-01-02T10:30:00.000123+02:00或2008-01-02T10:30:00.000123如果datetime是天真的|
| d	|月的日子，带前导零的2位数字。	|'01'到'31'|
| D	|一周中的文字，3个字母。	|“星期五”|
| e	|时区名称 可能是任何格式，或者可能返回一个空字符串，具体取决于datetime。	|''、'GMT'、'-500'、'US/Eastern'等|
| E	|月份，特定地区的替代表示通常用于长日期表示。	|'listopada'（对于波兰语区域，而不是'Listopad'）|
| f	|时间，在12小时的小时和分钟内，如果它们为零，则分钟停留。 专有扩展。	|'1'，'1:30'|
| F	|月，文，长。	'一月'
| g	|小时，12小时格式，无前导零。	|'1'到'12'|
| G	|小时，24小时格式，无前导零。	|'0'到'23'|
| h	|小时，12小时格式。	|'01'到'12'|
| H	|小时，24小时格式。	|'00'到'23'|
| i	|分钟。	'00'到'59'
| I	|夏令时间，无论是否生效。	|'1'或'0'|
| j	|没有前导零的月份的日子。	|'1'到'31'|
| l	|星期几，文字长。	|'星期五'|
| L	|布尔值是否是一个闰年。	|True或False|
| m	|月，2位数字带前导零。	|'01'到'12'|
| M	|月，文字，3个字母。	|“扬”|
| n	|月无前导零。	|'1'到'12'|
| N	|美联社风格的月份缩写。 专有扩展。	|'Jan.'，'Feb.'，'March'，'May'|
| o	|ISO-8601周编号，对应于使用闰年的ISO-8601周数（W）。 对于更常见的年份格式，请参见Y。	|'1999年'|
| O	|与格林威治时间的差异在几小时内。	'+0200'
| P	|时间为12小时，分钟和'a.m。'/'p.m。'，如果为零，分钟停留，特殊情况下的字符串“午夜”和“中午”。 专有扩展。	|'1 am'，'1:30 pm' / t3>，'midnight'，'noon'，'12：30 pm' / T10>|
| r	|RFC 5322格式化日期。	|'Thu, 21 Dec 2000 16:01:07 +0200'|
| s	|秒，带前导零的2位数字。	|'00'到'59'|
| S	|一个月的英文序数后缀，2个字符。	|'st'，'nd'，'rd'或'th'|
| t	|给定月份的天数。	|28 to 31|
| T	|本机的时区。	|'EST'，'MDT'|
| u	|微秒。	|000000 to 999999|
| U	|自Unix Epoch以来的二分之一（1970年1月1日00:00:00 UTC）。	| |
| w	|星期几，数字无前导零。	|'0'（星期日）至'6'（星期六）|
| W	|ISO-8601周数，周数从星期一开始。	|1，53|
| y	|年份，2位数字。	|'99'|
| Y	|年，4位数。	|'1999年'|
| z	|一年中的日子 |0到365|
| Z	|时区偏移量，单位为秒。 UTC以西时区的偏移量总是为负数，对于UTC以东时，它们总是为正。	|-43200到43200|

##### safe

Django的模板中会对HTML标签和JS等语法标签进行自动转义，原因显而易见，这样是为了安全。但是有的时候我们可能不希望这些HTML元素被转义，比如我们做一个内容管理系统，后台添加的文章中是经过修饰的，这些修饰可能是通过一个类似于FCKeditor编辑加注了HTML修饰符的文本，如果自动转义的话显示的就是保护HTML标签的源文件。为了在Django中关闭HTML的自动转义有两种方式，如果是一个单独的变量我们可以通过过滤器“|safe”的方式告诉Django这段代码是安全的不必转义。

比如：

value = "<a href='#'>点我</a>"

{{ value|safe}}

##### truncatechars

如果字符串字符多于指定的字符数量，那么会被截断。截断的字符串将以可翻译的省略号序列（“...”）结尾。

参数：截断的字符数

{{ value|truncatechars:9}}

##### truncatewords

在一定数量的字后截断字符串。

{{ value|truncatewords:9}}

##### cut

移除value中所有的与给出的变量相同的字符串

{{ value|cut:' ' }}
如果value为'i love you'，那么将输出'iloveyou'.

##### join

使用字符串连接列表，例如Python的str.join(list)

##### timesince

将日期格式设为自该日期起的时间（例如，“4天，6小时”）。

采用一个可选参数，它是一个包含用作比较点的日期的变量（不带参数，比较点为现在）。 例如，如果blog_date是表示2006年6月1日午夜的日期实例，并且comment_date是2006年6月1日08:00的日期实例，则以下将返回“8小时”：

{{ blog_date|timesince:comment_date }}
分钟是所使用的最小单位，对于相对于比较点的未来的任何日期，将返回“0分钟”。

##### timeuntil

似于timesince，除了它测量从现在开始直到给定日期或日期时间的时间。 例如，如果今天是2006年6月1日，而conference_date是保留2006年6月29日的日期实例，则{{ conference_date | timeuntil }}将返回“4周”。

使用可选参数，它是一个包含用作比较点的日期（而不是现在）的变量。 如果from_date包含2006年6月22日，则以下内容将返回“1周”：

{{ conference_date|timeuntil:from_date }}

##### 自定义filter

自定义过滤器只是带有一个或两个参数的Python函数:
    
变量（输入）的值 - -不一定是一个字符串
参数的值 - 这可以有一个默认值，或完全省略
例如，在过滤器{{var | foo:'bar'}}中，过滤器foo将传递变量var和参数“bar”。
    
自定义filter代码文件摆放位置：

    app01/
        __init__.py
        models.py
        templatetags/  # 在app01下面新建一个package package
            __init__.py
            app01_filters.py  # 建一个存放自定义filter的文件
        views.py
        
编写自定义filter:

    from django import template
    register = template.Library()
    
    
    @register.filter(name="cut")
    def cut(value, arg):
        return value.replace(arg, "")
    
    
    @register.filter(name="addSB")
    def add_sb(value):
        return "{} SB".format(value)

使用自定义filter:
    
    {# 先导入我们自定义filter那个文件 #}
    {% load app01_filters %}
    
    {# 使用我们自定义的filter #}
    {{ somevariable|cut:"0" }}
    {{ d.name|addSB }}

#### Tags

##### for循环

1. 普通for循环

<ul>
{% for user in user_list %}
    <li>{{ user.name }}</li>
{% endfor %}
</ul>

for循环可用的一些参数：

|Variable|	Description|
|--------|-------------|
|forloop.counter	|当前循环的索引值（从1开始）|
|forloop.counter0	|当前循环的索引值（从0开始）|
|forloop.revcounter	|当前循环的倒序索引值（从1开始）|
|forloop.revcounter0	|当前循环的倒序索引值（从0开始）|
|forloop.first	|当前循环是不是第一次循环（布尔值）|
|forloop.last	|当前循环是不是最后一次循环（布尔值）|
|forloop.parentloop	|本层循环的外层循环|

2. for ... empty

<ul>
{% for user in user_list %}
    <li>{{ user.name }}</li>
{% empty %}
    <li>空空如也</li>
{% endfor %}
</ul>

##### if判断

if,elif和else

    {% if user_list %}
      用户人数：{{ user_list|length }}
    {% elif black_list %}
      黑名单数：{{ black_list|length }}
    {% else %}
      没有用户
    {% endif %}

当然也可以只有if和else

    {% if user_list|length > 5 %}
      七座豪华SUV
    {% else %}
        黄包车
    {% endif %}
if语句支持 and 、or、==、>、<、!=、<=、>=、in、not in、is、is not判断。

##### with

定义一个中间变量，多用于给一个复杂的变量起别名。

注意等号左右不要加空格。

{% with total=business.employees.count %}
    {{ total }} employee{{ total|pluralize }}
{% endwith %}
或

{% with business.employees.count as total %}
    {{ total }} employee{{ total|pluralize }}
{% endwith %}

##### csrf_token

这个标签用于跨站请求伪造保护。

在页面的form表单里面写上{% csrf_token %}

##### 注释

{# ... #}

##### 注意事项

1. Django的模板语言不支持连续判断，即不支持以下写法：

{% if a > b > c %}
...
{% endif %}
 

2. Django的模板语言中属性的优先级大于方法

def xx(request):
    d = {"a": 1, "b": 2, "c": 3, "items": "100"}
    return render(request, "xx.html", {"data": d})
如上，我们在使用render方法渲染一个页面的时候，传的字典d有一个key是items并且还有默认的 d.items() 方法，此时在模板语言中:

{{ data.items }}
默认会取d的items key的值。

#### 母版

```html
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta http-equiv="x-ua-compatible" content="IE=edge">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>Title</title>
      {% block page-css %}
      
      {% endblock %}
    </head>
    <body>
    
    <h1>这是母板的标题</h1>
    
    {% block page-main %}
    
    {% endblock %}
    <h1>母板底部内容</h1>
    {% block page-js %}
    
    {% endblock %}
    </body>
    </html>
```

#### 继承母版

在子页面中在页面最上方使用下面的语法来继承母板。

{% extends 'layouts.html' %}

#### 块(block)

通过在母板中使用{% block  xxx %}来定义"块"。

在子页面中通过定义母板中的block名来对应替换母板中相应的内容。

{% block page-main %}
  <p>世情薄</p>
  <p>人情恶</p>
  <p>雨送黄昏花易落</p>
{% endblock %}

#### 组件

可以将常用的页面内容如导航条，页尾信息等组件保存在单独的文件中，然后在需要使用的地方按如下语法导入即可。

{% include 'navbar.html' %}

#### 静态文件相关

##### {% static %}

{% load static %}
<img src="{% static "images/hi.jpg" %}" alt="Hi!" />

引用JS文件时使用：

    {% load static %}
    <script src="{% static "mytest.js" %}"></script>
    
某个文件多处被用到可以存为一个变量

    {% load static %}
    {% static "images/hi.jpg" as myphoto %}
    <img src="{{ myphoto }}"></img>

##### {% get_static_prefix %} 

    {% load static %}
    <img src="{% get_static_prefix %}images/hi.jpg" alt="Hi!" />
    
    或者
    
    {% load static %}
    {% get_static_prefix as STATIC_PREFIX %}
    
    <img src="{{ STATIC_PREFIX }}images/hi.jpg" alt="Hi!" />
    <img src="{{ STATIC_PREFIX }}images/hi2.jpg" alt="Hello!" />

#### simple_tag

和自定义filter类似，只不过接收更灵活的参数。

定义注册simple tag

    @register.simple_tag(name="plus")
    def plus(a, b, c):
        return "{} + {} + {}".format(a, b, c)
        
使用自定义simple tag

    {% load app01_demo %}
    
    {# simple tag #}
    {% plus "1" "2" "abc" %}

#### inclusion_tag

多用于返回html代码片段
示例：
templatetags/my_inclusion.py

    from django import template

    register = template.Library()
   
    @register.inclusion_tag('result.html')
    def show_results(n):
        n = 1 if n < 1 else int(n)
        data = ["第{}项".format(i) for i in range(1, n+1)]
        return {"data": data}


templates/snippets/result.html

    <ul>
      {% for choice in data %}
        <li>{{ choice }}</li>
      {% endfor %}
    </ul>
    
    
templates/index.html

    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta http-equiv="x-ua-compatible" content="IE=edge">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>inclusion_tag test</title>
    </head>
    <body>
    
    {% load inclusion_tag_test %}
    
    {% show_results 10 %}
    </body>
    </html>

 