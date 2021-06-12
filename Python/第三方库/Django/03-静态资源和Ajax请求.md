###### datetime:2019/6/10 11:32
###### author:nzb

## AJAX

### AJAX准备知识：JSON

#### 什么是JSON?

- JSON 指的是 JavaScript 对象表示法（JavaScript Object Notation）
- JSON 是轻量级的文本数据交换格式
- JSON 独立于语言 *
- JSON 具有自我描述性，更易理解
*JSON 使用 JavaScript 语法来描述数据对象，但是 JSON 仍然独立于语言和平台。JSON 解析器和 JSON 库支持许多不同的编程语言。

 啥都别多说了，上图吧！
 ![](./res/ajax_python_js.jpg)

合格的json对象：

    ["one", "two", "three"]
    { "one": 1, "two": 2, "three": 3 }
    {"names": ["张三", "李四"] }
    [ { "name": "张三"}, {"name": "李四"} ]　
    
 不合格的json对象：
 
    { name: "张三", 'age': 32 }  // 属性名必须使用双引号
    [32, 64, 128, 0xFFF] // 不能使用十六进制值
    { "name": "张三", "age": undefined }  // 不能使用undefined
    { "name": "张三",
      "birthday": new Date('Fri, 26 Aug 2011 07:13:10 GMT'),
      "getName":  function() {return this.name;}  // 不能使用函数和日期对象
    }

#### stringify与parse方法

JavaScript中关于JSON对象和字符串转换的两个方法：

JSON.parse(): 用于将一个 JSON 字符串转换为 JavaScript 对象　

    JSON.parse('{"name":"Q1mi"}');
    JSON.parse('{name:"Q1mi"}') ;   // 错误
    JSON.parse('[18,undefined]') ;   // 错误

JSON.stringify(): 用于将 JavaScript 值转换为 JSON 字符串。　

    JSON.stringify({"name":"Q1mi"})

#### 和XML的比较

JSON 格式于2001年由 Douglas Crockford 提出，目的就是取代繁琐笨重的 XML 格式。

JSON 格式有两个显著的优点：书写简单，一目了然；符合 JavaScript 原生语法，可以由解释引擎直接处理，不用另外添加解析代码。所以，JSON迅速被接受，已经成为各大网站交换数据的标准格式，并被写入ECMAScript 5，成为标准的一部分。

XML和JSON都使用结构化方法来标记数据，下面来做一个简单的比较。

用XML表示中国部分省市数据如下：
```xml
    <?xml version="1.0" encoding="utf-8"?>
    <country>
        <name>中国</name>
        <province>
            <name>黑龙江</name>
            <cities>
                <city>哈尔滨</city>
                <city>大庆</city>
            </cities>
        </province>
        <province>
            <name>广东</name>
            <cities>
                <city>广州</city>
                <city>深圳</city>
                <city>珠海</city>
            </cities>
        </province>
        <province>
            <name>台湾</name>
            <cities>
                <city>台北</city>
                <city>高雄</city>
            </cities>
        </province>
        <province>
            <name>新疆</name>
            <cities>
                <city>乌鲁木齐</city>
            </cities>
        </province>
    </country>
```
用JSON表示如下：
```json
    {
        "name": "中国",
        "province": [{
            "name": "黑龙江",
            "cities": {
                "city": ["哈尔滨", "大庆"]
            }
        }, {
            "name": "广东",
            "cities": {
                "city": ["广州", "深圳", "珠海"]
            }
        }, {
            "name": "台湾",
            "cities": {
                "city": ["台北", "高雄"]
            }
        }, {
            "name": "新疆",
            "cities": {
                "city": ["乌鲁木齐"]
            }
        }]
    }
```
由上面的两端代码可以看出，JSON 简单的语法格式和清晰的层次结构明显要比 XML 容易阅读，并且在数据交换方面，由于 JSON 所使用的字符要比 XML 少得多，可以大大得节约传输数据所占用得带宽。 

### AJAX简介

发送请求的方式：
- 直接在地址栏输入URL回车           GET请求
- a标签                           GET请求
- form表单                        GET/POST请求
- AJAX                            GET/POST请求

AJAX（Asynchronous Javascript And XML）翻译成中文就是“异步的Javascript和XML”。即使用Javascript语言与服务器进行异步交互，传输的数据为XML（当然，传输的数据不只是XML）。

AJAX 不是新的编程语言，而是一种使用现有标准的新方法。

AJAX 最大的优点是在不重新加载整个页面的情况下，可以与服务器交换数据并更新部分网页内容。（这一特点给用户的感受是在不知不觉中完成请求和响应过程）

AJAX 不需要任何浏览器插件，但需要用户允许JavaScript在浏览器上执行。

- 同步交互：客户端发出一个请求后，需要等待服务器响应结束后，才能发出第二个请求；
- 异步交互：客户端发出一个请求后，无需等待服务器响应结束，就可以发出第二个请求。

##### 示例

**页面输入两个整数，通过AJAX传输到后端计算出结果并返回。**
```html
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta http-equiv="x-ua-compatible" content="IE=edge">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>AJAX局部刷新实例</title>
    </head>
    <body>
    
    <input type="text" id="i1">+
    <input type="text" id="i2">=
    <input type="text" id="i3">
    <input type="button" value="AJAX提交" id="b1">
    
    <script src="/static/jquery-3.2.1.min.js"></script>
    <script>
      $("#b1").on("click", function () {
        $.ajax({
          url:"/ajax_add/",
          type:"GET",
          data:{"i1":$("#i1").val(),"i2":$("#i2").val()},
          success:function (data) {
            $("#i3").val(data);
          }
        })
      })
    </script>
    </body>
    </html>
```
```python
    def ajax_demo1(request):
        return render(request, "ajax_demo1.html")
    
    
    def ajax_add(request):
        i1 = int(request.GET.get("i1"))
        i2 = int(request.GET.get("i2"))
        ret = i1 + i2
        return JsonResponse(ret, safe=False)
```
```python
    urlpatterns = [
        ...
        url(r'^ajax_add/', views.ajax_add),
        url(r'^ajax_demo1/', views.ajax_demo1),
        ...   
    ]
```

#### AJAX常见应用场景

搜索引擎根据用户输入的关键字，自动提示检索关键字。

还有一个很重要的应用场景就是注册时候的用户名的查重。

其实这里就使用了AJAX技术！当文件框发生了输入变化时，使用AJAX技术向服务器发送一个请求，然后服务器会把查询到的结果响应给浏览器，最后再把后端返回的结果展示出来。

- 整个过程中页面没有刷新，只是刷新页面中的局部位置而已！
- 当请求发出后，浏览器还可以进行其他操作，无需等待服务器的响应！
![](./res/ajax_example.png)
当输入用户名后，把光标移动到其他表单项上时，浏览器会使用AJAX技术向服务器发出请求，服务器会查询名为lemontree7777777的用户是否存在，最终服务器返回true表示名为lemontree7777777的用户已经存在了，浏览器在得到结果后显示“用户名已被注册！”。

- 整个过程中页面没有刷新，只是局部刷新了；
- 在请求发出后，浏览器不用等待服务器响应结果就可以进行其他操作；

#### AJAX的优缺点

优点：
- AJAX使用JavaScript技术向服务器发送异步请求；
- AJAX请求无须刷新整个页面；
- 因为服务器响应内容不再是整个页面，而是页面中的部分内容，所以AJAX性能高； 

### jQuery实现的AJAX

最基本的jQuery发送AJAX请求示例：
```html
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
      <meta charset="UTF-8">
      <meta http-equiv="x-ua-compatible" content="IE=edge">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>ajax test</title>
      <script src="https://cdn.bootcss.com/jquery/3.3.1/jquery.min.js"></script>
    </head>
    <body>
    <button id="ajaxTest">AJAX 测试</button>
    <script>
      $("#ajaxTest").click(function () {
        $.ajax({
          url: "/ajax_test/",
          type: "POST",
          data: {username: "Q1mi", password: 123456},
          success: function (data) {
            alert(data)
          }
        })
      })
    </script>
    </body>
    </html>
```
#### views.py

```python
    def ajax_test(request):
        user_name = request.POST.get("username")
        password = request.POST.get("password")
        print(user_name, password)
        return HttpResponse("OK")
```

#### $.ajax参数

data参数中的键值对，如果值值不为字符串，需要将其转换成字符串类型。
```python
    $("#b1").on("click", function () {
        $.ajax({
          url:"/ajax_add/",
          type:"GET",
          data:{"i1":$("#i1").val(),"i2":$("#i2").val(),"hehe": JSON.stringify([1, 2, 3])},
          success:function (data) {
            $("#i3").val(data);
          }
        })
      })
```

##### $.ajax参数data
- ajax有三种传递传递data的方式：
    - 1、json格式
    - 2、标准参数模式
    - 3、json字符串格式
    
- 1.json对象格式：
```text
    {“username”:”chen”,”nickname”:”alien”}
```
```javascript
    $.ajax({
        type:"post",
        url:"/test/saveUser",
        data:{"username":"chen","nickname":"alien"},
        dataType:"json", 	   //指定响应的data数据类型为JSON对象。
        success: function(data){
            console.log(data);
        }
    });
```
```text
    - 如：当前的Ajax请求是一个POST请求，对请求体中的数据 使用默认的数据编码，格式如：key1 = value2&key2 = value2 a中的数据变成这样的格式：key1 = value2&key2 = value2 ，包装在Http请求体中传送给后台。
    - dataType:"json"
    - dataType:“json” ：用来指定服务器返回的data数据类型必须是JSON类型。然后jQuery就会把后端返回的json字符串尝试通过JSON.parse()解析为js对象。
    - 如果不指定dataType，jQuery 将自动根据 HTTP 包的 MIME 信息来智能判断，若MIME信息的值为JSON，则jQuery会自动的把data数据转换成JS对象的json，接着Script把data传递给回调函数进行JS的脚本操作。
```

- 2、标准参数模式
```text
“username=Liudehua & age=50”
```

```javascript
    $.ajax({
        type:"post",
        url:"/test/saveUser",
        data:"username=chen&nickname=alien",
        dataType:"json", 
        success: function(data){
            console.log(data);
        }
    });
```
```text
 - $(“#form1”).serialize() 就是把表单的数据拼成这个格式（key1 = value2&key2 = value2）的字符串，然后放在Http请求体中传给后台！
```

- 3.json字符串 ————>只用于post请求
```text
    “{“username”:”chen”,”nickname”:”alien”}”————>JSON对象格式的字符串
    JSON.stringify({“username”:”chen”,”nickname”:”alien”})————>把JSON对象转成JSON格式的字符串。
```
```javascript
    $.ajax({
        type:"post",
        url:"/test/saveUser",
        data:JSON.stringify({"username":"chen","nickname":"alien"}),
        contentType:"json/application"
        dataType:"json",
        success: function(data){
            console.log(data);
        }
    });
```

** 第三种这种方式不能用于 Get请求。
    原因：
    
    1、因为此种方式发送的请求，后端必须得用@RequestBody进行接收，且接收的是Http请求体中的数据，Get请求没有请求体。
    
    2、而且此方式的Ajax 必须要添加 contentType:”json/application”这个字段信息。
**

###### 注意：

- 1、若为GET请求，则会把data的数据 附加在 URL 后，
    
    格式如：localhost://findAll ? key1=value1&key2=value2
    若为POST请求，则就会把data的数据 放在请求体中。
    
    格式如：key1 = value2&key2 = value2
    
- 2、dataType：指定服务器端返回的数据类型。
    若不指定，且后端返回的是Json，前端就会自动识别返回的数据是JSON。

### JS实现AJAX

```javascript
    var b2 = document.getElementById("b2");
      b2.onclick = function () {
        // 原生JS
        var xmlHttp = new XMLHttpRequest();
        xmlHttp.open("POST", "/ajax_test/", true);
        xmlHttp.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
        xmlHttp.send("username=q1mi&password=123456");
        xmlHttp.onreadystatechange = function () {
          if (xmlHttp.readyState === 4 && xmlHttp.status === 200) {
            alert(xmlHttp.responseText);
          }
        };
      };
```

### AJAX请求如何设置csrf_token

#### 方式1

通过获取隐藏的input标签中的csrfmiddlewaretoken值，放置在data中发送。
```html
    $.ajax({
      url: "/cookie_ajax/",
      type: "POST",
      data: {
        "username": "Q1mi",
        "password": 123456,
        "csrfmiddlewaretoken": $("[name = 'csrfmiddlewaretoken']").val()  // 使用jQuery取出csrfmiddlewaretoken的值，拼接到data中
      },
      success: function (data) {
        console.log(data);
      }
    })
```

#### 方式2

通过获取返回的cookie中的字符串 放置在请求头中发送。

注意：需要引入一个jquery.cookie.js插件。
```javascript
    $.ajax({
      url: "/cookie_ajax/",
      type: "POST",
      headers: {"X-CSRFToken": $.cookie('csrftoken')},  // 从Cookie取csrftoken，并设置到请求头中
      data: {"username": "Q1mi", "password": 123456},
      success: function (data) {
        console.log(data);
      }
    })
```

或者用自己写一个getCookie方法：
```javascript
    function getCookie(name) {
        var cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            var cookies = document.cookie.split(';');
            for (var i = 0; i < cookies.length; i++) {
                var cookie = jQuery.trim(cookies[i]);
                // Does this cookie string begin with the name we want?
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }
    var csrftoken = getCookie('csrftoken');
```

每一次都这么写太麻烦了，可以使用$.ajaxSetup()方法为ajax请求统一设置。
```javascript
    function csrfSafeMethod(method) {
      // these HTTP methods do not require CSRF protection
      return (/^(GET|HEAD|OPTIONS|TRACE)$/.test(method));
    }
    
    $.ajaxSetup({
      beforeSend: function (xhr, settings) {
        if (!csrfSafeMethod(settings.type) && !this.crossDomain) {
          xhr.setRequestHeader("X-CSRFToken", csrftoken);
        }
      }
    });
```
注意：

如果使用从cookie中取csrftoken的方式，需要确保cookie存在csrftoken值。

如果你的视图渲染的HTML文件中没有包含 { % csrf_token % }，Django可能不会设置CSRFtoken的cookie。

这个时候需要使用ensure_csrf_cookie()装饰器强制设置Cookie。

```python
    django.views.decorators.csrf import ensure_csrf_cookie
    
    
    @ensure_csrf_cookie
    def login(request):
        pass
```
更多细节详见：[Djagno官方文档中关于CSRF的内容](https://docs.djangoproject.com/en/1.11/ref/csrf/)

### AJAX上传文件

XMLHttpRequest 是一个浏览器接口，通过它，我们可以使得 Javascript 进行 HTTP (S) 通信。XMLHttpRequest 在现在浏览器中是一种常用的前后台交互数据的方式。2008年 2 月，XMLHttpRequest Level 2 草案提出来了，相对于上一代，它有一些新的特性，其中 FormData 就是 XMLHttpRequest Level 2 新增的一个对象，利用它来提交表单、模拟表单提交，当然最大的优势就是可以上传二进制文件。下面就具体

首先看一下formData的基本用法：FormData对象，可以把所有表单元素的name与value组成一个queryString，提交到后台。只需要把 form 表单作为参数传入 FormData 构造函数即可：

介绍一下如何利用 FormData 来上传文件。
```javascript
    // 上传文件示例
    $("#b3").click(function () {
      var formData = new FormData();
      formData.append("csrfmiddlewaretoken", $("[name='csrfmiddlewaretoken']").val());
      formData.append("f1", $("#f1")[0].files[0]);
      $.ajax({
        url: "/upload/",
        type: "POST",
        processData: false,  // 告诉jQuery不要去处理发送的数据
        contentType: false,  // 告诉jQuery不要去设置Content-Type请求头
        data: formData,
        success:function (data) {
          console.log(data)
        }
      })
    })
```
或者使用
```javascript
    var form = document.getElementById("form1");
    var fd = new FormData(form);
```
这样也可以直接通过ajax 的 send() 方法将 fd 发送到后台。

**注意：由于 FormData 是 XMLHttpRequest Level 2 新增的接口，现在 低于IE10 的IE浏览器不支持 FormData。**

#### 练习（用户名是否已被注册）

##### 功能介绍

在注册表单中，当用户填写了用户名后，把光标移开后，会自动向服务器发送异步请求。服务器返回这个用户名是否已经被注册过。

##### 案例分析

- 页面中给出注册表单；
- 在username input标签中绑定onblur事件处理函数。
- 当input标签失去焦点后获取 username表单字段的值，向服务端发送AJAX请求；
- django的视图函数中处理该请求，获取username值，判断该用户在数据库中是否被注册，如果被注册了就返回“该用户已被注册”，否则响应“该用户名可以注册”。

### ajax中参数traditional的作用
　　
在使用ajax向后台传值的时候，有的时候一个字段需要传多个值，这种情况下会想到用数组形式来传，比如：
```javascript
    $.ajax({
      type: "post",
      async: true,
      data: {
        "records": ["123","456","789"]
      },
      url: "xxxxx",
      error: function(request) {},
      success: function(data) {}
    });
```

但是通过测试很快就会发现java后台无法取到参数，因为jQuery需要调用jQuery.param序列化参数，jQuery.param(obj, traditional )默认情况下traditional为false，
即jquery会深度序列化参数对象，以适应如PHP和Ruby on Rails框架，但servelt api无法处理，我们可以通过**设置traditional 为true阻止深度序列化**，然后序列化结果如下：

    records: ["123", "456", "789"]    =>    records=123&p=456&p=789

随即，我们就可以在后台通过request.getParameterValues()来获取参数的值数组了，如下：
```javascript
    $.ajax({
      type: "post",
      async: true,
      traditional: true,  
      data: {
        "records": ["123","456","789"]
      },
      url: "xxxxx",
      error: function(request) {},
      success: function(data) {}
    });
```

### 序列化

#### Django内置的serializers

```python
    def books_json(request):
        book_list = models.Book.objects.all()[0:10]
        from django.core import serializers
        ret = serializers.serialize("json", book_list)
        return HttpResponse(ret)
```

### 补充一个SweetAlert插件示例

![Alt Text](https://media.giphy.com/media/H6tOrVbAn72nJws6eG/giphy.gif)

点击下载[Bootstrap-sweetalert项目](https://github.com/lipis/bootstrap-sweetalert)。
```javascript
    $(".btn-danger").on("click", function () {
      swal({
        title: "你确定要删除吗？",
        text: "删除可就找不回来了哦！",
        type: "warning",
        showCancelButton: true,
        confirmButtonClass: "btn-danger",
        confirmButtonText: "删除",
        cancelButtonText: "取消",
        closeOnConfirm: false
        },
        function () {
          var deleteId = $(this).parent().parent().attr("data_id");
          $.ajax({
            url: "/delete_book/",
            type: "post",
            data: {"id": deleteId},
            success: function (data) {
              if (data.status === 1) {
                swal("删除成功!", "你可以准备跑路了！", "success");
              } else {
                swal("删除失败", "你可以再尝试一下！", "error")
              }
            }
          })
        });
    })
```

## 静态资源和Ajax请求(100天)

基于前面两个章节讲解的知识，我们已经可以使用Django框架来实现Web应用的开发了。接下来我们就尝试实现一个投票应用，具体的需求是用户进入应用首先查看到“学科介绍”页面，该页面显示了一个学校所开设的所有学科；通过点击某个学科，可以进入“老师介绍”页面，该页面展示了该学科所有老师的详细情况，可以在该页面上给老师点击“好评”或“差评”，但是会先跳转到“登录页”要求用户登录，登录成功才能投票；对于未注册的用户，可以在“登录页”点击“新用户注册”进入“注册页”完成用户注册，注册成功后会跳转到“登录页”，注册失败会获得相应的提示信息。

### 准备工作

由于之前已经详细的讲解了如何创建Django项目以及项目的相关配置，因此我们略过这部分内容，唯一需要说明的是，从上面对投票应用需求的描述中我们可以分析出三个业务实体：学科、老师和用户。学科和老师之间通常是一对多关联关系（一个学科有多个老师，一个老师通常只属于一个学科），用户因为要给老师投票，所以跟老师之间是多对多关联关系（一个用户可以给多个老师投票，一个老师也可以收到多个用户的投票）。首先修改应用下的models.py文件来定义数据模型，先给出学科和老师的模型。

```Python
from django.db import models


class Subject(models.Model):
    """学科"""
    no = models.AutoField(primary_key=True, verbose_name='编号')
    name = models.CharField(max_length=31, verbose_name='名称')
    intro = models.CharField(max_length=511, verbose_name='介绍')

    def __str__(self):
        return self.name

    class Meta:
        db_table = 'tb_subject'
        verbose_name_plural = '学科'


class Teacher(models.Model):
    """老师"""
    no = models.AutoField(primary_key=True, verbose_name='编号')
    name = models.CharField(max_length=15, verbose_name='姓名')
    gender = models.BooleanField(default=True, choices=((True, '男'), (False, '女')), verbose_name='性别')
    birth = models.DateField(null=True, verbose_name='出生日期')
    intro = models.CharField(max_length=511, default='', verbose_name='')
    good_count = models.IntegerField(default=0, verbose_name='好评数')
    bad_count = models.IntegerField(default=0, verbose_name='差评数')
    photo = models.CharField(max_length=255, verbose_name='照片')
    subject = models.ForeignKey(to=Subject, on_delete=models.PROTECT, db_column='sno', verbose_name='所属学科')

    def __str__(self):
        return self.name

    class Meta:
        db_table = 'tb_teacher'
        verbose_name_plural = '老师'
```

模型定义完成后，可以通过“生成迁移”和“执行迁移”来完成关系型数据库中二维表的创建，当然这需要提前启动数据库服务器并创建好对应的数据库，同时我们在项目中已经安装了PyMySQL而且完成了相应的配置，这些内容此处不再赘述。

```Shell
(venv)$ python manage.py makemigrations vote
...
(venv)$ python manage.py migrate
...
```

> 注意：为了给vote应用生成迁移，需要先修改Django项目的配置文件settings.py，在INSTALLED_APPS中添加vote应用。

完成模型迁移之后，我们可以通过下面的SQL语句来添加学科和老师测试的数据。

```SQL
INSERT INTO `tb_subject` (`no`,`name`,`intro`) 
VALUES 
(1, 'Python全栈+人工智能', 'Python是一种面向对象的解释型计算机程序设计语言，由荷兰人Guido van Rossum于1989年发明，第一个公开发行版发行于1991年。'),
(2, 'JavaEE+分布式服务', 'Java是一门面向对象编程语言，不仅吸收了C++语言的各种优点，还摒弃了C++里难以理解的多继承、指针等概念，因此Java语言具有功能强大和简单易用两个特征。'),
(3, 'HTML5大前端', 'HTML5 将成为 HTML、XHTML 以及 HTML DOM 的新标准。'),
(4, '全栈软件测试', '在规定的条件下对程序进行操作，以发现程序错误，衡量软件质量，并对其是否能满足设计要求进行评估的过程'),
(5, '全链路UI/UE', '全链路要求设计师关注整个业务链中的每一个环节，将设计的价值融入每一个和用户的接触点中，让整个业务的用户体验质量得到几何级数的增长。');

INSERT INTO `tb_teacher` (`no`,`name`,`gender`,`birth`,`intro`,`good_count`,`bad_count`,`photo`,`sno`) 
VALUES 
(1, '骆昊', 1, '1980-11-28', '10年以上软硬件产品设计、研发、架构和管理经验，2003年毕业于四川大学，四川大学Java技术俱乐部创始人，四川省优秀大学毕业生，在四川省网络通信技术重点实验室工作期间，参与了2项国家自然科学基金项目、1项中国科学院中长期研究项目和多项四川省科技攻关项目，在国际会议和国内顶级期刊上发表多篇论文（1篇被SCI收录，3篇被EI收录），大规模网络性能测量系统DMC-TS的设计者和开发者，perf-TTCN语言的发明者。国内最大程序员社区CSDN的博客专家，在Github上参与和维护了多个高质量开源项目，精通C/C++、Java、Python、R、Swift、JavaScript等编程语言，擅长OOAD、系统架构、算法设计、协议分析和网络测量，主持和参与过电子政务系统、KPI考核系统、P2P借贷平台等产品的研发，一直践行“用知识创造快乐”的教学理念，善于总结，乐于分享。', 0, 0, 'images/luohao.png', 1),
(2, '王海飞', 1, '1993-05-24', '5年以上Python开发经验，先后参与了O2O商城、CRM系统、CMS平台、ERP系统等项目的设计与研发，曾在全国最大最专业的汽车领域相关服务网站担任Python高级研发工程师、项目经理等职务，擅长基于Python、Java、PHP等开发语言的企业级应用开发，全程参与了多个企业级应用从需求到上线所涉及的各种工作，精通Django、Flask等框架，熟悉基于微服务的企业级项目开发，拥有丰富的项目实战经验。善于用浅显易懂的方式在课堂上传授知识点，在授课过程中经常穿插企业开发的实际案例并分析其中的重点和难点，通过这种互动性极强的教学模式帮助学员找到解决问题的办法并提升学员的综合素质。', 0, 0, 'images/wangdachui.png', 1),
(3, '余婷', 0, '1992-03-12', '5年以上移动互联网项目开发经验和教学经验，曾担任上市游戏公司高级软件研发工程师和移动端（iOS）技术负责人，参了多个企业级应用和游戏类应用的移动端开发和后台服务器开发，拥有丰富的开发经验和项目管理经验，以个人开发者和协作开发者的身份在苹果的AppStore上发布过多款App。精通Python、C、Objective-C、Swift等开发语言，熟悉iOS原生App开发、RESTful接口设计以及基于Cocos2d-x的游戏开发。授课条理清晰、细致入微，性格活泼开朗、有较强的亲和力，教学过程注重理论和实践的结合，在学员中有良好的口碑。', 0, 0, 'images/yuting.png', 1),
(4, '肖世荣', 1, '1977-07-02', '10年以上互联网和移动互联网产品设计、研发、技术架构和项目管理经验，曾在中国移动、symbio、ajinga.com、万达信息等公司担任架构师、项目经理、技术总监等职务，长期为苹果、保时捷、耐克、沃尔玛等国际客户以及国内的政府机构提供信息化服务，主导的项目曾获得“世界科技先锋”称号，个人作品“许愿吧”曾在腾讯应用市场生活类App排名前3，拥有百万级用户群体，运营的公众号“卵石坊”是国内知名的智能穿戴设备平台。精通Python、C++、Java、Ruby、JavaScript等开发语言，主导和参与了20多个企业级项目（含国家级重大项目和互联网创新项目），涉及的领域包括政务、社交、电信、卫生和金融，有极为丰富的项目实战经验。授课深入浅出、条理清晰，善于调动学员的学习热情并帮助学员理清思路和方法。', 0, 0, 'images/xiaoshirong.png', 1),
(5, '张无忌', 1, '1987-07-07', '出生起便在冰火岛过着原始生活，踏入中土后因中玄冥神掌命危而带病习医，忍受寒毒煎熬七年最后因福缘际会练成“九阳神功”更在之后又练成了“乾坤大挪移”等盖世武功，几乎无敌于天下。 生性随和，宅心仁厚，精通医术和药理。20岁时便凭着盖世神功当上明教教主，率领百万教众及武林群雄反抗蒙古政权元朝的高压统治，最后推翻了元朝。由于擅长乾坤大挪移神功，上课遇到问题就转移话题，属于拉不出屎怪地球没有引力的类型。', 0, 0, 'images/zhangwuji.png', 5),
(6, '韦一笑', 1, '1975-12-15', '外号“青翼蝠王”，为明教四大护教法王之一。  身披青条子白色长袍，轻功十分了得。由于在修炼至阴至寒的“寒冰绵掌”时出了差错，经脉中郁积了至寒阴毒，只要运上内力，寒毒就会发作，如果不吸人血解毒，全身血脉就会凝结成冰，后得张无忌相助，以其高明医术配以“九阳神功”，终将寒毒驱去，摆脱了吸吮人血这一命运。由于轻功绝顶，学生一问问题就跑了。', 0, 0, 'images/weiyixiao.png', 3);
```

当然也可以直接使用Django提供的后台管理应用来添加学科和老师信息，这需要先注册模型类和模型管理类。

```SQL
from django.contrib import admin
from django.contrib.admin import ModelAdmin

from vote.models import Teacher, Subject


class SubjectModelAdmin(ModelAdmin):
    """学科模型管理"""
    list_display = ('no', 'name')
    ordering = ('no', )


class TeacherModelAdmin(ModelAdmin):
    """老师模型管理"""
    list_display = ('no', 'name', 'gender', 'birth', 'good_count', 'bad_count', 'subject')
    ordering = ('no', )
    search_fields = ('name', )


admin.site.register(Subject, SubjectModelAdmin)
admin.site.register(Teacher, TeacherModelAdmin)
```

接下来，我们就可以修改views.py文件，通过编写视图函数先实现“学科介绍”页面。

```Python
def show_subjects(request):
    """查看所有学科"""
    subjects = Subject.objects.all()
    return render(request, 'subject.html', {'subjects': subjects})
```

至此，我们还需要一个模板页，模板的配置以及模板页中模板语言的用法在之前已经进行过简要的介绍，如果不熟悉可以看看下面的代码，相信这并不是一件困难的事情。

```HTML
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>学科信息</title>
    <style>/* 此处略去了层叠样式表的选择器 */</style>
</head>
<body>
    <h1>千锋互联所有学科信息</h1>
    <hr>
    <div id="container">
        { % for subject in subjects % }
        <dl>
            <dt>
                <a href="/teachers?sno={ { subject.no } }">
                    { { subject.name } }
                </a>
            </dt>
            <dd>{ { subject.intro } }</dd>
        </dl>
        { % endfor % }
    </div>
</body>
</html>
```

在上面的模板中，我们为每个学科添加了一个超链接，点击超链接可以查看该学科的讲师信息，为此需要再编写一个视图函数来处理查看指定学科老师信息。

```Python
def show_teachers(request):
    """查看指定学科的老师"""
    try:
        sno = int(request.GET['sno'])
        subject = Subject.objects.get(no=sno)
        teachers = Teacher.objects.filter(subject__no=sno)
        context = {'subject': subject, 'teachers': teachers}
        return render(request, 'teacher.html', context)
    except (KeyError, ValueError, Subject.DoesNotExist):
        return redirect('/')
```

显示老师信息的模板页。

```HTML
<!DOCTYPE html>
{ % load static % }
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>老师信息</title>
    <style>/* 此处略去了层叠样式表的选择器 */</style>
</head>
<body>
    <h1>{ { subject.name } }学科老师信息</h1>
    <hr>
    { % if teachers % }
    <div id="container">
        { % for teacher in teachers % }
        <div class="teacher">
            <div class="photo">
                <img src="{ % static teacher.photo % }" height="140" alt="">
            </div>
            <div class="info">
                <div>
                    <span><strong>姓名：{ { teacher.name } }</strong></span>
                    <span>性别：{ { teacher.gender | yesno:'男,女' } }</span>
                    <span>出生日期：{ { teacher.birth } }</span>
                </div>
                <div class="intro">{ { teacher.intro } }</div>
                <div class="comment">
                    <a href="">好评（{ { teacher.good_count } }）</a>
                    <a href="">差评（{ { teacher.bad_count } }）</a>
                </div>
            </div>
        </div>
        { % endfor % }
    </div>
    { % else % }
    <h2>暂时没有该学科的老师信息</h2>
    { % endif % }
    <div class="back">
        <a href="/">&lt;&lt;&nbsp;返回学科</a>
    </div>
</body>
</html>
```

### 加载静态资源

在上面的模板页面中，我们使用了`<img>`标签来加载老师的照片，其中使用了引用静态资源的模板指令`{ % static % }`，要使用该指令，首先要使用`{ % load static % }`指令来加载静态资源，我们将这段代码放在了页码开始的位置。在上面的项目中，我们将静态资源置于名为static的文件夹中，在该文件夹下又创建了三个文件夹：css、js和images，分别用来保存外部层叠样式表、外部JavaScript文件和图片资源。为了能够找到保存静态资源的文件夹，我们还需要修改Django项目的配置文件settings.py，如下所示：

```Python
# 此处省略上面的代码

STATICFILES_DIRS = [os.path.join(BASE_DIR, 'static'), ]
STATIC_URL = '/static/'

# 此处省略下面的代码
```

接下来修改urls.py文件，配置用户请求的URL和视图函数的对应关系。

```Python
from django.contrib import admin
from django.urls import path

from vote import views

urlpatterns = [
    path('', views.show_subjects),
    path('teachers/', views.show_teachers),
    path('admin/', admin.site.urls),
]
```

启动服务器运行项目，进入首页查看学科信息。

![](./res/show-subjects.png)

点击学科查看老师信息。

![](./res/show-teachers.png)

### Ajax请求

接下来就可以实现“好评”和“差评”的功能了，很明显如果能够在不刷新页面的情况下实现这两个功能会带来更好的用户体验，因此我们考虑使用[Ajax](https://zh.wikipedia.org/wiki/AJAX)技术来实现“好评”和“差评”，Ajax技术我们在之前的章节中已经介绍过了，此处不再赘述。

首先修改项目的urls.py文件，为“好评”和“差评”功能映射对应的URL。

```Python
from django.contrib import admin
from django.urls import path

from vote import views

urlpatterns = [
    path('', views.show_subjects),
    path('teachers/', views.show_teachers),
    path('praise/', views.prise_or_criticize),
    path('criticize/', views.prise_or_criticize),
    path('admin/', admin.site.urls),
]
```

设计视图函数`praise_or_criticize`来支持“好评”和“差评”功能，该视图函数通过Django封装的JsonResponse类将字典序列化成JSON字符串作为返回给浏览器的响应内容。

```Python
def praise_or_criticize(request):
    """好评"""
    try:
        tno = int(request.GET['tno'])
        teacher = Teacher.objects.get(no=tno)
        if request.path.startswith('/prise'):
            teacher.good_count += 1
        else:
            teacher.bad_count += 1
        teacher.save()
        data = {'code': 200, 'hint': '操作成功'}
    except (KeyError, ValueError, Teacher.DoseNotExist):
        data = {'code': 404, 'hint': '操作失败'}
    return JsonResponse(data)
```

修改显示老师信息的模板页，引入jQuery库来实现事件处理、Ajax请求和DOM操作。

```HTML
<script src="{ % static 'js/jquery.min.js' % }"></script>
<script>
    $(() => {
        $('.comment>a').on('click', (evt) => {
            evt.preventDefault();
            let a = $(evt.target)
            let span = a.next()
            $.getJSON(a.attr('href'), (json) => {
                if (json.code == 200) {
                    span.text(parseInt(span.text()) + 1)
                } else {
                    alert(json.hint)
                }
            })
        })
    })
</script>
```

### 小结

到此为止，这个投票项目的核心功能已然完成，在下面的章节中我们会要求用户必须登录才能投票，没有账号的用户可以通过注册功能注册一个账号。
