###### datetime:2019/6/11 16:27
###### author:nzb

## Django的路由系统

[Django 1.11版本 URLConf官方文档](https://docs.djangoproject.com/en/1.11/topics/http/urls/)

URL配置(URLconf)就像Django 所支撑网站的目录。它的本质是URL与要为该URL调用的视图函数之间的映射表。

你就是以这种方式告诉Django，对于这个URL调用这段代码，对于那个URL调用那段代码。

### URLconfs配置

#### 基本格式

```python
    from django.conf.urls import url
    
    urlpatterns = [
         url(正则表达式, views视图函数，参数，别名),
    ]
```
注意：

Django 2.0版本中的路由系统已经替换成下面的写法[(官方文档)](https://docs.djangoproject.com/en/2.0/topics/http/urls/)：
```python
    from django.urls import path
    
    urlpatterns = [
        path('articles/2003/', views.special_case_2003),
        path('articles/<int:year>/', views.year_archive),
        path('articles/<int:year>/<int:month>/', views.month_archive),
        path('articles/<int:year>/<int:month>/<slug:slug>/', views.article_detail),
    ]
```

#### 参数说明

- 正则表达式：一个正则表达式字符串

- views视图函数：一个可调用对象，通常为一个视图函数或一个指定视图函数路径的字符串

- 参数：可选的要传递给视图函数的默认参数（字典形式）

- 别名：一个可选的name参数

### 正则表达式详解

#### 基本配置

```python
    from django.conf.urls import url
    
    from . import views
    
    urlpatterns = [
        url(r'^articles/2003/$', views.special_case_2003),
        url(r'^articles/([0-9]{4})/$', views.year_archive),
        url(r'^articles/([0-9]{4})/([0-9]{2})/$', views.month_archive),
        url(r'^articles/([0-9]{4})/([0-9]{2})/([0-9]+)/$', views.article_detail),
    ]
```

#### 注意事项

1. urlpatterns中的元素按照书写顺序从上往下逐一匹配正则表达式，一旦匹配成功则不再继续。

2. 若要从URL中捕获一个值，只需要在它周围放置一对圆括号（分组匹配）。

3. 不需要添加一个前导的反斜杠，因为每个URL 都有。例如，应该是^articles 而不是 ^/articles。

4. 每个正则表达式前面的'r' 是可选的但是建议加上。

#### 补充说明

是否开启URL访问地址后面不为/跳转至带有/的路径的配置项

    APPEND_SLASH=True

Django settings.py配置文件中默认没有 APPEND_SLASH 这个参数，但 Django 默认这个参数为 APPEND_SLASH = True。 其作用就是自动在网址结尾加'/'。

其效果就是：

我们定义了urls.py：
```python
    from django.conf.urls import url
    from app01 import views
    
    urlpatterns = [
            url(r'^blog/$', views.blog),
    ]
```
访问 http://www.example.com/blog 时，默认将网址自动转换为 http://www.example/com/blog/ 。

如果在settings.py中设置了 APPEND_SLASH=False，此时我们再请求 http://www.example.com/blog 时就会提示找不到页面。

### 分组命名匹配

上面的示例使用简单的正则表达式分组匹配（通过圆括号）来捕获URL中的值并以位置参数形式传递给视图。

在更高级的用法中，可以使用分组命名匹配的正则表达式组来捕获URL中的值并以关键字参数形式传递给视图。

在Python的正则表达式中，分组命名正则表达式组的语法是(?P<name>pattern)，其中name是组的名称，pattern是要匹配的模式。

下面是以上URLconf 使用命名组的重写：

```python
    from django.conf.urls import url
    
    from . import views
    
    urlpatterns = [
        url(r'^articles/2003/$', views.special_case_2003),
        url(r'^articles/(?P<year>[0-9]{4})/$', views.year_archive),
        url(r'^articles/(?P<year>[0-9]{4})/(?P<month>[0-9]{2})/$', views.month_archive),
        url(r'^articles/(?P<year>[0-9]{4})/(?P<month>[0-9]{2})/(?P<day>[0-9]{2})/$', views.article_detail),
    ]
```

这个实现与前面的示例完全相同，只有一个细微的差别：捕获的值作为关键字参数而不是位置参数传递给视图函数。

例如，针对url /articles/2017/12/相当于按以下方式调用视图函数：

```python
    views.month_archive(request, year="2017", month="12")
```
在实际应用中，使用分组命名匹配的方式可以让你的URLconf 更加明晰且不容易产生参数顺序问题的错误，但是有些开发人员则认为分组命名组语法太丑陋、繁琐。

至于究竟应该使用哪一种，你可以根据自己的喜好来决定。

#### URLconf匹配的位置

URLconf 在请求的URL 上查找，将它当做一个普通的Python 字符串。不包括GET和POST参数以及域名。

例如，http://www.example.com/myapp/ 请求中，URLconf 将查找myapp/。

在http://www.example.com/myapp/?page=3 请求中，URLconf 仍将查找myapp/。

URLconf 不检查请求的方法。换句话讲，所有的请求方法 —— 同一个URL的POST、GET、HEAD等等 —— 都将路由到相同的函数。

#### 捕获的参数永远都是字符串

每个在URLconf中捕获的参数都作为一个普通的Python字符串传递给视图，无论正则表达式使用的是什么匹配方式。例如，下面这行URLconf 中：
```python
    url(r'^articles/(?P<year>[0-9]{4})/$', views.year_archive),
```
传递到视图函数views.year_archive() 中的year 参数永远是一个字符串类型。

#### 视图函数中指定默认值

urls.py中:
```python
    from django.conf.urls import url
    
    from . import views
    
    urlpatterns = [
        url(r'^blog/$', views.page),
        url(r'^blog/page(?P<num>[0-9]+)/$', views.page),
    ]
    
    # views.py中，可以为num指定默认值
    def page(request, num="1"):
        pass
```
在上面的例子中，两个URL模式指向相同的view - views.page - 但是第一个模式并没有从URL中捕获任何东西。

如果第一个模式匹配上了，page()函数将使用其默认参数num=“1”,如果第二个模式匹配，page()将使用正则表达式捕获到的num值。

#### include其他的URLconfs

```python
    #At any point, your urlpatterns can “include” other URLconf modules. This
    #essentially “roots” a set of URLs below other ones.
    
    #For example, here’s an excerpt of the URLconf for the Django website itself.
    #It includes a number of other URLconfs:
    
    
    from django.conf.urls import include, url
    
    urlpatterns = [
       url(r'^admin/', admin.site.urls),
       url(r'^blog/', include('blog.urls')),  # 可以包含其他的URLconfs文件
    ]
```

### 传递额外的参数给视图函数（了解）

URLconfs 具有一个钩子，让你传递一个Python 字典作为额外的参数传递给视图函数。

django.conf.urls.url() 函数可以接收一个可选的第三个参数，它是一个字典，表示想要传递给视图函数的额外关键字参数。

例如：
```python
    from django.conf.urls import url
    from . import views
    
    urlpatterns = [
        url(r'^blog/(?P<year>[0-9]{4})/$', views.year_archive, {'foo': 'bar'}),
    ]
```
在这个例子中，对于/blog/2005/请求，Django 将调用views.year_archive(request, year='2005', foo='bar')。
这个技术在Syndication 框架中使用，来传递元数据和选项给视图。

### 命名URL和URL反向解析

在使用Django 项目时，一个常见的需求是获得URL的最终形式，以用于嵌入到生成的内容中（视图中和显示给用户的URL等）或者用于处理服务器端的导航（重定向等）。

人们强烈希望不要硬编码这些URL（费力、不可扩展且容易产生错误）或者设计一种与URLconf 毫不相关的专门的URL 生成机制，因为这样容易导致一定程度上产生过期的URL。

换句话讲，需要的是一个DRY 机制。除了其它有点，它还允许设计的URL 可以自动更新而不用遍历项目的源代码来搜索并替换过期的URL。

获取一个URL 最开始想到的信息是处理它视图的标识（例如名字），查找正确的URL 的其它必要的信息有视图参数的类型（位置参数、关键字参数）和值。

Django 提供一个办法是让URL 映射是URL 设计唯一的地方。你填充你的URLconf，然后可以双向使用它：

- 根据用户/浏览器发起的URL 请求，它调用正确的Django 视图，并从URL 中提取它的参数需要的值。

- 根据Django 视图的标识和将要传递给它的参数的值，获取与之关联的URL。

第一种方式是我们在前面的章节中一直讨论的用法。第二种方式叫做反向解析URL、反向URL 匹配、反向URL 查询或者简单的URL 反查。

在需要URL 的地方，对于不同层级，Django 提供不同的工具用于URL 反查：

- 在模板中：使用url模板标签。
- 在Python 代码中：使用django.core.urlresolvers.reverse() 函数。
- 在更高层的与处理Django 模型实例相关的代码中：使用get_absolute_url() 方法。

上面说了一大堆，你可能并没有看懂。（那是官方文档的生硬翻译）。

咱们简单来说就是可以给我们的URL匹配规则起个名字，一个URL匹配模式起一个名字。

这样我们以后就不需要写死URL代码了，只需要通过名字来调用当前的URL。

举个简单的例子：
```python
    url(r'^home', views.home, name='home'),  # 给我的url匹配模式起名为 home
    url(r'^index/(\d*)', views.index, name='index'),  # 给我的url匹配模式起名为index
```

这样：

在模板里面可以这样引用：
```python
    {% url 'home' %}
```

在views函数中可以这样引用：
```python
    from django.urls import reverse
    
    reverse("index", args=("2018", ))
```

例子：
考虑下面的URLconf：
```python
    from django.conf.urls import url
    
    from . import views
    
    urlpatterns = [
        # ...
        url(r'^articles/([0-9]{4})/$', views.year_archive, name='news-year-archive'),
        # ...
    ]
```

根据这里的设计，某一年nnnn对应的归档的URL是/articles/nnnn/。

你可以在模板的代码中使用下面的方法获得它们：
```html
    <a href="{% url 'news-year-archive' 2012 %}">2012 Archive</a>
    
    <ul>
    {% for yearvar in year_list %}
    <li><a href="{% url 'news-year-archive' yearvar %}">{{ yearvar }} Archive</a></li>
    {% endfor %}
    </ul>
```

在Python 代码中，这样使用：
```python
    from django.urls import reverse
    from django.shortcuts import redirect
    
    def redirect_to_year(request):
        # ...
        year = 2006
        # ...
        return redirect(reverse('news-year-archive', args=(year,)))
```
如果出于某种原因决定按年归档文章发布的URL应该调整一下，那么你将只需要修改URLconf 中的内容。

在某些场景中，一个视图是通用的，所以在URL 和视图之间存在多对一的关系。对于这些情况，当反查URL 时，只有视图的名字还不够。

**注意：**

为了完成上面例子中的URL 反查，你将需要使用命名的URL 模式。URL 的名称使用的字符串可以包含任何你喜欢的字符。不只限制在合法的Python 名称。

当命名你的URL 模式时，请确保使用的名称不会与其它应用中名称冲突。如果你的URL 模式叫做comment，而另外一个应用中也有一个同样的名称，当你在模板中使用这个名称的时候不能保证将插入哪个URL。

在URL 名称中加上一个前缀，比如应用的名称，将减少冲突的可能。我们建议使用myapp-comment 而不是comment。

### 命名空间模式

即使不同的APP使用相同的URL名称，URL的命名空间模式也可以让你唯一反转命名的URL。

举个例子：

project中的urls.py
```python
    from django.conf.urls import url, include
     
    urlpatterns = [
        url(r'^app01/', include('app01.urls', namespace='app01')),
        url(r'^app02/', include('app02.urls', namespace='app02')),
    ]
```

app01中的urls.py
```python
    from django.conf.urls import url
    from app01 import views
     
    app_name = 'app01'
    urlpatterns = [
        url(r'^(?P<pk>\d+)/$', views.detail, name='detail')
    ]
```

app02中的urls.py
```python
    from django.conf.urls import url
    from app02 import views
     
    app_name = 'app02'
    urlpatterns = [
        url(r'^(?P<pk>\d+)/$', views.detail, name='detail')
    ]
```

现在，我的两个app中 url名称重复了，我反转URL的时候就可以通过命名空间的名称得到我当前的URL。

语法：

'命名空间名称:URL名称'

模板中使用：

    {% url 'app01:detail' pk=12 pp=99 %}

views中的函数中使用

    v = reverse('app01:detail', kwargs={'pk':11})

这样即使app中URL的命名相同，我也可以反转得到正确的URL了。




