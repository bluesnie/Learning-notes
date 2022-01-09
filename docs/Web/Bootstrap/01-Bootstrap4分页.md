###### datetime:2019/10/14 14:23
###### author:nzb

[文章参考](https://www.jianshu.com/p/749a77a21b3e)

```html
    <link href="/static/css/bootstrap.min.css" rel="stylesheet">
    <script src="/static/js/jquery-2.1.1.min.js"></script>
    <script src="/static/js/bootstrap.min.js"></script>
    <script src="/static/js/bootstrap-paginator.js"></script>
    
```

```html
    <div id="example" style="text-align: center"> <ul id="pageLimit"></ul> </div>
    
    <script>
    $('#pageLimit').bootstrapPaginator({
     currentPage: 1,//当前页。
     totalPages: 20,//总页数。
     size:"normal",//应该是页眉的大小。
     bootstrapMajorVersion: 3,//bootstrap的版本要求。
     alignment:"right",
     numberOfPages:5,//显示的页数
     itemTexts: function (type, page, current) {//如下的代码是将页眉显示的中文显示我们自定义的中文。
             switch (type) {
             case "first": return "首页";
             case "prev": return "上一页";
             case "next": return "下一页";
             case "last": return "末页";
             case "page": return page;
            }
        },
      onPageClicked: function (event, originalEvent, type, page) {//给每个页眉绑定一个事件，其实就是ajax请求，其中page变量为当前点击的页上的数字。
                $.ajax({
                    url:'',
                     type:'',
                     data:{},
                     dataType:'JSON',
                     success:function (callback) {
                            
                     }
                 })
            }
     });
    </script>
```

![](../res/bootstrap-paginator.png)