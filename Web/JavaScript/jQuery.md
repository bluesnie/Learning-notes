###### datetime:2019/7/10 13:43
###### author:nzb


## jQuery简介

jQuery 库可以通过一行简单的标记被添加到网页中。

### 什么是 jQuery ？

- jQuery是一个JavaScript函数库。
- jQuery是一个轻量级的"写的少，做的多"的JavaScript库。
- jQuery库包含以下功能：
- HTML 元素选取
- HTML 元素操作
- CSS 操作
- HTML 事件函数
- JavaScript 特效和动画
- HTML DOM 遍历和修改
- AJAX
- Utilities
- **提示**： 除此之外，Jquery还提供了大量的插件。

## 语法

jQuery 语法是通过选取 HTML 元素，并对选取的元素执行某些操作。

- 基础语法： $(selector).action()
    - 美元符号定义 jQuery
    - 选择符（selector）"查询"和"查找" HTML 元素
    - jQuery 的 action() 执行对元素的操作

- 实例:
    - $(this).hide() - 隐藏当前元素
    - $("p").hide() - 隐藏所有 `<p>` 元素
    - $("p.test").hide() - 隐藏所有 class="test" 的 `<p>` 元素
    - $("#test").hide() - 隐藏所有 id="test" 的元素

- 文档就绪事件

您也许已经注意到在我们的实例中的所有 jQuery 函数位于一个 document ready 函数中：
```javascript
    $(document).ready(function(){
     
       // 开始写 jQuery 代码...
     
    });
```
这是为了防止文档在完全加载（就绪）之前运行 jQuery 代码，即在 DOM 加载完成后才可以对 DOM 进行操作。

如果在文档没有完全加载之前就运行函数，操作可能失败。下面是两个具体的例子：
- 试图隐藏一个不存在的元素
- 获得未完全加载的图像的大小

**提示：**简洁写法（与以上写法效果相同）:
```javascript
    $(function(){
     
       // 开始写 jQuery 代码...
     
    });
```
以上两种方式你可以选择你喜欢的方式实现文档就绪后执行 jQuery 方法。

JavaScript 入口函数:
```javascript
    window.onload = function () {
        // 执行代码
    }
```
jQuery 入口函数与 JavaScript 入口函数的区别：
 - jQuery 的入口函数是在 html 所有标签(DOM)都加载之后，就会去执行。
 - JavaScript 的 window.onload 事件是等到所有内容，包括外部图片之类的文件加载完后，才会执行。

## 选择器

jQuery 选择器允许您对 HTML 元素组或单个元素进行操作。

jQuery 中所有选择器都以美元符号开头：$()。

### 元素选择器

用户点击按钮后，所有 <p> 元素都隐藏：
```javascript
    $(document).ready(function(){
      $("button").click(function(){
        $("p").hide();
      });
    });
```

### #id 选择器

当用户点击按钮后，有 id="test" 属性的元素将被隐藏：
```javascript
    $(document).ready(function(){
      $("button").click(function(){
        $("#test").hide();
      });
    });
```

### .class 选择器

用户点击按钮后所有带有 class="test" 属性的元素都隐藏：
```javascript
    $(document).ready(function(){
      $("button").click(function(){
        $(".test").hide();
      });
    });
```

### 更多实例
| 语法 | 描述 |
|----------|------------|
| $("*") | 选取所有元素	 |
| $(this) | 选取当前 HTML 元素	 |
| $("p.intro") | 选取 class 为 intro 的 `<p>` 元素	 |
| $("p:first") | 选取第一个 `<p>` 元素	 |
| $("ul li:first") | 选取第一个 `<ul>` 元素的第一个 `<li>` 元素	 |
| $("ul li:first-child") | 选取每个 `<ul>` 元素的第一个 `<li>` 元素	 |
| $("[href]") | 选取带有 href 属性的元素	 |
| $("a[target='_blank']") | 选取所有 target 属性值等于 "_blank" 的 `<a>` 元素	 |
| $("a[target!='_blank']") | 选取所有 target 属性值不等于 "_blank" 的 `<a>` 元素	 |
| $(":button") | 选取所有 type="button" 的 <input> 元素 和 <button> 元素	 |
| $("tr:even") | 选取偶数位置的 `<tr>` 元素	 |
| $("tr:odd") | 选取奇数位置的 `<tr>` 元素	 |
| $("#id", ".class") | 复合选择器 |
| $(div p span)      | 层级选择器 //div下的p元素中的span元素 |
| $(div>p)           | 父子选择器 //div下的所有p元素 |
| $(div+p)           | 相邻元素选择器 //div后面的p元素(仅一个p) |
| $(div~p)           | 兄弟选择器  //div后面的所有p元素(同级别) |
| $(.p:last)         | 类选择器 加 过滤选择器  第一个和最后一个（first 或者 last） |
| $("#mytable td:odd")     | 层级选择 加 过滤选择器 奇偶（odd 或者 even） |
| $("div p:eq(2)")   | 索引选择器 div下的第三个p元素（索引是从0开始） |
| $("a[href='www.baidu.com']") | 属性选择器 |
| $("p:contains(test)")       | // 内容过滤选择器，包含text内容的p元素 |
| $(":emtyp")       | //内容过滤选择器，所有空标签（不包含子标签和内容的标签）parent 相反 |
| $(":hidden")      | //所有隐藏元素 visible  |
| $("input:enabled") | //选取所有启用的表单元素 |
| $(":disabled")    | //所有不可用的元素 |
| $("input:checked") | //获取所有选中的复选框单选按钮等 |
| $("select option:selected") | //获取选中的选项元素 |

关于 : 和 [] 这两个符号的理解

“**：**”：可以理解为种类的意思，如：p:first，p 的种类为第一个。

“**[]**” ：很自然的可以理解为属性的意思，如：[href] 选取带有 href 属性的元素。

$(":button") 为 jQuery 中表单选择器（貌似与过滤选择器同级），旨在选择所有的按钮，所以会找到 <input>、<button> 元素；而 $("button") 则为基本选择器，旨在选择为 <button> 的标签。

**:** 即为 jQuery 的过滤选择器，语法类似于 css 中的伪类选择器；其过滤选择器大概可以分为基本过滤（p:first 之类）、内容过滤（:empty）、子元素过滤(:first-child)和属性过滤 [href] 选择器。

## 事件

页面对不同访问者的响应叫做事件。

事件处理程序指的是当 HTML 中发生某些事件时所调用的方法。

实例：

- 在元素上移动鼠标。
- 选取单选按钮
- 点击元素

在事件中经常使用术语"触发"（或"激发"）例如： "当您按下按键时触发 keypress 事件"。

### 常用的 jQuery 事件方法

- $(document).ready()

    $(document).ready() 方法允许我们在文档完全加载完后执行函数。该事件方法在 jQuery 语法 章节中已经提到过。

- click()

    click() 方法是当按钮点击事件被触发时会调用一个函数。

- dblclick()

    当双击元素时，会发生 dblclick 事件。

- mouseenter()

    当鼠标指针穿过元素时，会发生 mouseenter 事件。
    
- mouseleave()

    当鼠标指针离开元素时，会发生 mouseleave 事件。

- mousedown()

    当鼠标指针移动到元素上方，并按下鼠标按键时，会发生 mousedown 事件。

- mouseup()

    当在元素上松开鼠标按钮时，会发生 mouseup 事件。

- hover()

    hover()方法用于模拟光标悬停事件。

    当鼠标移动到元素上时，会触发指定的第一个函数(mouseenter);当鼠标移出这个元素时，会触发指定的第二个函数(mouseleave)。

- focus()

    当元素获得焦点时，发生 focus 事件。

    当通过鼠标点击选中元素或通过 tab 键定位到元素时，该元素就会获得焦点。

- blur()

    当元素失去焦点时，发生 blur 事件。

- on() 和 off()
    
    绑定事件和解除绑定事件

### 笔记

一.keypress,keydown,keyup的区别:

     1.keydown：在键盘上按下某键时发生,一直按着则会不断触发（opera浏览器除外）, 它返回的是键盘代码;
     2.keypress：在键盘上按下一个按键，并产生一个字符时发生, 返回ASCII码。注意: shift、alt、ctrl等键按下并不会产生字符，所以监听无效 ,换句话说, 只有按下能在屏幕上输出字符的按键时keypress事件才会触发。若一直按着某按键则会不断触发。
     3.keyup：用户松开某一个按键时触发, 与keydown相对, 返回键盘代码.

二.两种常用用法举例

案例1:获取按键代码或字符的ASCII码
```javascript
    $(window).keydown( function(event){
       // 通过event.which可以拿到按键代码.  如果是keypress事件中,则拿到ASCII码.
    } );
```
案例2:传递数据给事件处理函数

语法:

jQueryObject.keypress( [[ data ,]  handler ] );
- data: 通过event.data传递给事件处理函数的任意数据;
- handler: 指定的事件处理函数;

举例:
```javascript
    // 只允许按下的字母键生效, 65~90是所有大写字母的键盘代码范围.
    var validKeys = { start: 65, end: 90  };
    $("#keys").keypress( validKeys, function(event){
        var keys = event.data;  //拿到validKeys对象.
        return event.which >= keys.start && event.which <= keys.end;
    } );
```

三.关于获取触发事件的说明：

1.获取事件对象
```javascript
    $(document).ready(function(){
        $(window).keypress(function(event){    
            //获取事件对象，里面包含各种有用的信息。
            console.log(event);
            //console.log(event.which);
        });
    });
```
2.keypress事件获取键入字符
```javascript
    $(window).keypress(function(event){
        //event.which是获取ASCII码，前面的函数是将ASCII码转换成字符，空格键和Enter键输出均为空白。
        console.log(String.fromCharCode(event.which));
        //从event对象中key属性获取字符，但是Enter键的key值为"Enter"，空白键还是空白" "。
        console.log(event.key);
    });
```

## JQuery HTML

### 获取内容和属性

- 获得内容： text()、html() 以及 val()
- 三个简单实用的用于 DOM 操作的 jQuery 方法：
    - text() - 设置或返回所选元素的文本内容
    - html() - 设置或返回所选元素的内容（包括 HTML 标记）
    - val() - 设置或返回表单字段的值
    ```javascript
        $("#btn1").click(function(){
          alert("Text: " + $("#test").text());
        });
        $("#btn2").click(function(){
          alert("HTML: " + $("#test").html());
        });
        $("#btn1").click(function(){
          alert("值为: " + $("#test").val());
        });
    ```

- 获取属性 - attr()：用于获取属性值。
    ```javascript
        $("button").click(function(){
          alert($("#runoob").attr("href"));
        });
    ```

### 设置内容和属性

- 设置内容： text()、html() 以及 val()
- 我们将使用前一章中的三个相同的方法来设置内容：
    - text() - 设置或返回所选元素的文本内容
    - html() - 设置或返回所选元素的内容（包括 HTML 标记）
    - val() - 设置或返回表单字段的值
    ```javascript
        $("#btn1").click(function(){
            $("#test1").text("Hello world!");
        });
        $("#btn2").click(function(){
            $("#test2").html("<b>Hello world!</b>");
        });
        $("#btn3").click(function(){
            $("#test3").val("RUNOOB");
        });
    ```

- text()、html() 以及 val() 的回调函数

上面的三个 jQuery 方法：text()、html() 以及 val()，同样拥有回调函数。回调函数有两个参数：被选元素列表中当前元素的下标，以及原始（旧的）值。然后以函数新值返回您希望使用的字符串。

下面的例子演示带有回调函数的 text() 和 html()：
```javascript
    $("#btn1").click(function(){
        $("#test1").text(function(i,origText){
            return "旧文本: " + origText + " 新文本: Hello world! (index: " + i + ")"; 
        });
    });
     
    $("#btn2").click(function(){
        $("#test2").html(function(i,origText){
            return "旧 html: " + origText + " 新 html: Hello <b>world!</b> (index: " + i + ")"; 
        });
    });
```

- 设置属性 - attr()：用于设置/改变属性值。
    ```javascript
        $("button").click(function(){
          $("#runoob").attr("href","http://www.runoob.com/jquery");
        });
    ```
attr() 方法也允许您同时设置多个属性。

下面的例子演示如何同时设置 href 和 title 属性：
```javascript
    $("button").click(function(){
        $("#runoob").attr({
            "href" : "http://www.runoob.com/jquery",
            "title" : "jQuery 教程"
        });
    });
```

- attr() 的回调函数

jQuery 方法 attr()，也提供回调函数。回调函数有两个参数：被选元素列表中当前元素的下标，以及原始（旧的）值。然后以函数新值返回您希望使用的字符串。

下面的例子演示带有回调函数的 attr() 方法：
```javascript
    $("button").click(function(){
      $("#runoob").attr("href", function(i,origValue){
        return origValue + "/jquery"; 
      });
    });
```

### 添加元素

- 添加新内容的四个 jQuery 方法：
    - append() - 在被选元素的结尾插入内容
    - prepend() - 在被选元素的开头插入内容
    - after() - 在被选元素之后插入内容
    - before() - 在被选元素之前插入内容

- append() 方法

    jQuery append() 方法在被选元素的结尾插入内容（仍然该元素的内部）。
    ```javascript
        $("p").append("追加文本");
    ```

- prepend() 方法

    jQuery prepend() 方法在被选元素的开头插入内容。
    ```javascript
        $("p").prepend("在开头追加文本");
    ```

- 通过 append() 和 prepend() 方法添加若干新元素

    在上面的例子中，我们只在被选元素的开头/结尾插入文本/HTML。
    
    不过，append() 和 prepend() 方法能够通过参数接收无限数量的新元素。可以通过 jQuery 来生成文本/HTML（就像上面的例子那样），或者通过 JavaScript 代码和 DOM 元素。
    
    在下面的例子中，我们创建若干个新元素。这些元素可以通过 text/HTML、jQuery 或者 JavaScript/DOM 来创建。然后我们通过 append() 方法把这些新元素追加到文本中（对 prepend() 同样有效）：
    ```javascript
        function appendText()
        {
            var txt1="<p>文本。</p>";              // 使用 HTML 标签创建文本
            var txt2=$("<p></p>").text("文本。");  // 使用 jQuery 创建文本
            var txt3=document.createElement("p");
            txt3.innerHTML="文本。";               // 使用 DOM 创建文本 text with DOM
            $("body").append(txt1,txt2,txt3);        // 追加新元素
        }
    ```

- jQuery after() 和 before() 方法
    ```javascript
        $("img").after("在后面添加文本");
         
        $("img").before("在前面添加文本");
    ```

- 通过 after() 和 before() 方法添加若干新元素

    after() 和 before() 方法能够通过参数接收无限数量的新元素。可以通过 text/HTML、jQuery 或者 JavaScript/DOM 来创建新元素。
    
    在下面的例子中，我们创建若干新元素。这些元素可以通过 text/HTML、jQuery 或者 JavaScript/DOM 来创建。然后我们通过 after() 方法把这些新元素插到文本中（对 before() 同样有效）：
    ```javascript
        function afterText()
        {
            var txt1="<b>I </b>";                    // 使用 HTML 创建元素
            var txt2=$("<i></i>").text("love ");     // 使用 jQuery 创建元素
            var txt3=document.createElement("big");  // 使用 DOM 创建元素
            txt3.innerHTML="jQuery!";
            $("img").after(txt1,txt2,txt3);          // 在图片后添加文本
        }
    ```

- append/prepend和after/before有什么区别？
    - append
        ```javascript
            <p>
              <span class="s1">s1</span>
            </p>
            <script>
            $("p").append('<span class="s2">s2</span>');
            </script>
        ```
        结果是这样的:
        ```html
            <p>
              <span class="s1">s1</span>
              <span class="s2">s2</span>
            </p>
        ```
        
    - after
        ```javascript
            <p>
              <span class="s1">s1</span>
            </p>
            <script>
            $("p").after('<span class="s2">s2</span>');
            </script>
        ```
        结果是这样的:
        ```html
            <p>
              <span class="s1">s1</span>
            </p>
            <span class="s2">s2</span>
        ```
        
    - **总结：**
        - append/prepend 是在选择元素内部嵌入。
        - after/before 是在元素外面追加。

### 删除元素

- 使用以下两个 jQuery 方法：
    - remove() - 删除被选元素（及其子元素）
    - empty() - 从被选元素中删除子元素

- jQuery remove() 方法：删除被选元素及其子元素。
    ```javascript
        $("#div1").remove();
    ```

- jQuery empty() 方法：删除被选元素的子元素。
    ```javascript
        $("#div1").empty();
    ```

- 过滤被删除的元素

    jQuery remove() 方法也可接受一个参数，允许您对被删元素进行过滤。

    该参数可以是任何 jQuery 选择器的语法。

    下面的例子删除 class="italic" 的所有 `<p>` 元素：
    ```javascript
        $("p").remove(".italic");
    ```

## jQuery 效果

### 隐藏和显示

- hide() 和 show()

    使用 hide() 和 show() 方法来隐藏和显示 HTML 元素：
    ```javascript
        $("#hide").click(function(){
          $("p").hide();
        });
         
        $("#show").click(function(){
          $("p").show();
        });
    ```
    - 语法:
        ```javascript
            $(selector).hide(speed,callback);
            
            $(selector).show(speed,callback);
        ```
        可选的 speed 参数规定隐藏/显示的速度，可以取以下值："slow"、"fast" 或毫秒。
    
        可选的 callback 参数是隐藏或显示完成后所执行的函数名称。
        
        对于可选的 callback 参数，有以下两点说明：
    
        1.$(selector)选中的元素的个数为n个，则callback函数会执行n次；当 callback 函数加上括号时，函数立即执行，只会调用一次， 如果不加括号，元素显示或隐藏后调用函数，才会调用多次。
        
        2.callback函数名后加括号，会立刻执行函数体，而不是等到显示/隐藏完成后才执行；
        
        3.callback既可以是函数名，也可以是匿名函数；
            
        下面的例子演示了带有 speed 参数的 hide() 方法：
        ```javascript
            $("button").click(function(){
              $("p").hide(1000);
            });
        ```
        下面的例子演示了带有 speed 参数的 hide() 方法，并使用回调函数：
        ```javascript
            $(document).ready(function(){
              $(".hidebtn").click(function(){
                $("div").hide(1000,"linear",function(){
                  alert("Hide() 方法已完成!");
                });
              });
            });
        ```
        第二个参数是一个字符串，表示过渡使用哪种缓动函数。（译者注：jQuery自身提供"linear" 和 "swing"，其他可以使用相关的插件）。

- toggle()

    使用 toggle() 方法来切换 hide() 和 show() 方法。显示被隐藏的元素，并隐藏已显示的元素：
    ```javascript
        $("button").click(function(){
          $("p").toggle();
        });
    ```
    
    - 语法:
        ```javascript
            $(selector).toggle(speed,callback);
        ```
        可选的 speed 参数规定隐藏/显示的速度，可以取以下值："slow"、"fast" 或毫秒。
    
        可选的 callback 参数是隐藏或显示完成后所执行的函数名称。

### 淡入淡出

- jQuery 拥有下面四种 fade 方法：
    - fadeIn()
    - fadeOut()
    - fadeToggle()
    - fadeTo()

- fadeIn() 方法：用于淡入已隐藏的元素。

    - 语法:
        ```javascript
            $(selector).fadeIn(speed,callback);
        ```
        可选的 speed 参数规定效果的时长。它可以取以下值："slow"、"fast" 或毫秒。.
        
        可选的 callback 参数是 fading 完成后所执行的函数名称。
    
        下面的例子演示了带有不同参数的 fadeIn() 方法：
        ```javascript
            $("button").click(function(){
              $("#div1").fadeIn();
              $("#div2").fadeIn("slow");
              $("#div3").fadeIn(3000);
            });
        ```

- fadeOut() 方法：用于淡出可见元素。

    - 语法:
        ```javascript
            $(selector).fadeOut(speed,callback);
        ```
        可选的 speed 参数规定效果的时长。它可以取以下值："slow"、"fast" 或毫秒。
        
        可选的 callback 参数是 fading 完成后所执行的函数名称。
        
        下面的例子演示了带有不同参数的 fadeOut() 方法：
        ```javascript
            $("button").click(function(){
              $("#div1").fadeOut();
              $("#div2").fadeOut("slow");
              $("#div3").fadeOut(3000);
            });
        ```

- fadeToggle() 方法：

    jQuery fadeToggle() 方法可以在 fadeIn() 与 fadeOut() 方法之间进行切换。
    
    如果元素已淡出，则 fadeToggle() 会向元素添加淡入效果。
    
    如果元素已淡入，则 fadeToggle() 会向元素添加淡出效果。

    - 语法:
    ```javascript
        $(selector).fadeToggle(speed,callback);
    ```
    可选的 speed 参数规定效果的时长。它可以取以下值："slow"、"fast" 或毫秒。
    
    可选的 callback 参数是 fading 完成后所执行的函数名称。
    
    下面的例子演示了带有不同参数的 fadeToggle() 方法：
    ```javascript
        $("button").click(function(){
          $("#div1").fadeToggle();
          $("#div2").fadeToggle("slow");
          $("#div3").fadeToggle(3000);
        });
    ```

- fadeTo() 方法

    jQuery fadeTo() 方法允许渐变为给定的不透明度（值介于 0 与 1 之间）。

    - 语法:
    
    必需的 speed 参数规定效果的时长。它可以取以下值："slow"、"fast" 或毫秒。
    
    fadeTo() 方法中必需的 opacity 参数将淡入淡出效果设置为给定的不透明度（值介于 0 与 1 之间）。
    
    可选的 callback 参数是该函数完成后所执行的函数名称。
    
    下面的例子演示了带有不同参数的 fadeTo() 方法：
    ```javascript
        $("button").click(function(){
          $("#div1").fadeTo("slow",0.15);
          $("#div2").fadeTo("slow",0.4);
          $("#div3").fadeTo("slow",0.7);
        });
    ```

### 滑动

- jQuery 拥有以下滑动方法：
    - slideDown()
    - slideUp()
    - slideToggle()

- slideDown() 方法

    jQuery slideDown() 方法用于向下滑动元素。

    - 语法:
    ```javascript
        $(selector).slideDown(speed,callback);
    ```
    可选的 speed 参数规定效果的时长。它可以取以下值："slow"、"fast" 或毫秒。
    
    可选的 callback 参数是滑动完成后所执行的函数名称。
    
    下面的例子演示了 slideDown() 方法：
    ```javascript
        $("#flip").click(function(){
          $("#panel").slideDown();
        });
    ```

- slideUp() 方法

    jQuery slideUp() 方法用于向上滑动元素。
    
    - 语法:
    ```javascript
        $(selector).slideUp(speed,callback);
    ```
    可选的 speed 参数规定效果的时长。它可以取以下值："slow"、"fast" 或毫秒。
    
    可选的 callback 参数是滑动完成后所执行的函数名称。
    
    下面的例子演示了 slideUp() 方法：
    ```javascript
        $("#flip").click(function(){
          $("#panel").slideUp();
        });
    ```

- slideToggle() 方法

    jQuery slideToggle() 方法可以在 slideDown() 与 slideUp() 方法之间进行切换。
    
    如果元素向下滑动，则 slideToggle() 可向上滑动它们。
    
    如果元素向上滑动，则 slideToggle() 可向下滑动它们。
    ```javascript
        $(selector).slideToggle(speed,callback);
    ```
    可选的 speed 参数规定效果的时长。它可以取以下值："slow"、"fast" 或毫秒。
    
    可选的 callback 参数是滑动完成后所执行的函数名称。
    
    下面的例子演示了 slideToggle() 方法：
    ```javascript
        $("#flip").click(function(){
          $("#panel").slideToggle();
        });
    ```

### Callback(回调) 方法

Callback 函数在当前动画 100% 完成之后执行。

- jQuery 动画的问题

    许多 jQuery 函数涉及动画。这些函数也许会将 speed 或 duration 作为可选参数。
    
    例子：$("p").hide("slow")
    
    speed 或 duration 参数可以设置许多不同的值，比如 "slow", "fast", "normal" 或毫秒。
    
    实例
    
    以下实例在隐藏效果完全实现后回调函数:
    ```javascript
        $("button").click(function(){
          $("p").hide("slow",function(){
            alert("段落现在被隐藏了");
          });
        });
    ```
    以下实例没有回调函数，警告框会在隐藏效果完成前弹出：
    ```javascript
        $("button").click(function(){
          $("p").hide(1000);
          alert("段落现在被隐藏了");
        });
    ```
- 被立即停止的动画不会触发回调，被立即完成的动画会触发回调。
    ```javascript
        $(document).ready(function(){
            
          $("button").click(function(){
            $("p").hide(3000,function(){
              alert("段落现在被隐藏了");
            });
          });
          $("#happy").click(function(){
              $("p").stop(false,true);
          });
        });
    ```

- 如果动画有队列的话，想实现其快速完成所有动画并停止，就要相应的与队列数对应条数的停止语句。
    ```javascript
        $(document).ready(function(){
          $("#start").click(function(){
            $("div").animate({left:'300px'},5000);
            $("div").animate({fontSize:'3em'},5000);
          });
        
          $("#stop1").click(function(){
            $("div").stop();
          });
        
          $("#stop2").click(function(){
            $("div").stop(true);
          });
        
          $("#stop3").click(function(){
            $("div").stop(false,true);
            $("div").stop(false,true);
          });
        });
    ```



