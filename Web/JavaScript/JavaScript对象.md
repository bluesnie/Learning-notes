###### datetime:2019/7/4 15:38
###### author:nzb

## 对象

- JavaScript 中的所有事物都是对象：字符串、数值、数组、函数...
    - 对象只是一种特殊的数据。对象拥有属性和方法。
    - 此外，JavaScript 允许自定义对象。

- 访问对象的属性
    - 属性是与对象相关的值。
    - 访问对象属性的语法是：
    `objectName.propertyName`
    
        下面这个例子使用了 String 对象的 length 属性来获得字符串的长度：
        
        `var message="Hello World!";`
        
        `var x=message.length;`
        
        在以上代码执行后，x 的值将是：`12`

- 访问对象的方法

    - 方法是能够在对象上执行的动作。
    - 您可以通过以下语法来调用方法：
    `objectName.methodName()`
        
        下面这个例子使用了 String 对象的 toUpperCase() 方法来将文本转换为大写：
        `var message="Hello world!";`
        
        `var x=message.toUpperCase();`
        
        在以上代码执行后，x 的值将是：`HELLO WORLD!`

- 创建 JavaScript 对象
    - 通过 JavaScript，您能够定义并创建自己的对象。
    - 创建新对象有两种不同的方法：
        - 定义并创建对象的实例
            ```javascript
                person=new Object();
                person.firstname="John";
                person.lastname="Doe";
                person.age=50;
                person.eyecolor="blue";
                // 或
                person={firstname:"John",lastname:"Doe",age:50,eyecolor:"blue"};
            ```
            
        - 使用函数来定义对象，然后创建新的对象实例
            ```javascript
                function person(firstname,lastname,age,eyecolor)
                {
                    this.firstname=firstname;
                    this.lastname=lastname;
                    this.age=age;
                    this.eyecolor=eyecolor;
                }
            ```
            在JavaScript中，this通常指向的是我们正在执行的函数本身，或者是指向该函数所属的对象（运行时）

- 创建 JavaScript 对象实例
    
    一旦您有了对象构造器，就可以创建新的对象实例，就像这样：
    - `var myFather=new person("John","Doe",50,"blue");`
    - `var myMother=new person("Sally","Rally",48,"green");`

- 把属性添加到 JavaScript 对象
    ```javascript
        person.firstname="John";
        person.lastname="Doe";
        person.age=30;
        person.eyecolor="blue";
        x=person.firstname;
    ```
    在以上代码执行后，x 的值将是：`John`

- 把方法添加到 JavaScript 对象
    ```javascript
        function person(firstname,lastname,age,eyecolor)
        {
            this.firstname=firstname;
            this.lastname=lastname;
            this.age=age;
            this.eyecolor=eyecolor;
        
            this.changeName=changeName;
            function changeName(name)
            {
                this.lastname=name;
            }
        }
    ```

## String对象

- String 对象用于处理已有的字符块。

- 一个字符串可以使用单引号或双引号。

- 字符串（String）使用长度属性length来计算字符串的长度。

- 字符串使用 indexOf() 来定位字符串中某一个指定的字符首次出现的位置。
    ```javascript
        var str="Hello world, welcome to the universe.";
        var n=str.indexOf("welcome");
    ```
    - 如果没找到对应的字符函数返回-1
    - lastIndexOf() 方法在字符串末尾开始查找字符串出现的位置。

- 内容匹配：match()函数用来查找字符串中特定的字符，并且如果找到的话，则返回这个字符。

- 替换内容：replace() 方法在字符串中用某些字符替换另一些字符。

- 字符串大小写转换：字符串大小写转换使用函数 toUpperCase() / toLowerCase()

- 字符串转为数组：字符串使用split()函数转为数组。

- 特殊字符：Javascript 中可以使用反斜线（\）插入特殊符号，如：撇号,引号等其他特殊符号。

    | 代码 | 输出 |
    |------|-----|
    | \' | 单引号 |
    | \" | 双引号 |
    | \\ | 斜杆 |
    | \n | 换行 |
    | \r | 回车 |
    | \t | tab |
    | \b | 空格 |
    | \f | 换页 |
    
- 字符串属性和方法
    - 属性:
    
        | 属性 | 描述 |
        |------|-----|
        | constructor | 对创建该对象的函数的引用 |
        | length | 字符串的长度 |
        | prototype | 允许您向对象添加属性和方法 |
        
    - 方法:
    
        | 方法 | 描述 |
        |-----|------|
        | charAt() | 	返回在指定位置的字符 |
        | charCodeAt() | 	返回在指定的位置的字符的 Unicode 编码 |
        | concat() | 	连接两个或更多字符串，并返回新的字符串 |
        | fromCharCode() | 	将 Unicode 编码转为字符 |
        | indexOf() | 	返回某个指定的字符串值在字符串中首次出现的位置 |
        | includes() | 	查找字符串中是否包含指定的子字符串 |
        | lastIndexOf() | 	从后向前搜索字符串，并从起始位置（0）开始计算返回字符串最后出现的位置 |
        | match() | 	查找找到一个或多个正则表达式的匹配 |
        | repeat() | 	复制字符串指定次数，并将它们连接在一起返回 |
        | replace() | 	在字符串中查找匹配的子串， 并替换与正则表达式匹配的子串 |
        | search() | 	查找与正则表达式相匹配的值 |
        | slice() | 	提取字符串的片断，并在新的字符串中返回被提取的部分 |
        | split() | 	把字符串分割为字符串数组 |
        | startsWith() | 	查看字符串是否以指定的子字符串开头 |
        | substr() | 	从起始索引号提取字符串中指定数目的字符 |
        | substring() | 	提取字符串中两个指定的索引号之间的字符 |
        | toLowerCase() | 	把字符串转换为小写 |
        | toUpperCase() | 	把字符串转换为大写 |
        | trim() | 去除字符串两边的空白 |
        | toLocaleLowerCase() | 	根据本地主机的语言环境把字符串转换为小写 |
        | toLocaleUpperCase() | 	根据本地主机的语言环境把字符串转换为大写 |
        | valueOf() | 	返回某个字符串对象的原始值 |
        | toString() | 	返回一个字符串 |

## Date日期对象

- Date() 方法获得当日的日期。
- getFullYear() 获取年份。
- getTime() 返回从 1970 年 1 月 1 日至今的毫秒数。
- setFullYear() 设置具体的日期。
- toUTCString() 将当日的日期（根据 UTC）转换为字符串。
- getDay() 和数组来显示星期，而不仅仅是数字。
- 创建日期
    - Date 对象用于处理日期和时间。 
    - 可以通过 new 关键词来定义 Date 对象。以下代码定义了名为 myDate 的 Date 对象：
    - 有四种方式初始化日期:
        - new Date() // 当前日期和时间
        - new Date(milliseconds) //返回从 1970 年 1 月 1 日至今的毫秒数
        - new Date(dateString)
        - new Date(year, month, day, hours, minutes, seconds, milliseconds)
    - 上面的参数大多数都是可选的，在不指定的情况下，默认参数是0。
        - 实例化一个日期的一些例子：
            - var today = new Date()
            - var d1 = new Date("October 13, 1975 11:13:00")
            - var d2 = new Date(79,5,24)
            - var d3 = new Date(79,5,24,11,33,0)
- 设置日期
    - 通过使用针对日期对象的方法，我们可以很容易地对日期进行操作。
    - 在下面的例子中，我们为日期对象设置了一个特定的日期 (2010 年 1 月 14 日)：
        - var myDate=new Date();
        - myDate.setFullYear(2010,0,14);
    - 在下面的例子中，我们将日期对象设置为 5 天后的日期：
        - var myDate=new Date();
        - myDate.setDate(myDate.getDate()+5);
    - **注意**: 如果增加天数会改变月份或者年份，那么日期对象会自动完成这种转换。
- 两个日期比较
    - 日期对象也可用于比较两个日期。
    - 下面的代码将当前日期与 2100 年 1 月 14 日做了比较：
        ```javascript
            var x=new Date();
            x.setFullYear(2100,0,14);
            var today = new Date();
            
            if (x>today)
            {
                alert("今天是2100年1月14日之前");
            }
            else
            {
                alert("今天是2100年1月14日之后");
            }
        ```
- 字符串属性和方法
    - 属性:
    
        | 属性 | 描述 |
        |------|-----|
        | constructor | 返回对创建此对象的 Date 函数的引用 |
        | prototype | 使您有能力向对象添加属性和方法 |
    - 方法:
    
        | 方法 | 描述 |
        |-----|------|
        | getDate() | 	从 Date 对象返回一个月中的某一天 (1 ~ 31) |
        | getDay() | 	从 Date 对象返回一周中的某一天 (0 ~ 6) |
        | getFullYear() | 	从 Date 对象以四位数字返回年份 |
        | getHours() | 	返回 Date 对象的小时 (0 ~ 23) |
        | getMilliseconds() | 	返回 Date 对象的毫秒(0 ~ 999) |
        | getMinutes() | 	返回 Date 对象的分钟 (0 ~ 59) |
        | getMonth() | 	从 Date 对象返回月份 (0 ~ 11) |
        | getSeconds() | 	返回 Date 对象的秒数 (0 ~ 59) |
        | getTime() | 	返回 1970 年 1 月 1 日至今的毫秒数 |
        | getTimezoneOffset() | 	返回本地时间与格林威治标准时间 (GMT) 的分钟差 |
        | getUTCDate() | 	根据世界时从 Date 对象返回月中的一天 (1 ~ 31) |
        | getUTCDay() | 	根据世界时从 Date 对象返回周中的一天 (0 ~ 6) |
        | getUTCFullYear() | 	根据世界时从 Date 对象返回四位数的年份 |
        | getUTCHours() | 	根据世界时返回 Date 对象的小时 (0 ~ 23) |
        | getUTCMilliseconds() | 	根据世界时返回 Date 对象的毫秒(0 ~ 999) |
        | getUTCMinutes() | 	根据世界时返回 Date 对象的分钟 (0 ~ 59) |
        | getUTCMonth() | 	根据世界时从 Date 对象返回月份 (0 ~ 11) |
        | getUTCSeconds() | 	根据世界时返回 Date 对象的秒钟 (0 ~ 59) |
        | getYear() |	已废弃。 请使用 getFullYear() 方法代替 |
        | parse() | 	返回1970年1月1日午夜到指定日期（字符串）的毫秒数 |
        | setDate() | 	设置 Date 对象中月的某一天 (1 ~ 31) |
        | setFullYear() | 	设置 Date 对象中的年份（四位数字） |
        | setHours() | 	设置 Date 对象中的小时 (0 ~ 23) |
        | setMilliseconds() | 	设置 Date 对象中的毫秒 (0 ~ 999) |
        | setMinutes() | 	设置 Date 对象中的分钟 (0 ~ 59) |
        | setMonth() | 	设置 Date 对象中月份 (0 ~ 11) |
        | setSeconds() | 	设置 Date 对象中的秒钟 (0 ~ 59) |
        | setTime()	setTime() | 方法以毫秒设置 Date 对象 |
        | setUTCDate() | 	根据世界时设置 Date 对象中月份的一天 (1 ~ 31) |
        | setUTCFullYear() | 	根据世界时设置 Date 对象中的年份（四位数字） |
        | setUTCHours() | 	根据世界时设置 Date 对象中的小时 (0 ~ 23) |
        | setUTCMilliseconds() | 	根据世界时设置 Date 对象中的毫秒 (0 ~ 999) |
        | setUTCMinutes() | 	根据世界时设置 Date 对象中的分钟 (0 ~ 59) |
        | setUTCMonth() | 	根据世界时设置 Date 对象中的月份 (0 ~ 11) |
        | setUTCSeconds() | 	setUTCSeconds() 方法用于根据世界时 (UTC) 设置指定时间的秒字段 |
        | setYear() | 	已废弃。请使用 setFullYear() 方法代替。
        | toDateString() | 	把 Date 对象的日期部分转换为字符串 |
        | toGMTString() | 	已废弃。请使用 toUTCString() 方法代替 |
        | toISOString() | 	使用 ISO 标准返回字符串的日期格式 |
        | toJSON() | 	以 JSON 数据格式返回日期字符串 |
        | toLocaleDateString() | 	根据本地时间格式，把 Date 对象的日期部分转换为字符串 |
        | toLocaleTimeString() | 	根据本地时间格式，把 Date 对象的时间部分转换为字符串 |
        | toLocaleString() | 	据本地时间格式，把 Date 对象转换为字符串 |
        | toString() | 	把 Date 对象转换为字符串 |
        | toTimeString() | 	把 Date 对象的时间部分转换为字符串 |
        | toUTCString() | 	根据世界时，把 Date 对象转换为字符串 |
        | UTC() | 	根据世界时返回 1970 年 1 月 1 日 到指定日期的毫秒数 |
        | valueOf() | 	返回 Date 对象的原始值 |
    
    

## Array对象

- 数组对象是使用单独的变量名来存储一系列的值。
    - 数组可以用一个变量名存储所有的值，并且可以用变量名访问任何一个值。
    - 数组中的每个元素都有自己的的ID，以便它可以很容易地被访问到。

- 创建一个数组
    - 常规方式:
        `var myCars=new Array(); 
        myCars[0]="Saab";       
        myCars[1]="Volvo";
        myCars[2]="BMW";`
    - 简洁方式:
        `var myCars=new Array("Saab","Volvo","BMW");`
    - 字面:
        `var myCars=["Saab","Volvo","BMW"];`

- 访问数组

    通过指定数组名以及索引号码，你可以访问某个特定的元素。

- 在一个数组中你可以有不同的对象

    所有的JavaScript变量都是对象。数组元素是对象。函数是对象。
    因此，你可以在数组中有不同的变量类型。
    你可以在一个数组中包含对象元素、函数、数组

- 数组方法和属性
    - 属性:
    
        | 属性 | 描述 |
        |------|-----|
        | constructor |	返回创建数组对象的原型函数 |
        | length | 设置或返回数组元素的个数 |
        | prototype | 允许你向数组对象添加属性或方法 |

    - 方法:
    
        | 方法 | 描述 |
        |-----|------|
        | concat() | 	连接两个或更多的数组，并返回结果 |
        | copyWithin() | 	从数组的指定位置拷贝元素到数组的另一个指定位置中 |
        | entries() | 	返回数组的可迭代对象 |
        | every() | 	检测数值元素的每个元素是否都符合条件 |
        | fill() | 	使用一个固定值来填充数组 |
        | filter() | 	检测数值元素，并返回符合条件所有元素的数组 |
        | find() | 	返回符合传入测试（函数）条件的数组元素 |
        | findIndex() | 	返回符合传入测试（函数）条件的数组元素索引 |
        | forEach() | 	数组每个元素都执行一次回调函数 |
        | from() | 	通过给定的对象中创建一个数组 |
        | includes() | 	判断一个数组是否包含一个指定的值 |
        | indexOf() | 	搜索数组中的元素，并返回它所在的位置 |
        | isArray() | 	判断对象是否为数组 |
        | join() | 	把数组的所有元素放入一个字符串 |
        | keys() | 	返回数组的可迭代对象，包含原始数组的键(key) |
        | lastIndexOf() | 	搜索数组中的元素，并返回它最后出现的位置 |
        | map() | 	通过指定函数处理数组的每个元素，并返回处理后的数组 |
        | pop() | 	删除数组的最后一个元素并返回删除的元素 |
        | push() | 	向数组的末尾添加一个或更多元素，并返回新的长度 |
        | reduce() | 	将数组元素计算为一个值（从左到右） |
        | reduceRight() | 	将数组元素计算为一个值（从右到左） |
        | reverse() | 	反转数组的元素顺序 |
        | shift() | 	删除并返回数组的第一个元素 |
        | slice() | 	选取数组的的一部分，并返回一个新数组 |
        | some() | 	检测数组元素中是否有元素符合指定条件 |
        | sort() | 	对数组的元素进行排序 |
        | splice() | 	从数组中添加或删除元素 |
        | toString() | 	把数组转换为字符串，并返回结果 |
        | unshift() | 	向数组的开头添加一个或更多元素，并返回新的长度 |
        | valueOf() | 	返回数组对象的原始值 |

## Math对象

- Math方法和属性
    - 属性:
    
        | 属性 | 描述 |
        |------|-----|
        | E | 返回算术常量 e，即自然对数的底数（约等于2.718） |
        | LN2 | 返回 2 的自然对数（约等于0.693） |
        | LN10 | 返回 10 的自然对数（约等于2.302） |
        | LOG2E | 返回以 2 为底的 e 的对数（约等于 1.4426950408889634） |
        | LOG10E | 返回以 10 为底的 e 的对数（约等于0.434） |
        | PI | 返回圆周率（约等于3.14159） |
        | SQRT1_2 | 返回 2 的平方根的倒数（约等于 0.707） |
        | SQRT2 | 返回 2 的平方根（约等于 1.414） |
    - 方法:
    
        | 方法 | 描述 |
        |-----|------|
        | abs(x) | 	返回 x 的绝对值 |
        | acos(x) | 	返回 x 的反余弦值 |
        | asin(x) | 	返回 x 的反正弦值 |
        | atan(x) | 	以介于 -PI/2 与 PI/2 弧度之间的数值来返回 x 的反正切值 |
        | atan2(y,x) |	返回从 x 轴到点 (x,y) 的角度（介于 -PI/2 与 PI/2 弧度之间） |
        | ceil(x) | 	对数进行上舍入 |
        | cos(x) | 	返回数的余弦 |
        | exp(x) | 	返回 Ex 的指数 |
        | floor(x) | 	对 x 进行下舍入 |
        | log(x) | 	返回数的自然对数（底为e） |
        | max(x,y,z,...,n) |	返回 x,y,z,...,n 中的最高值 |
        | min(x,y,z,...,n) |	返回 x,y,z,...,n中的最低值 |
        | pow(x,y) | 返回 x 的 y 次幂 |
        | random() | 	返回 0 ~ 1 之间的随机数 |
        | round(x) | 	四舍五入 |
        | sin(x) | 	返回数的正弦 |
        | sqrt(x) | 	返回数的平方根 |
        | tan(x) | 	返回角的正切 |

## DOM对象

- HTML DOM 节点

    在 HTML DOM (Document Object Model) 中 , 每一个元素都是 节点:
    - 文档是一个文档节点。
    - 所有的HTML元素都是元素节点。
    - 所有 HTML 属性都是属性节点。
    - 文本插入到 HTML 元素是文本节点。are text nodes。
    - 注释是注释节点。

- Document 对象
    - 当浏览器载入 HTML 文档, 它就会成为 Document 对象。
    - Document 对象是 HTML 文档的根节点。
    - Document 对象使我们可以从脚本中对 HTML 页面中的所有元素进行访问。
    - 提示：Document 对象是 Window 对象的一部分，可通过 window.document 属性对其进行访问。

- Document 对象属性和方法

    - 属性和方法

        | 属性 / 方法 | 描述 |
        |------|------------|
        | document.activeElement | 返回当前获取焦点元素 |
        | document.addEventListener() | 向文档添加句柄 |
        | document.adoptNode(node) | 从另外一个文档返回 adapded 节点到当前文档 |
        | document.anchors | 返回对文档中所有 Anchor 对象的引用 |
        | document.applets | 返回对文档中所有 Applet 对象的引用。注意: HTML5 已不支持 <applet> 元素 |
        | document.baseURI | 返回文档的绝对基础 URI |
        | document.body | 返回文档的body元素 |
        | document.close() | 关闭用 document.open() 方法打开的输出流，并显示选定的数据 |
        | document.cookie | 设置或返回与当前文档有关的所有 cookie |
        | document.createAttribute() | 创建一个属性节点 |
        | document.createComment() | createComment() 方法可创建注释节点 |
        | document.createDocumentFragment() | 创建空的 DocumentFragment 对象，并返回此对象 |
        | document.createElement() | 创建元素节点 |
        | document.createTextNode() | 创建文本节点 |
        | document.doctype | 返回与文档相关的文档类型声明 (DTD) |
        | document.documentElement | 返回文档的根节点 |
        | document.documentMode | 返回用于通过浏览器渲染文档的模式 |
        | document.documentURI | 设置或返回文档的位置 |
        | document.domain | 返回当前文档的域名 |
        | document.domConfig | 已废弃。返回 normalizeDocument() 被调用时所使用的配置 |
        | document.embeds | 返回文档中所有嵌入的内容（embed）集合 |
        | document.forms | 返回对文档中所有 Form 对象引用 |
        | document.getElementsByClassName() | 返回文档中所有指定类名的元素集合，作为 NodeList 对象 |
        | document.getElementById() | 返回对拥有指定 id 的第一个对象的引用 |
        | document.getElementsByName() | 返回带有指定名称的对象集合 |
        | document.getElementsByTagName() | 返回带有指定标签名的对象集合 |
        | document.images | 返回对文档中所有 Image 对象引用 |
        | document.implementation | 返回处理该文档的 DOMImplementation 对象 |
        | document.importNode() | 把一个节点从另一个文档复制到该文档以便应用 |
        | document.inputEncoding | 返回用于文档的编码方式（在解析时） |
        | document.lastModified | 返回文档被最后修改的日期和时间 |
        | document.links | 返回对文档中所有 Area 和 Link 对象引用 |
        | document.normalize() | 删除空文本节点，并连接相邻节点 |
        | document.normalizeDocument() | 删除空文本节点，并连接相邻节点的 |
        | document.open() | 打开一个流，以收集来自任何 document.write() 或 document.writeln() 方法的输出 |
        | document.querySelector() | 返回文档中匹配指定的CSS选择器的第一元素 |
        | document.querySelectorAll() | document.querySelectorAll() 是 HTML5中引入的新方法，返回文档中匹配的CSS选择器的所有元素节点列表 |
        | document.readyState | 返回文档状态 (载入中……) |
        | document.referrer | 返回载入当前文档的文档的 URL |
        | document.removeEventListener() | 移除文档中的事件句柄(由 addEventListener() 方法添加) |
        | document.renameNode() | 重命名元素或者属性节点 |
        | document.scripts | 返回页面中所有脚本的集合 |
        | document.strictErrorChecking | 设置或返回是否强制进行错误检查 |
        | document.title | 返回当前文档的标题 |
        | document.URL | 返回文档完整的URL |
        | document.write() | 向文档写 HTML 表达式 或 JavaScript 代码 |
        | document.writeln() | 等同于 write() 方法，不同的是在每个表达式之后写一个换行符 |
    
    - **警告 !!!**
        在 W3C DOM核心，文档对象 继承节点对象的所有属性和方法。
        
        很多属性和方法在文档中是没有意义的。
        
        HTML 文档对象可以避免使用这些节点对象和属性：
        
        | 属性 / 方法 | 避免的原因 |
        |-------|----------------|
        | document.attributes | 文档没有该属性 |
        | document.hasAttributes() | 文档没有该属性 |
        | document.nextSibling | 文档没有下一节点 |
        | document.nodeName | 这个通常是 #document |
        | document.nodeType | 这个通常是 9(DOCUMENT_NODE) |
        | document.nodeValue | 文档没有一个节点值 |
        | document.ownerDocument | 文档没有主文档 |
        | document.ownerElement | 文档没有自己的节点 |
        | document.parentNode | 文档没有父节点 |
        | document.previousSibling | 文档没有兄弟节点 |
        | document.textContent | 文档没有文本节点 |

- 元素对象
    - 在 HTML DOM 中, 元素对象代表着一个 HTML 元素。
    - 元素对象 的 子节点可以是, 可以是元素节点，文本节点，注释节点。
    - NodeList 对象 代表了节点列表，类似于 HTML元素的子节点集合。
    - 属性和方法
    
        | 属性 / 方法 | 描述 |
        |------|------------|
        | element.accessKey | 设置或返回accesskey一个元素 |
        | element.addEventListener() | 向指定元素添加事件句柄 |
        | element.appendChild() | 为元素添加一个新的子元素 |
        | element.attributes | 返回一个元素的属性数组 |
        | element.childNodes | 返回元素的一个子节点的数组 |
        | element.children | 返回元素的子元素的集合 |
        | element.classList | 返回元素的类名，作为 DOMTokenList 对象。 |
        | element.className | 设置或返回元素的class属性 |
        | element.clientHeight | 在页面上返回内容的可视高度（不包括边框，边距或滚动条） |
        | element.clientWidth | 在页面上返回内容的可视宽度（不包括边框，边距或滚动条） |
        | element.cloneNode() | 克隆某个元素 |
        | element.compareDocumentPosition() | 比较两个元素的文档位置。 |
        | element.contentEditable | 设置或返回元素的内容是否可编辑 |
        | element.dir | 设置或返回一个元素中的文本方向 |
        | element.firstChild | 返回元素的第一个子节点 |
        | element.focus() | 设置文档或元素获取焦点 |
        | element.getAttribute() | 返回指定元素的属性值 |
        | element.getAttributeNode() | 返回指定属性节点 |
        | element.getElementsByTagName() | 返回指定标签名的所有子元素集合。 |
        | element. getElementsByClassName() | 返回文档中所有指定类名的元素集合，作为 NodeList 对象。 |
        | element.getFeature() | 返回指定特征的执行APIs对象。 |
        | element.getUserData() | 返回一个元素中关联键值的对象。 |
        | element.hasAttribute() | 如果元素中存在指定的属性返回 true，否则返回false。 |
        | element.hasAttributes() | 如果元素有任何属性返回true，否则返回false。 |
        | element.hasChildNodes() | 返回一个元素是否具有任何子元素 |
        | element.hasFocus() | 返回布尔值，检测文档或元素是否获取焦点 |
        | element.id | 设置或者返回元素的 id。 |
        | element.innerHTML | 设置或者返回元素的内容。 |
        | element.insertBefore() | 现有的子元素之前插入一个新的子元素 |
        | element.isContentEditable | 如果元素内容可编辑返回 true，否则返回false |
        | element.isDefaultNamespace() | 如果指定了namespaceURI 返回 true，否则返回 false。 |
        | element.isEqualNode() | 检查两个元素是否相等 |
        | element.isSameNode() | 检查两个元素所有有相同节点。 |
        | element.isSupported() | 如果在元素中支持指定特征返回 true。 |
        | element.lang | 设置或者返回一个元素的语言。 |
        | element.lastChild | 返回的最后一个子元素 |
        | element.namespaceURI | 返回命名空间的 URI。 |
        | element.nextSibling | 返回该元素紧跟的一个节点 |
        | element.nextElementSibling | 返回指定元素之后的下一个兄弟元素（相同节点树层中的下一个元素节点）。 |
        | element.nodeName | 返回元素的标记名（大写） |
        | element.nodeType | 返回元素的节点类型 |
        | element.nodeValue | 返回元素的节点值 |
        | element.normalize() | 使得此成为一个"normal"的形式，其中只有结构（如元素，注释，处理指令，CDATA节和实体引用）隔开Text节点，即元素（包括属性）下面的所有文本节点，既没有相邻的文本节点也没有空的文本节点 |
        | element.offsetHeight | 返回任何一个元素的高度包括边框和填充，但不是边距 |
        | element.offsetWidth | 返回元素的宽度，包括边框和填充，但不是边距 |
        | element.offsetLeft | 返回当前元素的相对水平偏移位置的偏移容器 |
        | element.offsetParent | 返回元素的偏移容器 |
        | element.offsetTop | 返回当前元素的相对垂直偏移位置的偏移容器 |
        | element.ownerDocument | 返回元素的根元素（文档对象） |
        | element.parentNode | 返回元素的父节点 |
        | element.previousSibling | 返回某个元素紧接之前元素 |
        | element.previousElementSibling | 返回指定元素的前一个兄弟元素（相同节点树层中的前一个元素节点）。 |
        | element.querySelector() | 返回匹配指定 CSS 选择器元素的第一个子元素 |
        | document.querySelectorAll() | 返回匹配指定 CSS 选择器元素的所有子元素节点列表 |
        | element.removeAttribute() | 从元素中删除指定的属性 |
        | element.removeAttributeNode() | 删除指定属性节点并返回移除后的节点。 |
        | element.removeChild() | 删除一个子元素 |
        | element.removeEventListener() | 移除由 addEventListener() 方法添加的事件句柄 |
        | element.replaceChild() | 替换一个子元素 |
        | element.scrollHeight | 返回整个元素的高度（包括带滚动条的隐蔽的地方） |
        | element.scrollLeft | 返回当前视图中的实际元素的左边缘和左边缘之间的距离 |
        | element.scrollTop | 返回当前视图中的实际元素的顶部边缘和顶部边缘之间的距离 |
        | element.scrollWidth | 返回元素的整个宽度（包括带滚动条的隐蔽的地方） |
        | element.setAttribute() | 设置或者改变指定属性并指定值。 |
        | element.setAttributeNode() | 设置或者改变指定属性节点。 |
        | element.setIdAttribute() |  |
        | element.setIdAttributeNode() |  |
        | element.setUserData() | 在元素中为指定键值关联对象。 |
        | element.style | 设置或返回元素的样式属性 |
        | element.tabIndex | 设置或返回元素的标签顺序。 |
        | element.tagName | 作为一个字符串返回某个元素的标记名（大写） |
        | element.textContent | 设置或返回一个节点和它的文本内容 |
        | element.title | 设置或返回元素的title属性 |
        | element.toString() | 一个元素转换成字符串 |
        | nodelist.item() | 返回某个元素基于文档树的索引 |
        | nodelist.length | 返回节点列表的节点数目。 |

- Attr(属性) 对象
    - 在 HTML DOM 中, Attr 对象 代表一个 HTML 属性。
    - HTML属性总是属于HTML元素。

    - NamedNodeMap 对象
        - 在 HTML DOM 中, the NamedNodeMap 对象 表示一个无顺序的节点列表。
        - 我们可通过节点名称来访问 NamedNodeMap 中的节点。
    - 属性和方法
    
        | 属性 / 方法 | 描述 |
        |------|------------|
        | attr.isId | 如果属性是 ID 类型，则 isId 属性返回 true，否则返回 false。 |
        | attr.name | 返回属性名称 |
        | attr.value | 设置或者返回属性值 |
        | attr.specified | 如果属性被指定返回 true ，否则返回 false |
        | nodemap.getNamedItem() | 从节点列表中返回的指定属性节点。 |
        | nodemap.item() | 返回节点列表中处于指定索引号的节点。 |
        | nodemap.length | 返回节点列表的节点数目。 |
        | nodemap.removeNamedItem() | 删除指定属性节点 |
        | nodemap.setNamedItem() | 设置指定属性节点(通过名称) |

    - **DOM 4 警告 !!!**
        - 在 W3C DOM 内核中, Attr (属性) 对象继承节点对象的所有属性和方法 。
        - 在 DOM 4 中, Attr (属性) 对象不再从节点对象中继承。 
        - 从长远的代码质量来考虑，在属性对象中你需要避免使用节点对象属性和方法:
        
            | 属性 / 方法 | 避免原因 |
            |------|------------|
            | attr.appendChild() | 属性没有子节点 |
            | attr.attributes | 属性没有属性 |
            | attr.baseURI | 使用 document.baseURI 替代 |
            | attr.childNodes | 属性没有子节点 |
            | attr.cloneNode() | 使用 attr.value 替代 |
            | attr.firstChild | 属性没有子节点 |
            | attr.hasAttributes() | 属性没有属性 |
            | attr.hasChildNodes | 属性没有子节点 |
            | attr.insertBefore() | 属性没有子节点 |
            | attr.isEqualNode() | 没有意义 |
            | attr.isSameNode() | 没有意义 |
            | attr.isSupported() | 通常为 true |
            | attr.lastChild | 属性没有子节点 |
            | attr.nextSibling | 属性没有兄弟节点 |
            | attr.nodeName | 使用 attr.name 替代 |
            | attr.nodeType | 通常为 2 (ATTRIBUTE-NODE) |
            | attr.nodeValue | 使用 attr.value 替代 |
            | attr.normalize() | 属性没有规范 |
            | attr.ownerDocument | 通常为你的 HTML 文档 |
            | attr.ownerElement | 你用来访问属性的 HTML 元素 |
            | attr.parentNode | 你用来访问属性的 HTML 元素 |
            | attr.previousSibling | 属性没有兄弟节点 |
            | attr.removeChild | 属性没有子节点 |
            | attr.replaceChild | 属性没有子节点 |
            | attr.textContent | 使用 attr.value 替代 |

- Console 对象
    
    Console 对象提供了访问浏览器调试模式的信息到控制台。
    
    | 方法 | 描述 |
    |------|------------|
    | assert() | 如果断言为 false，则在信息到控制台输出错误信息。 |
    | clear() | 清除控制台上的信息。 |
    | count() | 记录 count() 调用次数，一般用于计数。 |
    | error() | 输出错误信息到控制台 |
    | group() | 在控制台创建一个信息分组。 一个完整的信息分组以 console.group() 开始，console.groupEnd() 结束 |
    | groupCollapsed() | 在控制台创建一个信息分组。 类似 console.group() ，但它默认是折叠的。 |
    | groupEnd() | 设置当前信息分组结束 |
    | info() | 控制台输出一条信息 |
    | log() | 控制台输出一条信息 |
    | table() | 以表格形式显示数据 |
    | time() | 计时器，开始计时间，与 timeEnd() 联合使用，用于算出一个操作所花费的准确时间。 |
    | timeEnd() | 计时结束 |
    | trace() | 显示当前执行的代码在堆栈中的调用路径。 |
    | warn() | 输出警告信息，信息最前面加一个黄色三角，表示警告 |

- CSS 样式声明对象(CSSStyleDeclaration)
    - CSSStyleDeclaration 对象
    
        CSSStyleDeclaration 对象表示一个 CSS 属性-值（property-value）对的集合。
    
    - CSSStyleDeclaration 对象属性
    
        | 属性 | 描述 |
        |------|------------|
        | cssText | 设置或返回样式声明文本，cssText 对应的是 HTML 元素的 style 属性。 |
        | length | 返回样式中包含多少条声明。 |
        | parentRule | 返回包含当前规则的规则。 |
    
    - CSSStyleDeclaration 对象方法
    
        | 属性 | 描述 |
        |------|------------|
        | getPropertyPriority() | 返回指定的 CSS 属性是否设置了 "important!" 属性。 |
        | getPropertyValue() | 返回指定的 CSS 属性值。 |
        | item() | 通过索引方式返回 CSS 声明中的 CSS 属性名。 |
        | removeProperty() | 移除 CSS 声明中的 CSS 属性。 |
        | setProperty() | 在 CSS 声明块中新建或者修改 CSS 属性。 |
    
- DOM 事件
    
    DOM： 指明使用的 DOM 属性级别。
    
    - 鼠标事件
    
        | 属性 | 描述 | DOM |
        |------|------|------|
        | onclick | 当用户点击某个对象时调用的事件句柄。 | 2 |
        | oncontextmenu | 在用户点击鼠标右键打开上下文菜单时触发 | |	 
        | ondblclick | 当用户双击某个对象时调用的事件句柄。 | 2 |
        | onmousedown | 鼠标按钮被按下。 | 2 |
        | onmouseenter | 当鼠标指针移动到元素上时触发。 | 2 |
        | onmouseleave | 当鼠标指针移出元素时触发 | 2 |
        | onmousemove | 鼠标被移动。 | 2 |
        | onmouseover | 鼠标移到某元素之上。 | 2 |
        | onmouseout | 鼠标从某元素移开。 | 2 |
        | onmouseup | 鼠标按键被松开。 | 2 |
        
    - 键盘事件
    
        | 属性 | 描述 | DOM |
        |------|------|------|
        | onkeydown | 某个键盘按键被按下。 | 2 |
        | onkeypress | 某个键盘按键被按下并松开。 | 2 |
        | onkeyup | 某个键盘按键被松开。 | 2 |
        
    - 框架/对象（Frame/Object）事件     

        | 属性 | 描述 | DOM |
        |------|------|------|
        | onabort | 图像的加载被中断。 ( `<object>`) | 2 |
        | onbeforeunload | 该事件在即将离开页面（刷新或关闭）时触发 | 2 |
        | onerror | 在加载文档或图像时发生错误。 ( `<object>`, `<body>`和 `<frameset>`) |  | 
        | onhashchange | 该事件在当前 URL 的锚部分发生修改时触发。 |  | 
        | onload | 一张页面或一幅图像完成加载。 | 2 |
        | onpageshow | 该事件在用户访问页面时触发 |  |
        | onpagehide | 该事件在用户离开当前网页跳转到另外一个页面时触发 |  |
        | onresize | 窗口或框架被重新调整大小。 | 2 |
        | onscroll | 当文档被滚动时发生的事件。 | 2 |
        | onunload | 用户退出页面。 ( `<body>` 和 `<frameset>`) | 2 |
        
    - 表单事件
        
        | 属性 | 描述 | DOM |
        |------|------|------|
        | onblur | 元素失去焦点时触发 | 2 |
        | onchange | 该事件在表单元素的内容改变时触发( `<input>`, `<keygen>`, `<select>`, 和 `<textarea>`) | 2 |
        | onfocus | 元素获取焦点时触发 | 2 |
        | onfocusin | 元素即将获取焦点时触发 | 2 |
        | onfocusout | 元素即将失去焦点时触发 | 2 |
        | oninput | 元素获取用户输入时触发 | 3 |
        | onreset | 表单重置时触发 | 2 |
        | onsearch | 用户向搜索域输入文本时触发 ( `<input="search">`) |  | 
        | onselect | 用户选取文本时触发 ( `<input> 和 <textarea>`) | 2 |
        | onsubmit | 表单提交时触发 | 2 |
    
    - 剪贴板事件
        
        | 属性 | 描述 | DOM |
        |------|------|------|
        | oncopy | 该事件在用户拷贝元素内容时触发 |  | 
        | oncut | 该事件在用户剪切元素内容时触发 |  | 
        | onpaste | 该事件在用户粘贴元素内容时触发 |  | 
    
    - 打印事件
        
        | 属性 | 描述 | DOM |
        |------|------|------| 
        | onafterprint | 该事件在页面已经开始打印，或者打印窗口已经关闭时触发 |  | 
        | onbeforeprint | 该事件在页面即将开始打印时触发 |  | 
        
    - 拖动事件
        
        | 事件 | 描述 | DOM |
        |------|------|------|
        | ondrag | 该事件在元素正在拖动时触发 |  | 
        | ondragend | 该事件在用户完成元素的拖动时触发 |  | 
        | ondragenter | 该事件在拖动的元素进入放置目标时触发 |  | 
        | ondragleave | 该事件在拖动元素离开放置目标时触发 |  | 
        | ondragover | 该事件在拖动元素在放置目标上时触发 |  | 
        | ondragstart | 该事件在用户开始拖动元素时触发 |  | 
        | ondrop | 该事件在拖动元素放置在目标区域时触发 |  | 

    - 多媒体（Media）事件
        
        | 事件 | 描述 | DOM |
        |------|------|------|
        | onabort | 事件在视频/音频（audio/video）终止加载时触发。 |  | 
        | oncanplay | 事件在用户可以开始播放视频/音频（audio/video）时触发。 |  | 
        | oncanplaythrough | 事件在视频/音频（audio/video）可以正常播放且无需停顿和缓冲时触发。 |  | 
        | ondurationchange | 事件在视频/音频（audio/video）的时长发生变化时触发。 |  | 
        | onemptied | 当期播放列表为空时触发 |  | 
        | onended | 事件在视频/音频（audio/video）播放结束时触发。 |  | 
        | onerror | 事件在视频/音频（audio/video）数据加载期间发生错误时触发。 |  | 
        | onloadeddata | 事件在浏览器加载视频/音频（audio/video）当前帧时触发触发。 |  | 
        | onloadedmetadata | 事件在指定视频/音频（audio/video）的元数据加载后触发。 |  | 
        | onloadstart | 事件在浏览器开始寻找指定视频/音频（audio/video）触发。 |  | 
        | onpause | 事件在视频/音频（audio/video）暂停时触发。 |  | 
        | onplay | 事件在视频/音频（audio/video）开始播放时触发。 |  | 
        | onplaying | 事件在视频/音频（audio/video）暂停或者在缓冲后准备重新开始播放时触发。 |  | 
        | onprogress | 事件在浏览器下载指定的视频/音频（audio/video）时触发。 |  | 
        | onratechange | 事件在视频/音频（audio/video）的播放速度发送改变时触发。 |  | 
        | onseeked | 事件在用户重新定位视频/音频（audio/video）的播放位置后触发。 |  | 
        | onseeking | 事件在用户开始重新定位视频/音频（audio/video）时触发。 |  | 
        | onstalled | 事件在浏览器获取媒体数据，但媒体数据不可用时触发。 |  | 
        | onsuspend | 事件在浏览器读取媒体数据中止时触发。 |  | 
        | ontimeupdate | 事件在当前的播放位置发送改变时触发。 |  | 
        | onvolumechange | 事件在音量发生改变时触发。 |  | 
        | onwaiting | 事件在视频由于要播放下一帧而需要缓冲时触发。 |  | 

    - 动画事件
        
        | 事件 | 描述 | DOM |
        |------|------|------|
        | animationend | 该事件在 CSS 动画结束播放时触发 |  | 
        | animationiteration | 该事件在 CSS 动画重复播放时触发 |  | 
        | animationstart | 该事件在 CSS 动画开始播放时触发 |  | 

    - 过渡事件
        
        | 事件 | 描述 | DOM |
        |------|------|------|
        | transitionend | 该事件在 CSS 完成过渡后触发。 |  | 

    - 其他事件
        
        | 事件 | 描述 | DOM |
        |------|------|------|
        | onmessage | 该事件通过或者从对象(WebSocket, Web Worker, Event Source 或者子 frame 或父窗口)接收到消息时触发 |  | 
        | onmousewheel | 已废弃。 使用 onwheel 事件替代 |  | 
        | ononline | 该事件在浏览器开始在线工作时触发。 |  | 
        | onoffline | 该事件在浏览器开始离线工作时触发。 |  | 
        | onpopstate | 该事件在窗口的浏览历史（history 对象）发生改变时触发。 |  | 
        | onshow | 该事件当 <menu> 元素在上下文菜单显示时触发 |  | 
        | onstorage | 该事件在 Web Storage(HTML 5 Web 存储)更新时触发 |  | 
        | ontoggle | 该事件在用户打开或关闭 <details> 元素时触发 |  | 
        | onwheel | 该事件在鼠标滚轮在元素上下滚动时触发 |  | 

    - 事件对象
        - 常量
        
            | 静态变量 | 描述 | DOM |
            |------|------|------|
            | CAPTURING-PHASE | 当前事件阶段为捕获阶段(1) | 1 |
            | AT-TARGET | 当前事件是目标阶段,在评估目标事件(1) | 2 |
            | BUBBLING-PHASE | 当前的事件为冒泡阶段 (3) | 3 |

        - 属性
        
            | 属性 | 描述 | DOM |
            |------|------|------|
            | bubbles | 返回布尔值，指示事件是否是起泡事件类型。 | 2 |
            | cancelable | 返回布尔值，指示事件是否可拥可取消的默认动作。 | 2 |
            | currentTarget | 返回其事件监听器触发该事件的元素。 | 2 |
            | eventPhase | 返回事件传播的当前阶段。 | 2 |
            | target | 返回触发此事件的元素（事件的目标节点）。 | 2 |
            | timeStamp | 返回事件生成的日期和时间。 | 2 |
            | type | 返回当前 Event 对象表示的事件的名称。 | 2 |

        - 方法
        
            | 方法 | 描述 | DOM |
            |------|------|------|
            | initEvent() | 初始化新创建的 Event 对象的属性。 | 2 |
            | preventDefault() | 通知浏览器不要执行与事件关联的默认动作。 | 2 |
            | stopPropagation() | 不再派发事件。 | 2 |

    - 目标事件对象
        - 方法
        
            | 方法 | 描述 | DOM |
            |------|------|------|
            | addEventListener() | 允许在目标事件中注册监听事件(IE8 = attachEvent()) | 2 |
            | dispatchEvent() | 允许发送事件到监听器上 (IE8 = fireEvent()) | 2 |
            | removeEventListener() | 运行一次注册在事件目标上的监听事件(IE8 = detachEvent()) | 2 |
        
    - 事件监听对象
        - 方法
        
            | 方法 | 描述 | DOM |
            |------|------|------|
            | handleEvent() | 把任意对象注册为事件处理程序 | 2 |

    - 文档事件对象
        - 方法
        
            | 方法 | 描述 | DOM |
            |------|------|------|
            | createEvent() |   | 2 |

    - 鼠标/键盘事件对象
        - 属性
        
            | 属性 | 描述 | DOM |
            |------|------|------|
            | altKey | 返回当事件被触发时，"ALT" 是否被按下。 | 2 |
            | button | 返回当事件被触发时，哪个鼠标按钮被点击。 | 2 |
            | clientX | 返回当事件被触发时，鼠标指针的水平坐标。 | 2 |
            | clientY | 返回当事件被触发时，鼠标指针的垂直坐标。 | 2 |
            | ctrlKey | 返回当事件被触发时，"CTRL" 键是否被按下。 | 2 |
            | Location | 返回按键在设备上的位置 | 3 |
            | charCode | 返回onkeypress事件触发键值的字母代码。 | 2 |
            | key | 在按下按键时返回按键的标识符。 | 3 |
            | keyCode | 返回onkeypress事件触发的键的值的字符代码，或者 onkeydown 或 onkeyup 事件的键的代码。 | 2 |
            | which | 返回onkeypress事件触发的键的值的字符代码，或者 onkeydown 或 onkeyup 事件的键的代码。 | 2 |
            | metaKey | 返回当事件被触发时，"meta" 键是否被按下。 | 2 |
            | relatedTarget | 返回与事件的目标节点相关的节点。 | 2 |
            | screenX | 返回当某个事件被触发时，鼠标指针的水平坐标。 | 2 |
            | screenY | 返回当某个事件被触发时，鼠标指针的垂直坐标。 | 2 |
            | shiftKey | 返回当事件被触发时，"SHIFT" 键是否被按下。 | 2 |            

        - 方法
        
            | 方法 | 描述 | DOM |
            |------|------|------|
            | initMouseEvent() | 初始化鼠标事件对象的值 | 2 |
            | initKeyboardEvent() | 初始化键盘事件对象的值 | 3 |
            


















