###### datetime:2019/7/10 15:29
###### author:nzb

## [jQuery 选择器](https://www.runoob.com/jquery/jquery-ref-selectors.html)

| 选择器 | 实例 | 选取 |
|----------|-----------|-------------|
| * | $("*") | 所有元素 |
| #id | $("#lastname") | id="lastname" 的元素 |
| .class | $(".intro") | class="intro" 的所有元素 |
| .class,.class | $(".intro,.demo") | class 为 "intro" 或 "demo" 的所有元素 |
| element | $("p") | 所有 `<p>` 元素 |
| el1,el2,el3 | $("h1,div,p") | 所有 `<h1>`、`<div>` 和 `<p>` 元素 |
| :first | $("p:first") | 第一个 `<p>` 元素 |
| :last | $("p:last") | 最后一个 `<p>` 元素 |
| :even | $("tr:even") | 所有偶数 `<tr>` 元素，索引值从 0 开始，第一个元素是偶数 (0)，第二个元素是奇数 (1)，以此类推。 |
| :odd | $("tr:odd") | 所有奇数 `<tr>` 元素，索引值从 0 开始，第一个元素是偶数 (0)，第二个元素是奇数 (1)，以此类推。 |
| :first-child | $("p:first-child") | 属于其父元素的第一个子元素的所有 `<p>` 元素 |
| :first-of-type | $("p:first-of-type") | 属于其父元素的第一个 `<p>` 元素的所有 `<p>` 元素 |
| :last-child | $("p:last-child") | 属于其父元素的最后一个子元素的所有 `<p>` 元素 |
| :last-of-type | $("p:last-of-type") | 属于其父元素的最后一个 `<p>` 元素的所有 `<p>` 元素 |
| :nth-child(n) | $("p:nth-child(2)") | 属于其父元素的第二个子元素的所有 `<p>` 元素 |
| :nth-last-child(n) | $("p:nth-last-child(2)") | 属于其父元素的第二个子元素的所有 `<p>` 元素，从最后一个子元素开始计数 |
| :nth-of-type(n) | $("p:nth-of-type(2)") | 属于其父元素的第二个 `<p>` 元素的所有 `<p>` 元素 |
| :nth-last-of-type(n) | $("p:nth-last-of-type(2)") | 属于其父元素的第二个 `<p>` 元素的所有 `<p>` 元素，从最后一个子元素开始计数 |
| :only-child | $("p:only-child") | 属于其父元素的唯一子元素的所有 `<p>` 元素 |
| :only-of-type | $("p:only-of-type") | 属于其父元素的特定类型的唯一子元素的所有 `<p>` 元素 |
| parent > child | $("div > p") | `<div>` 元素的直接子元素的所有 `<p>` 元素 |
| parent descendant | $("div p") | `<div>` 元素的后代的所有 `<p>` 元素 |
| element + next | $("div + p") | 每个 `<div>` 元素相邻的下一个 `<p>` 元素 |
| element ~ siblings | $("div ~ p") | `<div>` 元素同级的所有 `<p>` 元素 |
| :eq(index) | $("ul li:eq(3)") | 列表中的第四个元素（index 值从 0 开始） |
| :gt(no) | $("ul li:gt(3)") | 列举 index 大于 3 的元素 |
| :lt(no) | $("ul li:lt(3)") | 列举 index 小于 3 的元素 |
| :not(selector) | $("input:not(:empty)") | 所有不为空的输入元素 |
| :header | $(":header") | 所有标题元素 `<h1>`, `<h2>` ... |
| :animated | $(":animated") | 所有动画元素 |
| :focus | $(":focus") | 当前具有焦点的元素 |
| :contains(text) | $(":contains('Hello')") | 所有包含文本 "Hello" 的元素 |
| :has(selector) | $("div:has(p)") | 所有包含有 `<p>` 元素在其内的 `<div>` 元素 |
| :empty | $(":empty") | 所有空元素 |
| :parent | $(":parent") | 匹配所有含有子元素或者文本的父元素。 |
| :hidden | $("p:hidden") | 所有隐藏的 `<p>` 元素 |
| :visible | $("table:visible") | 所有可见的表格 |
| :root | $(":root") | 文档的根元素 |
| :lang(language) | $("p:lang(de)") | 所有 lang 属性值为 "de" 的 `<p>` 元素 |
|   |   |   |
| [attribute] | $("[href]") | 所有带有 href 属性的元素 |
| [attribute=value] | $("[href='default.htm']") | 所有带有 href 属性且值等于 "default.htm" 的元素 |
| [attribute!=value] | $("[href!='default.htm']") | 所有带有 href 属性且值不等于 "default.htm" 的元素 |
| [attribute$=value] | $("[href$='.jpg']") | 所有带有 href 属性且值以 ".jpg" 结尾的元素 |
| [attribute|=value] | $("[title|='Tomorrow']") | 所有带有 title 属性且值等于 'Tomorrow' 或者以 'Tomorrow' 后跟连接符作为开头的字符串 |
| [attribute^=value] | $("[title^='Tom']") | 所有带有 title 属性且值以 "Tom" 开头的元素 |
| [attribute~=value] | $("[title~='hello']") | 所有带有 title 属性且值包含单词 "hello" 的元素 |
| [attribute*=value] | $("[title*='hello']") | 所有带有 title 属性且值包含字符串 "hello" 的元素 |
| [name=value][name2=value2] | $( "input[id][name$='man']" ) | 带有 id 属性，并且 name 属性以 man 结尾的输入框 |
|   |   |   |
| :input | $(":input") | 所有 input 元素 |
| :text | $(":text") | 所有带有 type="text" 的 input 元素 |
| :password | $(":password") | 所有带有 type="password" 的 input 元素 |
| :radio | $(":radio") | 所有带有 type="radio" 的 input 元素 |
| :checkbox | $(":checkbox") | 所有带有 type="checkbox" 的 input 元素 |
| :submit | $(":submit") | 所有带有 type="submit" 的 input 元素 |
| :reset | $(":reset") | 所有带有 type="reset" 的 input 元素 |
| :button | $(":button") | 所有带有 type="button" 的 input 元素 |
| :image | $(":image") | 所有带有 type="image" 的 input 元素 |
| :file | $(":file") | 所有带有 type="file" 的 input 元素 |
| :enabled | $(":enabled") | 所有启用的元素 |
| :disabled | $(":disabled") | 所有禁用的元素 |
| :selected | $(":selected") | 所有选定的下拉列表元素 |
| :checked | $(":checked") | 所有选中的复选框选项 |
| .selector | $(selector).selector | 在jQuery 1.7中已经不被赞成使用。返回传给jQuery()的原始选择器 |
| :target | $( "p:target" ) | 选择器将选中ID和URI中一个格式化的标识符相匹配的`<p>`元素 |

## [jQuery 事件方法](https://www.runoob.com/jquery/jquery-ref-events.html)

| 方法 | 描述 |
|----------|-----------|
| bind() | 向元素添加事件处理程序 |
| blur() | 添加/触发失去焦点事件 |
| change() | 添加/触发 change 事件 |
| click() | 添加/触发 click 事件 |
| dblclick() | 添加/触发 double click 事件 |
| delegate() | 向匹配元素的当前或未来的子元素添加处理程序 |
| die() | 在版本 1.9 中被移除。移除所有通过 live() 方法添加的事件处理程序 |
| error() | 在版本 1.8 中被废弃。添加/触发 error 事件 |
| event.currentTarget | 在事件冒泡阶段内的当前 DOM 元素 |
| event.data | 包含当前执行的处理程序被绑定时传递到事件方法的可选数据 |
| event.delegateTarget | 返回当前调用的 jQuery 事件处理程序所添加的元素 |
| event.isDefaultPrevented() | 返回指定的 event 对象上是否调用了 event.preventDefault() |
| event.isImmediatePropagationStopped() | 返回指定的 event 对象上是否调用了 event.stopImmediatePropagation() |
| event.isPropagationStopped() | 返回指定的 event 对象上是否调用了 event.stopPropagation() |
| event.namespace | 返回当事件被触发时指定的命名空间 |
| event.pageX | 返回相对于文档左边缘的鼠标位置 |
| event.pageY | 返回相对于文档上边缘的鼠标位置 |
| event.preventDefault() | 阻止事件的默认行为 |
| event.relatedTarget | 返回当鼠标移动时哪个元素进入或退出 |
| event.result | 包含由被指定事件触发的事件处理程序返回的最后一个值 |
| event.stopImmediatePropagation() | 阻止其他事件处理程序被调用 |
| event.stopPropagation() | 阻止事件向上冒泡到 DOM 树，阻止任何父处理程序被事件通知 |
| event.target | 返回哪个 DOM 元素触发事件 |
| event.timeStamp | 返回从 1970 年 1 月 1 日到事件被触发时的毫秒数 |
| event.type | 返回哪种事件类型被触发 |
| event.which | 返回指定事件上哪个键盘键或鼠标按钮被按下 |
| event.metaKey | 事件触发时 META 键是否被按下 |
| focus() | 添加/触发 focus 事件 |
| focusin() | 添加事件处理程序到 focusin 事件 |
| focusout() | 添加事件处理程序到 focusout 事件 |
| hover() | 添加两个事件处理程序到 hover 事件 |
| keydown() | 添加/触发 keydown 事件 |
| keypress() | 添加/触发 keypress 事件 |
| keyup() | 添加/触发 keyup 事件 |
| live() | 在版本 1.9 中被移除。添加一个或多个事件处理程序到当前或未来的被选元素 |
| load() | 在版本 1.8 中被废弃。添加一个事件处理程序到 load 事件 |
| mousedown() | 添加/触发 mousedown 事件 |
| mouseenter() | 添加/触发 mouseenter 事件 |
| mouseleave() | 添加/触发 mouseleave 事件 |
| mousemove() | 添加/触发 mousemove 事件 |
| mouseout() | 添加/触发 mouseout 事件 |
| mouseover() | 添加/触发 mouseover 事件 |
| mouseup() | 添加/触发 mouseup 事件 |
| off() | 移除通过 on() 方法添加的事件处理程序 |
| on() | 向元素添加事件处理程序 |
| one() | 向被选元素添加一个或多个事件处理程序。该处理程序只能被每个元素触发一次 |
| $.proxy() | 接受一个已有的函数，并返回一个带特定上下文的新的函数 |
| ready() | 规定当 DOM 完全加载时要执行的函数 |
| resize() | 添加/触发 resize 事件 |
| scroll() | 添加/触发 scroll 事件 |
| select() | 添加/触发 select 事件 |
| submit() | 添加/触发 submit 事件 |
| toggle() | 在版本 1.9 中被移除。添加 click 事件之间要切换的两个或多个函数 |
| trigger() | 触发绑定到被选元素的所有事件 |
| triggerHandler() | 触发绑定到被选元素的指定事件上的所有函数 |
| unbind() | 从被选元素上移除添加的事件处理程序 |
| undelegate() | 从现在或未来的被选元素上移除事件处理程序 |
| unload() | 在版本 1.8 中被废弃。添加事件处理程序到 unload 事件 |
| contextmenu() | 添加事件处理程序到 contextmenu 事件 |
| $.holdReady() | 用于暂停或恢复.ready() 事件的执行 |

## [jQuery 效果方法](https://www.runoob.com/jquery/jquery-ref-effects.html)

| 方法 | 描述 |
|----------|-----------|
| animate() | 对被选元素应用"自定义"的动画 |
| clearQueue() | 对被选元素移除所有排队函数（仍未运行的） |
| delay() | 对被选元素的所有排队函数（仍未运行）设置延迟 |
| dequeue() | 移除下一个排队函数，然后执行函数 |
| fadeIn() | 逐渐改变被选元素的不透明度，从隐藏到可见 |
| fadeOut() | 逐渐改变被选元素的不透明度，从可见到隐藏 |
| fadeTo() | 把被选元素逐渐改变至给定的不透明度 |
| fadeToggle() | 在 fadeIn() 和 fadeOut() 方法之间进行切换 |
| finish() | 对被选元素停止、移除并完成所有排队动画 |
| hide() | 隐藏被选元素 |
| queue() | 显示被选元素的排队函数 |
| show() | 显示被选元素 |
| slideDown() | 通过调整高度来滑动显示被选元素 |
| slideToggle() | slideUp() 和 slideDown() 方法之间的切换 |
| slideUp() | 通过调整高度来滑动隐藏被选元素 |
| stop() | 停止被选元素上当前正在运行的动画 |
| toggle() | hide() 和 show() 方法之间的切换 |

## [jQuery HTML / CSS 方法](https://www.runoob.com/jquery/jquery-ref-html.html)

| 方法 | 描述 |
|----------|-----------|
| addClass() | 向被选元素添加一个或多个类名 |
| after() | 在被选元素后插入内容 |
| append() | 在被选元素的结尾插入内容 |
| appendTo() | 在被选元素的结尾插入 HTML 元素 |
| attr() | 设置或返回被选元素的属性/值 |
| before() | 在被选元素前插入内容 |
| clone() | 生成被选元素的副本 |
| css() | 为被选元素设置或返回一个或多个样式属性 |
| detach() | 移除被选元素（保留数据和事件） |
| empty() | 从被选元素移除所有子节点和内容 |
| hasClass() | 检查被选元素是否包含指定的 class 名称 |
| height() | 设置或返回被选元素的高度 |
| html() | 设置或返回被选元素的内容 |
| innerHeight() | 返回元素的高度（包含 padding，不包含 border） |
| innerWidth() | 返回元素的宽度（包含 padding，不包含 border） |
| insertAfter() | 在被选元素后插入 HTML 元素 |
| insertBefore() | 在被选元素前插入 HTML 元素 |
| offset() | 设置或返回被选元素的偏移坐标（相对于文档） |
| offsetParent() | 返回第一个定位的祖先元素 |
| outerHeight() | 返回元素的高度（包含 padding 和 border） |
| outerWidth() | 返回元素的宽度（包含 padding 和 border） |
| position() | 返回元素的位置（相对于父元素） |
| prepend() | 在被选元素的开头插入内容 |
| prependTo() | 在被选元素的开头插入 HTML 元素 |
| prop() | 设置或返回被选元素的属性/值 |
| remove() | 移除被选元素（包含数据和事件） |
| removeAttr() | 从被选元素移除一个或多个属性 |
| removeClass() | 从被选元素移除一个或多个类 |
| removeProp() | 移除通过 prop() 方法设置的属性 |
| replaceAll() | 把被选元素替换为新的 HTML 元素 |
| replaceWith() | 把被选元素替换为新的内容 |
| scrollLeft() | 设置或返回被选元素的水平滚动条位置 |
| scrollTop() | 设置或返回被选元素的垂直滚动条位置 |
| text() | 设置或返回被选元素的文本内容 |
| toggleClass() | 在被选元素中添加/移除一个或多个类之间切换 |
| unwrap() | 移除被选元素的父元素 |
| val() | 设置或返回被选元素的属性值（针对表单元素） |
| width() | 设置或返回被选元素的宽度 |
| wrap() | 在每个被选元素的周围用 HTML 元素包裹起来 |
| wrapAll() | 在所有被选元素的周围用 HTML 元素包裹起来 |
| wrapInner() | 在每个被选元素的内容周围用 HTML 元素包裹起来 |
| $.escapeSelector() | 转义CSS选择器中有特殊意义的字符或字符串 |
| $.cssHooks | 提供了一种方法通过定义函数来获取和设置特定的CSS值 |

## [jQuery 遍历方法](https://www.runoob.com/jquery/jquery-ref-traversing.html)

| 方法 | 描述 |
|----------|-----------|
| add() | 把元素添加到匹配元素的集合中 |
| addBack() | 把之前的元素集添加到当前集合中 |
| andSelf() | 在版本 1.8 中被废弃。addBack() 的别名 |
| children() | 返回被选元素的所有直接子元素 |
| closest() | 返回被选元素的第一个祖先元素 |
| contents() | 返回被选元素的所有直接子元素（包含文本和注释节点） |
| each() | 为每个匹配元素执行函数 |
| end() | 结束当前链中最近的一次筛选操作，并把匹配元素集合返回到前一次的状态 |
| eq() | 返回带有被选元素的指定索引号的元素 |
| filter() | 把匹配元素集合缩减为匹配选择器或匹配函数返回值的新元素 |
| find() | 返回被选元素的后代元素 |
| first() | 返回被选元素的第一个元素 |
| has() | 返回拥有一个或多个元素在其内的所有元素 |
| is() | 根据选择器/元素/jQuery 对象检查匹配元素集合，如果存在至少一个匹配元素，则返回 true |
| last() | 返回被选元素的最后一个元素 |
| map() | 把当前匹配集合中的每个元素传递给函数，产生包含返回值的新 jQuery 对象 |
| next() | 返回被选元素的后一个同级元素 |
| nextAll() | 返回被选元素之后的所有同级元素 |
| nextUntil() | 返回介于两个给定参数之间的每个元素之后的所有同级元素 |
| not() | 从匹配元素集合中移除元素 |
| offsetParent() | 返回第一个定位的父元素 |
| parent() | 返回被选元素的直接父元素 |
| parents() | 返回被选元素的所有祖先元素 |
| parentsUntil() | 返回介于两个给定参数之间的所有祖先元素 |
| prev() | 返回被选元素的前一个同级元素 |
| prevAll() | 返回被选元素之前的所有同级元素 |
| prevUntil() | 返回介于两个给定参数之间的每个元素之前的所有同级元素 |
| siblings() | 返回被选元素的所有同级元素 |
| slice() | 把匹配元素集合缩减为指定范围的子集 |

## [jQuery AJAX 方法](https://www.runoob.com/jquery/jquery-ref-ajax.html)

| 方法 | 描述 |
|----------|-----------|
| $.ajax() | 执行异步 AJAX 请求 |
| $.ajaxPrefilter() | 在每个请求发送之前且被 $.ajax() 处理之前，处理自定义 Ajax 选项或修改已存在选项 |
| $.ajaxSetup() | 为将来的 AJAX 请求设置默认值 |
| $.ajaxTransport() | 创建处理 Ajax 数据实际传送的对象 |
| $.get() | 使用 AJAX 的 HTTP GET 请求从服务器加载数据 |
| $.getJSON() | 使用 HTTP GET 请求从服务器加载 JSON 编码的数据 |
| $.getScript() | 使用 AJAX 的 HTTP GET 请求从服务器加载并执行 JavaScript |
| $.param() | 创建数组或对象的序列化表示形式（可用于 AJAX 请求的 URL 查询字符串） |
| $.post() | 使用 AJAX 的 HTTP POST 请求从服务器加载数据 |
| ajaxComplete() | 规定 AJAX 请求完成时运行的函数 |
| ajaxError() | 规定 AJAX 请求失败时运行的函数 |
| ajaxSend() | 规定 AJAX 请求发送之前运行的函数 |
| ajaxStart() | 规定第一个 AJAX 请求开始时运行的函数 |
| ajaxStop() | 规定所有的 AJAX 请求完成时运行的函数 |
| ajaxSuccess() | 规定 AJAX 请求成功完成时运行的函数 |
| load() | 从服务器加载数据，并把返回的数据放置到指定的元素中 |
| serialize() | 编码表单元素集为字符串以便提交 |
| serializeArray() | 编码表单元素集为 names 和 values 的数组 |

## [jQuery 杂项方法](https://www.runoob.com/jquery/jquery-ref-misc.html)

### jQuery 杂项方法

| 方法 | 描述 |
|----------|-----------|
| data() | 向被选元素附加数据，或者从被选元素获取数据 |
| each() | 为每个匹配元素执行函数 |
| get() | 获取由选择器指定的 DOM 元素 |
| index() | 从匹配元素中搜索给定元素 |
| $.noConflict() | 释放变量 $ 的 jQuery 控制权 |
| $.param() | 创建数组或对象的序列化表示形式（可在生成 AJAX 请求时用于 URL 查询字符串中） |
| removeData() | 移除之前存放的数据 |
| size() | 在版本 1.8 中被废弃。返回被 jQuery 选择器匹配的 DOM 元素的数量 |
| toArray() | 以数组的形式检索所有包含在 jQuery 集合中的所有 DOM 元素 |
| pushStack() | 将一个DOM元素集合加入到jQuery栈 |
| $.when() | 提供一种方法来执行一个或多个对象的回调函数 |

### jQuery 实用工具

| 方法 | 描述 |
|----------|-----------|
| $.boxModel | 在版本 1.8 中被废弃。检测浏览器是否使用W3C的CSS盒模型渲染当前页面 |
| $.browser | 在版本 1.9 中被废弃。返回用户当前使用的浏览器的相关信息 |
| $.contains() | 判断另一个DOM元素是否是指定DOM元素的后代 |
| $.each() | 遍历指定的对象和数组 |
| $.extend() | 将一个或多个对象的内容合并到目标对象 |
| $.fn.extend() | 为jQuery扩展一个或多个实例属性和方法 |
| $.globalEval() | 全局性地执行一段JavaScript代码 |
| $.grep() | 过滤并返回满足指定函数的数组元素 |
| $.inArray() | 在数组中查找指定值并返回它的索引值（如果没有找到，则返回-1） |
| $.isArray() | 判断指定参数是否是一个数组 |
| $.isEmptyObject() | 检查对象是否为空（不包含任何属性） |
| $.isFunction() | 判断指定参数是否是一个函数 |
| $.isNumeric() | 判断指定参数是否是一个数字值 |
| $.isPlainObject() | 判断指定参数是否是一个纯粹的对象 |
| $.isWindow() | 判断指定参数是否是一个窗口 |
| $.isXMLDoc() | 判断一个DOM节点是否位于XML文档中，或者其本身就是XML文档 |
| $.makeArray() | 将一个类似数组的对象转换为真正的数组对象 |
| $.map() | 指定函数处理数组中的每个元素(或对象的每个属性)，并将处理结果封装为新的数组返回 |
| $.merge() | 合并两个数组内容到第一个数组 |
| $.noop() | 一个空函数 |
| $.now() | 返回当前时间 |
| $.parseHTML() | 将HTML字符串解析为对应的DOM节点数组 |
| $.parseJSON() | 将符合标准格式的JSON字符串转为与之对应的JavaScript对象 |
| $.parseXML() | 将字符串解析为对应的XML文档 |
| $.trim() | 去除字符串两端的空白字符 |
| $.type() | 确定JavaScript内置对象的类型 |
| $.unique() | 在jQuery 3.0中被弃用。对DOM元素数组进行排序，并移除重复的元素 |
| $.uniqueSort() | 对DOM元素数组进行排序，并移除重复的元素 |
| $.data() | 在指定的元素上存取数据，并返回设置值 |
| $.hasData() | 确定一个元素是否有相关的jQuery数据 |
| $.sub() | 创建一个新的jQuery副本，其属性和方法可以修改，而不会影响原来的jQuery对象 |
| $.speed | 创建一个包含一组属性的对象用来定义自定义动画 |
| $.htmlPrefilter() | 通过jQuery操作方法修改和过滤HTML字符串 |
| $.readyException() | 处理包裹在jQuery()中函数同步抛出的错误 |

### jQuery 回调对象

| 方法 | 描述 |
|----------|-----------|
| $.Callbacks() | 一个多用途的回调列表对象，用来管理回调函数列表 |
| callbacks.add() | 在回调列表中添加一个回调或回调的集合 |
| callbacks.disable() | 禁用回调列表中的回调函数 |
| callbacks.disabled() | 确定回调列表是否已被禁用 |
| callbacks.empty() | 从列表中清空所有的回调 |
| callbacks.fire() | 传入指定的参数调用所有的回调 |
| callbacks.fired() | 确定回调是否至少已经调用一次 |
| callbacks.firewith() | 给定的上下文和参数访问列表中的所有回调 |
| callbacks.has() | 判断回调列表中是否添加过某回调函数 |
| callbacks.lock() | 锁定当前状态的回调列表 |
| callbacks.locked() | 判断回调列表是否被锁定 |
| callbacks.remove() | 从回调列表中的删除一个回调或回调集合 |

### jQuery 延迟对象

| 方法 | 描述 |
|----------|-----------|
| $.Deferred() | 返回一个链式实用对象方法来注册多个回调 |
| deferred.always() | 当Deferred（延迟）对象被受理或被拒绝时，调用添加的处理程序 |
| deferred.done() | 当Deferred（延迟）对象被受理时，调用添加的处理程序 |
| deferred.fail() | 当Deferred（延迟）对象被拒绝时，调用添加的处理程序 |
| deferred.isRejected() | 从jQuery1.7开始已经过时，确定 Deferred 对象是否已被拒绝 |
| deferred.isResolved() | 从jQuery1.7开始已经过时，确定 Deferred 对象是否已被解决 |
| deferred.notify() | 给定一个参数，调用正在延迟对象上进行的回调函数( progressCallbacks ) |
| deferred.notifyWith() | 给定上下文和参数，调用正在延迟对象上进行的回调函数( progressCallbacks ) |
| deferred.pipe() | 过滤 and/or 链式延迟对象的工具方法 |
| deferred.progress() | 当Deferred（延迟）对象生成进度通知时，调用添加处理程序 |
| deferred.promise() | 返回 Deferred(延迟)的 Promise 对象 |
| deferred.reject() | 拒绝 Deferred（延迟）对象，并根据给定的参数调用任何 failCallbacks 回调函数 |
| deferred.rejectWith() | 拒绝 Deferred（延迟）对象，并根据给定的 context 和 args 参数调用任何 failCallbacks 回调函数 |
| deferred.resolve() | 解决Deferred（延迟）对象，并根据给定的参数调用任何 doneCallbacks 回调函数 |
| deferred.resolveWith() | 解决Deferred（延迟）对象，并根据给定的context 和 args 参数调用任何 doneCallbacks 回调函数 |
| deferred.state() | 确定一个Deferred（延迟）对象的当前状态 |
| deferred.then() | 当Deferred（延迟）对象解决，拒绝或仍在进行中时，调用添加处理程序 |
| .promise() | 返回一个 Promise 对象，观察某种类型被绑定到集合的所有行动，是否已被加入到队列中 |

## jQuery 属性

| 方法 | 描述 |
|----------|-----------|
| context | 在版本 1.10 中被废弃。包含被传递到 jQuery 的原始上下文 |
| jquery | 包含 jQuery 的版本号 |
| jQuery.fx.interval | 改变以毫秒计的动画运行速率 |
| jQuery.fx.off | 对所有动画进行全局禁用或启用 |
| jQuery.support | 包含表示不同浏览器特性或漏洞的属性集（主要用于 jQuery 的内部使用） |
| length | 包含 jQuery 对象中元素的数目 |
| jQuery.cssNumber | 包含所有可以不使用单位的CSS属性的对象 |
