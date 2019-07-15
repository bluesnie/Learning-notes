###### datetime:2019/7/15 10:09
###### author:nzb

# Vue.js

## Vue.js 是什么

Vue (读音 /vjuː/，类似于 view) 是一套用于构建用户界面的渐进式框架。与其它大型框架不同的是，Vue 被设计为可以自底向上逐层应用。
Vue 的核心库只关注视图层，不仅易于上手，还便于与第三方库或既有项目整合。另一方面，当与现代化的工具链以及各种支持类库结合使用时，
Vue 也完全能够为复杂的单页应用提供驱动。

- 简单示例
    ```html
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Vue基础</title>
            <!--1.导入Vue包-->
            <script src="../static/vue.js"></script>
        </head>
        <body>
            <!--将来new的Vue实例会控制这个元素的所有内容-->
            <!--这个需要就是MVVM中V-->
            <div id="app">
                <p>{{ msg }}</p>
            </div>
            <script>
                // 2.创建一个Vue示例
                // 当我们导入包后，浏览器的内存中就多了一个Vue的构造函数
                // 这个new出来的vm对象就是MVVM中的VM调度者
                var vm = new Vue({
                    el: '#app',   // 表示element, 网页上需要控制的区域
                    // 这里的data就是MVVM中的M，用于保存页面的数据
                    data: {       // 存储需要的数据。
                        msg: "Hello world, I'm Vue.js!!!"   // 通过Vue指令，把数据渲染到页面，不需要再操作DOM元素。
                        // (前端Vue之类的框架，不提倡我们去手动操作DOM元素了)
                    }
                })
            </script>
        </body>
        </html>
    ```

## 模板语法

Vue.js 使用了基于 HTML 的模板语法，允许开发者声明式地将 DOM 绑定至底层 Vue 实例的数据。所有 Vue.js 的模板都是合法的 HTML ，所以能被遵循规范的浏览器和 HTML 解析器解析。

在底层的实现上，Vue 将模板编译成虚拟 DOM 渲染函数。结合响应系统，Vue 能够智能地计算出最少需要重新渲染多少组件，并把 DOM 操作次数减到最少。

如果你熟悉虚拟 DOM 并且偏爱 JavaScript 的原始力量，你也可以不用模板，[直接写渲染 (render) 函数](https://cn.vuejs.org/v2/guide/render-function.html)，使用可选的 JSX 语法。

### 插值

- 文本

    数据绑定最常见的形式就是使用“Mustache”语法 (双大括号) 的文本插值：
    
        <span>Message: {{ msg }}</span>
    Mustache 标签将会被替代为对应数据对象上 msg 属性的值。无论何时，绑定的数据对象上 msg 属性发生了改变，插值处的内容都会更新。

    通过使用 v-once 指令，你也能执行一次性地插值，当数据改变时，插值处的内容不会更新。但请留心这会影响到该节点上的其它数据绑定：
        
        <span v-once>这个将不会改变: {{ msg }}</span>

- 原始HTML

    双大括号会将数据解释为普通文本，而非 HTML 代码。为了输出真正的 HTML，你需要使用 v-html 指令：
        
        <p>Using mustaches: {{ rawHtml }}</p>
        <p>Using v-html directive: <span v-html="rawHtml"></span></p>
        
    这个 span 的内容将会被替换成为属性值 rawHtml，直接作为 HTML——会忽略解析属性值中的数据绑定。注意，你不能使用 v-html 来复合局部模板，
    因为 Vue 不是基于字符串的模板引擎。反之，对于用户界面 (UI)，组件更适合作为可重用和可组合的基本单位。

    **注意**：你的站点上动态渲染的任意 HTML 可能会非常危险，因为它很容易导致 XSS 攻击。请只对可信内容使用 HTML 插值，绝不要对用户提供的内容使用插值。

- 特性

    Mustache 语法不能作用在 HTML 特性上，遇到这种情况应该使用 v-bind 指令：
    
        <div v-bind:id="dynamicId"></div>
    对于布尔特性 (它们只要存在就意味着值为 true)，v-bind 工作起来略有不同，在这个例子中：
    
        <button v-bind:disabled="isButtonDisabled">Button</button>
    如果 isButtonDisabled 的值是 null、undefined 或 false，则 disabled 特性甚至不会被包含在渲染出来的 <button> 元素中。

-  使用 JavaScript 表达式

    迄今为止，在我们的模板中，我们一直都只绑定简单的属性键值。但实际上，对于所有的数据绑定，Vue.js 都提供了完全的 JavaScript 表达式支持。

        {{ number + 1 }}
    
        {{ ok ? 'YES' : 'NO' }}
        
        {{ message.split('').reverse().join('') }}
        
        <div v-bind:id="'list-' + id"></div>
    这些表达式会在所属 Vue 实例的数据作用域下作为 JavaScript 被解析。有个限制就是，每个绑定都只能包含单个表达式，所以下面的例子都不会生效。
    
        <!-- 这是语句，不是表达式 -->
        {{ var a = 1 }}
        
        <!-- 流控制也不会生效，请使用三元表达式 -->
        {{ if (ok) { return message } }}
    **注意**：模板表达式都被放在沙盒中，只能访问全局变量的一个白名单，如 Math 和 Date 。你不应该在模板表达式中试图访问用户定义的全局变量。

### 指令

指令 (Directives) 是带有 v- 前缀的特殊特性。指令特性的值预期是单个 JavaScript 表达式 (v-for 是例外情况，稍后我们再讨论)。指令的职责是，当表达式的值改变时，将其产生的连带影响，响应式地作用于 DOM。回顾我们在介绍中看到的例子：

    <p v-if="seen">现在你看到我了</p>
这里，v-if 指令将根据表达式 seen 的值的真假来插入/移除 <p> 元素。

- 参数

    一些指令能够接收一个“参数”，在指令名称之后以冒号表示。例如，v-bind 指令可以用于响应式地更新 HTML 特性：
        
        <a v-bind:href="url">...</a>
    在这里 href 是参数，告知 v-bind 指令将该元素的 href 特性与表达式 url 的值绑定。
    
    另一个例子是 v-on 指令，它用于监听 DOM 事件：
    
        <a v-on:click="doSomething">...</a>
    在这里参数是监听的事件名。我们也会更详细地讨论事件处理。

- 动态参数

    从 2.6.0 开始，可以用方括号括起来的 JavaScript 表达式作为一个指令的参数：
    
        <a v-bind:[attributeName]="url"> ... </a>
    这里的 attributeName 会被作为一个 JavaScript 表达式进行动态求值，求得的值将会作为最终的参数来使用。例如，如果你的 Vue 实例有一个 data 属性 attributeName，其值为 "href"，那么这个绑定将等价于 v-bind:href。

    同样地，你可以使用动态参数为一个动态的事件名绑定处理函数：
        
        <a v-on:[eventName]="doSomething"> ... </a>
    同样地，当 eventName 的值为 "focus" 时，v-on:[eventName] 将等价于 v-on:focus。

    - 对动态参数的值的约束
    
    **注意**：动态参数表达式有一些语法约束，因为某些字符，例如空格和引号，放在 HTML 特性名里是无效的。同样，在 DOM 中使用模板时你需要回避大写键名。
    
    例如，下面的代码是无效的：
    
        <!-- 这会触发一个编译警告 -->
        <a v-bind:['foo' + bar]="value"> ... </a>
    变通的办法是使用没有空格或引号的表达式，或用计算属性替代这种复杂表达式。

    另外，如果你在 DOM 中使用模板 (直接在一个 HTML 文件里撰写模板)，需要留意浏览器会把特性名全部强制转为小写：

        <!-- 在 DOM 中使用模板时这段代码会被转换为 `v-bind:[someattr]` -->
        <a v-bind:[someAttr]="value"> ... </a>
 
- 修饰符

    修饰符 (modifier) 是以半角句号 . 指明的特殊后缀，用于指出一个指令应该以特殊方式绑定。例如，.prevent 修饰符告诉 v-on 指令对于触发的事件调用 event.preventDefault()：
    
        <form v-on:submit.prevent="onSubmit">...</form>
    - 事件修饰符：
        - .stop 阻止冒泡
        
        - .prevent 阻止默认事件
        
        - .capture 添加事件侦听器时使用事件捕获模式
        
        - .self 只当事件在该元素本身（比如不是子元素）触发时触发回调
        
        - .once 事件只触发一次
    
    - 例子
        ```html
            <div id='app'>
                <div class="inner" @click.self="divclick"> <!-- .self只有点击当前元素时才会触发函数，自己不会被冒泡影响但是不会阻止其他元素的冒泡-->
                    <input type="button" value="阻止冒泡" @click.stop="btnclick">  <!-- .stop阻止冒泡 -->
                    <input type="button" value="捕获" @click.capture="btnclick">  <!-- .capture触发捕获 -->
                    <a href="http://www.baidu.com" @click.prevent.stop.once="noredirect">百度一下</a>  <!-- .prevent阻止默认行为 .once只触发一次函数-->
                </div>
        
            </div>
        
            <script>
                var vm = new Vue({
                    el: '#app',
                    data: {},
                    methods: {
                        divclick(){
                            console.log('div');
                        },
                        btnclick(){
                            console.log('btn')
                        },
                        noredirect(){
                            console.log('a')
                        }
                    },
                });
            </script>
        ```
        
- 归纳常见指令

    - v-cloak：解决闪烁问题
    
        当网速较慢时，vue加载较慢，插值表达式渲染的数据就会产生闪烁，使用v-cloak指令加上如上样式可以解决该问题
    
    - v-text：插入文本字符串
    
        v-text默认就没有闪烁，但是会以文本字符串的方式覆盖所在元素的文本
        
    - v-html：插入html
    
        v-html与v-text的不同在于插入的是html。
    
    - v-bind：绑定属性
    
        在vue中想要让属性如title等于data对象中的一个变量（键），会被直接当作字符串显示出来，这时就要用到v-bind，v-bind可以将绑定的属性值当作变量对待，在data对象中去找。
        
        - v-bind的三种用法：
            - 直接使用指令v-bind
            - 使用简化指令":"
            - 在绑定的时候，拼接绑定内容：:title="btnTitle + ', 这是追加的内容'"
        
    - v-on：绑定方法,缩写："@"
    
    - 示例
    
        ```html
            <div id="app">
                <!--使用v-cloak能够解决插值表达式闪烁的问题-->
                <p v-cloak>+++++{{ msg }}-----</p>
                <h3 v-text="msg">======</h3>
                <!--默认v-text是没有闪烁问题的-->
                <!--v-text会覆盖元素中原本的内容，但是插值表达式只会替换自己的这个占位符，不会吧整个元素的内容清空-->
                <div v-html="msg2">123123</div>
                <!--v-bind是vue提供的用于绑定属性的指令-->
                <input type="button" value="按钮" v-bind:title="mytitle + '123'">
                <!--vue中提供了v-on事件绑定机制-->
                <input type="button" value="按钮" :title="mytitle + '123'" v-on:click="show">
                <input type="button" value="按钮" :title="mytitle + '123'" v-on:mouseover="show">
            </div>
            <script src="../static/vue.js"></script>
            <script>
                var vm = new Vue({
                    el: '#app',
                    data: {
                        msg: 'hello',
                        msg2: '<h1>我是h1</h1>',
                        mytitle: '这是一个自定义title',
                    },
                    methods: {       // 这个methods属性中定义了当前vue实例所有可用的方法
                        show: function () {
                            alert("Hello!!!")
                        }
                    }
                })
            </script>
        ```
        
    - v-model：、实现双向数据绑定：
        ```html
            <div id="app">
                <p>{{ message }}</p>
                <input v-model="message">
            </div>
                
            <script>
            new Vue({
              el: '#app',
              data: {
                message: 'Runoob!'
              }
            })
            </script>
        ```
        **注意：**v-model 指令用来在 input、select、textarea、checkbox、radio 等表单控件元素上创建双向数据绑定，根据表单上的值，自动更新绑定的元素的值。

### 缩写

v- 前缀作为一种视觉提示，用来识别模板中 Vue 特定的特性。当你在使用 Vue.js 为现有标签添加动态行为 (dynamic behavior) 时，
v- 前缀很有帮助，然而，对于一些频繁用到的指令来说，就会感到使用繁琐。
同时，在构建由 Vue 管理所有模板的单页面应用程序 (SPA - single page application) 时，v- 前缀也变得没那么重要了。
因此，Vue 为 v-bind 和 v-on 这两个最常用的指令，提供了特定简写：

- v-bind 缩写
    ```html
        <!-- 完整语法 -->
        <a v-bind:href="url">...</a>
        
        <!-- 缩写 -->
        <a :href="url">...</a>
    ```

- v-on 缩写
    ```html
        <!-- 完整语法 -->
        <a v-on:click="doSomething">...</a>
        
        <!-- 缩写 -->
        <a @click="doSomething">...</a>
    ```
它们看起来可能与普通的 HTML 略有不同，但 : 与 @ 对于特性名来说都是合法字符，在所有支持 Vue 的浏览器都能被正确地解析。而且，它们不会出现在最终渲染的标记中。缩写语法是完全可选的，但随着你更深入地了解它们的作用，你会庆幸拥有它们。

### 跑马灯
```html
    <div id="app">
        <input type="button" value="浪起来" @click="lang">
        <input type="button" value="别浪" @click="stop">
        <h4>{{ msg }}</h4>
    </div>

    <script>
        var vm = new Vue({
            el: "#app",
            data: {
                msg: "猥琐发育，别浪~~~",
                intervalId: null, //计时器ID
            },
            methods: {
                lang(){
                    // 第一种：构造函数保证函数里面的this相同
                    if(this.intervalId != null) return; // 如果已开启返回
                    this.intervalId = setInterval(() => {
                        // console.log(this.msg);
                        var start = this.msg.substring(0, 1);
                        var end = this.msg.substring(1);
                        this.msg = end + start;
                    }, 400);

                    // 第二种：声明一个变量保存this
                    // var _this = this;
                    // setInterval(function () {
                    //     // console.log(this.msg);
                    //     var start = _this.msg.substring(0, 1);
                    //     var end = _this.msg.substring(1);
                    //     _this.msg = end + start;
                    // }, 400);
                },
                stop(){
                    clearInterval(this.intervalId);  // 清除定时器
                    this.intervalId = null;  // 每当清除定时器ID置为null
                }
            },
        })
    </script>
```

## 在Vue中使用样式

### 使用class样式

- 数组
    ```
    <h1 :class="['red', 'thin']">这是一个邪恶的H1</h1>
    ```

- 数组中使用三元表达式
    ```
    <h1 :class="['red', 'thin', isactive?'active':'']">这是一个邪恶的H1</h1>
    ```

-  数组中嵌套对象(对象就是键值对)
    ```
    <h1 :class="['red', 'thin', {'active': isactive}]">这是一个邪恶的H1</h1>
    ```

- 直接使用对象
    ```
    <h1 :class="classObj">这是一个邪恶的H1</h1>
    
    <script>
        var vm = new Vue({
            el: '#app',
            data: {
                flag: true,
                classObj: {red:true, thin:true, italic:true, active:false},

            },
            methods: {},
        });
    </script>
    ```

### 使用内联样式

- 直接在元素上通过 `:style` 的形式，书写样式对象
    ```
    <h1 :style="{color: 'red', 'font-size': '40px'}">这是一个善良的H1</h1>
    ```

- 将样式对象，定义到 `data` 中，并直接引用到 `:style` 中
    - 在data上定义样式：
        ```
        data: {
                h1StyleObj: { color: 'red', 'font-size': '40px', 'font-weight': '200' }
        }
        ```
    - 在元素中，通过属性绑定的形式，将样式对象应用到元素中：
        ```
        <h1 :style="h1StyleObj">这是一个善良的H1</h1>
        ```

- 在 `:style` 中通过数组，引用多个 `data` 上的样式对象
    - 在data上定义样式：
        ```
        data: {
                h1StyleObj: { color: 'red', 'font-size': '40px', 'font-weight': '200' },
                h1StyleObj2: { fontStyle: 'italic' }
        }
        ```
    - 在元素中，通过属性绑定的形式，将样式对象应用到元素中：
        ```
        <h1 :style="[h1StyleObj, h1StyleObj2]">这是一个善良的H1</h1>
        ```







