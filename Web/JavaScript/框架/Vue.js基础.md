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
                <p>{ { msg } }</p>
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

## [模板语法](https://cn.vuejs.org/v2/guide/syntax.html)

Vue.js 使用了基于 HTML 的模板语法，允许开发者声明式地将 DOM 绑定至底层 Vue 实例的数据。所有 Vue.js 的模板都是合法的 HTML ，所以能被遵循规范的浏览器和 HTML 解析器解析。

在底层的实现上，Vue 将模板编译成虚拟 DOM 渲染函数。结合响应系统，Vue 能够智能地计算出最少需要重新渲染多少组件，并把 DOM 操作次数减到最少。

如果你熟悉虚拟 DOM 并且偏爱 JavaScript 的原始力量，你也可以不用模板，[直接写渲染 (render) 函数](https://cn.vuejs.org/v2/guide/render-function.html)，使用可选的 JSX 语法。

### 插值

- 文本

    数据绑定最常见的形式就是使用“Mustache”语法 (双大括号) 的文本插值：
    
        <span>Message: { { msg } }</span>
    Mustache 标签将会被替代为对应数据对象上 msg 属性的值。无论何时，绑定的数据对象上 msg 属性发生了改变，插值处的内容都会更新。

    通过使用 v-once 指令，你也能执行一次性地插值，当数据改变时，插值处的内容不会更新。但请留心这会影响到该节点上的其它数据绑定：
        
        <span v-once>这个将不会改变: { { msg } }</span>

- 原始HTML

    双大括号会将数据解释为普通文本，而非 HTML 代码。为了输出真正的 HTML，你需要使用 v-html 指令：
        
        <p>Using mustaches: { { rawHtml } }</p>
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

        { { number + 1 } }
    
        { { ok ? 'YES' : 'NO' } }
        
        { { message.split('').reverse().join('') } }
        
        <div v-bind:id="'list-' + id"></div>
    这些表达式会在所属 Vue 实例的数据作用域下作为 JavaScript 被解析。有个限制就是，每个绑定都只能包含单个表达式，所以下面的例子都不会生效。
    
        <!-- 这是语句，不是表达式 -->
        { { var a = 1 } }
        
        <!-- 流控制也不会生效，请使用三元表达式 -->
        { { if (ok) { return message } } }
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
        - `.stop` 阻止冒泡
        
        - `.prevent` 阻止默认事件
        
        - `.capture` 添加事件侦听器时使用事件捕获模式
        
        - `.self` 只当事件在该元素本身（比如不是子元素）触发时触发回调
        
        - `.once` 事件只触发一次
    
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
    
    - 


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
                <p v-cloak>+++++{ { msg } }-----</p>
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
                <p>{ { message } }</p>
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

## [在Vue中使用样式](https://cn.vuejs.org/v2/guide/class-and-style.html)

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

## [条件渲染](https://cn.vuejs.org/v2/guide/conditional.html)

### v-if

- 条件判断使用 v-if 指令：
    ```html
        <div id="app">
            <p v-if="seen">现在你看到我了</p>
            <template v-if="ok">
              <h1>菜鸟教程</h1>
              <p>学的不仅是技术，更是梦想！</p>
              <p>哈哈哈，打字辛苦啊！！！</p>
            </template>
        </div>
            
        <script>
        new Vue({
          el: '#app',
          data: {
            seen: true,
            ok: true
          }
        })
        </script>
    ```

- 你可以使用 v-else 指令来表示 v-if 的“else 块”：
```html
    <div v-if="Math.random() > 0.5">
      Now you see me
    </div>
    <div v-else>
      Now you don't
    </div>
```

- v-else-if，顾名思义，充当 v-if 的“else-if 块”，可以连续使用：
```html
    <div v-if="type === 'A'">
      A
    </div>
    <div v-else-if="type === 'B'">
      B
    </div>
    <div v-else-if="type === 'C'">
      C
    </div>
    <div v-else>
      Not A/B/C
    </div>
```
**注意**：v-else，v-else-if 也必须紧跟在带 v-if 或者 v-else-if 的元素之后。

### v-show
    <h1 v-show="ok">Hello!</h1>
注意，v-show 不支持 <template> 元素，也不支持 v-else。

### v-if vs v-show

v-if 是“真正”的条件渲染，因为它会确保在切换过程中条件块内的事件监听器和子组件适当地被销毁和重建。

v-if 也是惰性的：如果在初始渲染时条件为假，则什么也不做——直到条件第一次变为真时，才会开始渲染条件块。

相比之下，v-show 就简单得多——不管初始条件是什么，元素总是会被渲染，并且只是简单地基于 CSS 进行切换。

一般来说，v-if 有更高的切换开销，而 v-show 有更高的初始渲染开销。因此，如果需要非常频繁地切换，
则使用 v-show 较好；如果在运行时条件很少改变，则使用 v-if 较好。

### v-if 与 v-for 一起使用

不推荐同时使用 v-if 和 v-for。请查阅[风格指南](https://cn.vuejs.org/v2/style-guide/#%E9%81%BF%E5%85%8D-v-if-%E5%92%8C-v-for-%E7%94%A8%E5%9C%A8%E4%B8%80%E8%B5%B7-%E5%BF%85%E8%A6%81)以获取更多信息。

当 v-if 与 v-for 一起使用时，v-for 具有比 v-if 更高的优先级。请查阅[列表渲染指南](https://cn.vuejs.org/v2/guide/list.html#v-for-with-v-if) 以获取详细信息。

## [列表渲染-循环](https://cn.vuejs.org/v2/guide/list.html)

### v-for迭代数组
```html
    <ul id="example-2">
      <li v-for="(item, index) in items">
        { { parentMessage } } - { { index } } - { { item.message } }
      </li>
    </ul>
    <script>
        var example2 = new Vue({
          el: '#example-2',
          data: {
            parentMessage: 'Parent',
            items: [
              { message: 'Foo' },
              { message: 'Bar' }
            ]
          }
        })
    </script>
```
第二个参数为当前项的索引。

### v-for迭代对象中的属性

```html
    <ul id="v-for-object" class="demo">
      <li v-for="value in object">
        { { value } }
      </li>
    </ul>
    <script>
        new Vue({
          el: '#v-for-object',
          data: {
            object: {
              title: 'How to do lists in Vue',
              author: 'Jane Doe',
              publishedAt: '2016-04-10'
            }
          }
        })
    </script>
```

你也可以提供第二个的参数为 property 名称 (也就是键名)：
```html
    <div v-for="(value, name) in object">
      { { name } }: { { value } }
    </div>
```
还可以用第三个参数作为索引：
```html
    <div v-for="(value, name, index) in object">
      { { index } }. { { name } }: { { value } }
    </div>
```

### v-for迭代数字
```html
    <!--count从1开始-->
    <div id="app">
      <ul>
        <li v-for="n in 10">
         { { n } }
        </li>
      </ul>
    </div>
```

### 注意事项
 
2.2.0+ 的版本里，当在组件中使用 v-for 时，key 现在是必须的。

当 Vue 正在更新使用 `v-for` 渲染的元素列表时，它默认使用“就地更新”的策略。如果数据项的顺序被改变，Vue 将不会移动 DOM 元素来匹配数据项的顺序，
而是就地更新每个元素，并且确保它们在每个索引位置正确渲染。这个类似 Vue 1.x 的 `track-by="$index"`。

这个默认的模式是高效的，但是**只适用于不依赖子组件状态或临时 DOM 状态 (例如：表单输入值) 的列表渲染输出**。

为了给 Vue 一个提示，以便它能跟踪每个节点的身份，从而重用和重新排序现有元素，你需要为每项提供一个唯一 `key` 属性：

```html
    <div v-for="item in items" v-bind:key="item.id">
      <!-- 内容 -->
    </div>
```

建议尽可能在使用 `v-for` 时提供 `key` attribute，除非遍历输出的 DOM 内容非常简单，或者是刻意依赖默认行为以获取性能上的提升。

因为它是 Vue 识别节点的一个通用机制，`key` 并不仅与 `v-for` 特别关联。后面我们将在指南中看到，它还具有其它用途。

**注意**：不要使用对象或数组之类的非基本类型值作为 v-for 的 key。请用字符串或数值类型的值。

## [过滤器](https://cn.vuejs.org/v2/guide/filters.html)

Vue.js 允许你自定义过滤器，可被用于一些常见的文本格式化。过滤器可以用在两个地方：双花括号插值和 v-bind 表达式 (后者从 2.1.0+ 开始支持)。
过滤器应该被添加在 JavaScript 表达式的尾部，由“管道”符号指示：

    <!-- 在双花括号中 -->
    { { message | capitalize } }
    
    <!-- 在 `v-bind` 中 -->
    <div v-bind:id="rawId | formatId"></div>
- 你可以在一个组件的选项中定义本地的过滤器：

        filters: {
          capitalize: function (value) {
            if (!value) return ''
            value = value.toString()
            return value.charAt(0).toUpperCase() + value.slice(1)
          }
        }
    
- 或者在创建 Vue 实例之前全局定义过滤器：

        Vue.filter('capitalize', function (value) {
          if (!value) return ''
          value = value.toString()
          return value.charAt(0).toUpperCase() + value.slice(1)
        })
        
        new Vue({
          // ...
        })
当全局过滤器和局部过滤器重名时，会采用局部过滤器。

过滤器可以串联：

    { { message | filterA | filterB } }
在这个例子中，filterA 被定义为接收单个参数的过滤器函数，表达式 message 的值将作为参数传入到函数中。然后继续调用同样被定义为接收单个参数的过滤器函数 filterB，将 filterA 的结果传递到 filterB 中。

过滤器是 JavaScript 函数，因此可以接收参数：

    { { message | filterA('arg1', arg2) } }   
这里，filterA 被定义为接收三个参数的过滤器函数。其中 message 的值作为第一个参数，普通字符串 'arg1' 作为第二个参数，表达式 arg2 的值作为第三个参数。

## [事件处理](https://cn.vuejs.org/v2/guide/events.html)

### 监听事件

可以用 `v-on` 指令监听 DOM 事件，并在触发时运行一些 JavaScript 代码。

示例：
```html
    <div id="example-1">
      <button v-on:click="counter += 1">Add 1</button>
      <p>The button above has been clicked { { counter } } times.</p>
    </div>
    <script>
        var example1 = new Vue({
          el: '#example-1',
          data: {
            counter: 0
          }
        })
    </script>
```

### 事件处理方法

然而许多事件处理逻辑会更为复杂，所以直接把 JavaScript 代码写在 `v-on` 指令中是不可行的。因此 `v-on` 还可以接收一个需要调用的方法名称。

示例：
```html
    <div id="example-2">
      <!-- `greet` 是在下面定义的方法名 -->
      <button v-on:click="greet">Greet</button>
    </div>
    <script>
        var example2 = new Vue({
          el: '#example-2',
          data: {
            name: 'Vue.js'
          },
          // 在 `methods` 对象中定义方法
          methods: {
            greet: function (event) {
              // `this` 在方法里指向当前 Vue 实例
              alert('Hello ' + this.name + '!')
              // `event` 是原生 DOM 事件
              if (event) {
                alert(event.target.tagName)
              }
            }
          }
        })
        
        // 也可以用 JavaScript 直接调用方法
        example2.greet() // => 'Hello Vue.js!'
    </script>
```

### 内联处理器中的方法

除了直接绑定到一个方法，也可以在内联 JavaScript 语句中调用方法：
```html
    <div id="example-3">
      <button v-on:click="say('hi')">Say hi</button>
      <button v-on:click="say('what')">Say what</button>
    </div>
    <script>
        new Vue({
          el: '#example-3',
          methods: {
            say: function (message) {
              alert(message)
            }
          }
        })
    </script>
```

有时也需要在内联语句处理器中访问原始的 DOM 事件。可以用特殊变量 $event 把它传入方法：
```html
    <button v-on:click="warn('Form cannot be submitted yet.', $event)">
      Submit
    </button>
    <script>
        // ...
        methods: {
          warn: function (message, event) {
            // 现在我们可以访问原生事件对象
            if (event) event.preventDefault()
            alert(message)
          }
        }
    </script>
```

### 事件修饰符

在事件处理程序中调用 `event.preventDefault()` 或 `event.stopPropagation()` 是非常常见的需求。尽管我们可以在方法中轻松实现这点，但更好的方式是：方法只有纯粹的数据逻辑，而不是去处理 DOM 事件细节。

为了解决这个问题，Vue.js 为 `v-on` 提供了事件修饰符。之前提过，修饰符是由点开头的指令后缀来表示的。

- .stop
- .prevent
- .capture
- .self
- .once
- .passive
    ```html
    <!-- 阻止单击事件继续传播 -->
        <a v-on:click.stop="doThis"></a>
        
        <!-- 提交事件不再重载页面 -->
        <form v-on:submit.prevent="onSubmit"></form>
        
        <!-- 修饰符可以串联 -->
        <a v-on:click.stop.prevent="doThat"></a>
        
        <!-- 只有修饰符 -->
        <form v-on:submit.prevent></form>
        
        <!-- 添加事件监听器时使用事件捕获模式 -->
        <!-- 即元素自身触发的事件先在此处理，然后才交由内部元素进行处理 -->
        <div v-on:click.capture="doThis">...</div>
        
        <!-- 只当在 event.target 是当前元素自身时触发处理函数 -->
        <!-- 即事件不是从内部元素触发的 -->
        <div v-on:click.self="doThat">...</div>
        <!-- 点击事件将只会触发一次 -->
        <a v-on:click.once="doThis"></a>
    ```
**注意**：使用修饰符时，顺序很重要；相应的代码会以同样的顺序产生。因此，用 `v-on:click.prevent.self` 会阻止所有的点击，而 `v-on:click.self.prevent` 只会阻止对元素自身的点击。

- 2.3.0 新增

    Vue 还对应 `addEventListener` 中的 `passive` 选项提供了 `.passive` 修饰符。
    ```html
        <!-- 滚动事件的默认行为 (即滚动行为) 将会立即触发 -->
        <!-- 而不会等待 `onScroll` 完成  -->
        <!-- 这其中包含 `event.preventDefault()` 的情况 -->
        <div v-on:scroll.passive="onScroll">...</div>
    ```
    这个 `.passive` 修饰符尤其能够提升移动端的性能。
    
**注意**：不要把 `.passive` 和 `.prevent` 一起使用，因为 `.prevent` 将会被忽略，同时浏览器可能会向你展示一个警告。请记住，`.passive` 会告诉浏览器你不想阻止事件的默认行为。

### 按键修饰符

在监听键盘事件时，我们经常需要检查详细的按键。Vue 允许为 `v-on` 在监听键盘事件时添加按键修饰符：

    <!-- 只有在 `key` 是 `Enter` 时调用 `vm.submit()` -->
    <input v-on:keyup.enter="submit">
你可以直接将 `KeyboardEvent.key` 暴露的任意有效按键名转换为 `kebab-case` 来作为修饰符。

    <input v-on:keyup.page-down="onPageDown">
在上述示例中，处理函数只会在 `$event.key` 等于 `PageDown` 时被调用。

- 按键码
    
    keyCode 的事件用法已经被废弃了并可能不会被最新的浏览器支持。
    
    使用 keyCode 特性也是允许的：
    
        <input v-on:keyup.13="submit">
    为了在必要的情况下支持旧浏览器，Vue 提供了绝大多数常用的按键码的别名：
    
    - `.enter` 
    - `.tab` 
    - `.delete` (捕获“删除”和“退格”键)
    - `.esc` 
    - `.space` 
    - `.up` 
    - `.down` 
    - `.left` 
    - `.right` 
    
    有一些按键 (`.esc` 以及所有的方向键) 在 IE9 中有不同的 `key` 值, 如果你想支持 IE9，这些内置的别名应该是首选。
    
    你还可以通过全局 config.keyCodes 对象[自定义按键修饰符别名](https://cn.vuejs.org/v2/api/#keyCodes)：
    
        // 可以使用 `v-on:keyup.f1`
        Vue.config.keyCodes.f1 = 112

### 系统修饰键

可以用如下修饰符来实现仅在按下相应按键时才触发鼠标或键盘事件的监听器。
- `.ctrl`
- `.alt`
- `.shift`
- `.meta`

**注意**：在 Mac 系统键盘上，meta 对应 command 键 (⌘)。在 Windows 系统键盘 meta 对应 Windows 徽标键 (⊞)。在 Sun 操作系统键盘上，meta 对应实心宝石键 (◆)。在其他特定键盘上，尤其在 MIT 和 Lisp 机器的键盘、以及其后继产品，比如 Knight 键盘、space-cadet 键盘，meta 被标记为“META”。在 Symbolics 键盘上，meta 被标记为“META”或者“Meta”。

例如：
```html
    <!-- Alt + C -->
    <input @keyup.alt.67="clear">
    
    <!-- Ctrl + Click -->
    <div @click.ctrl="doSomething">Do something</div>
```
请注意修饰键与常规按键不同，在和 `keyup` 事件一起用时，事件触发时修饰键必须处于按下状态。换句话说，只有在按住 `ctrl` 的情况下释放其它按键，才能触发 `keyup.ctrl`。而单单释放 `ctrl` 也不会触发事件。如果你想要这样的行为，请为 `ctrl` 换用 `keyCode：keyup.17`。

- `.exact` 修饰符

    `.exact` 修饰符允许你控制由精确的系统修饰符组合触发的事件。
    ```html
        <!-- 即使 Alt 或 Shift 被一同按下时也会触发 -->
        <button @click.ctrl="onClick">A</button>
        
        <!-- 有且只有 Ctrl 被按下的时候才触发 -->
        <button @click.ctrl.exact="onCtrlClick">A</button>
        
        <!-- 没有任何系统修饰符被按下的时候才触发 -->
        <button @click.exact="onClick">A</button>
    ```
    
- 鼠标按钮修饰符
    - `.left`
    - `.right`
    - `.middle`
    
这些修饰符会限制处理函数仅响应特定的鼠标按钮。

### 为什么在 HTML 中监听事件?

你可能注意到这种事件监听的方式违背了关注点分离 (separation of concern) 这个长期以来的优良传统。但不必担心，因为所有的 Vue.js 事件处理方法和表达式都严格绑定在当前视图的 ViewModel 上，它不会导致任何维护上的困难。实际上，使用 `v-on` 有几个好处：

- 扫一眼 HTML 模板便能轻松定位在 JavaScript 代码里对应的方法。

- 因为你无须在 JavaScript 里手动绑定事件，你的 ViewModel 代码可以是非常纯粹的逻辑，和 DOM 完全解耦，更易于测试。

- 当一个 ViewModel 被销毁时，所有的事件处理器都会自动被删除。你无须担心如何清理它们。

## [自定义指令](https://cn.vuejs.org/v2/guide/custom-directive.html)

### 简介

除了核心功能默认内置的指令 (`v-model` 和 `v-show`)，Vue 也允许注册自定义指令。注意，在 Vue2.0 中，代码复用和抽象的主要形式是组件。
然而，有的情况下，你仍然需要对普通 DOM 元素进行底层操作，这时候就会用到自定义指令。举个聚焦输入框的例子。

当页面加载时，该元素将获得焦点 (注意：`autofocus` 在移动版 Safari 上不工作)。事实上，只要你在打开这个页面后还没点击过任何内容，
这个输入框就应当还是处于聚焦状态。现在让我们用指令来实现这个功能：
```javascript
    // 注册一个全局自定义指令 `v-focus`
    Vue.directive('focus', {
      // 当被绑定的元素插入到 DOM 中时……
      inserted: function (el) {
        // 聚焦元素
        el.focus()
      }
    })
```
如果想注册局部指令，组件中也接受一个 `directives` 的选项：
```javascript
    directives: {
      focus: {
        // 指令的定义
        inserted: function (el) {
          el.focus()
        }
      }
    }
```
然后你可以在模板中任何元素上使用新的 `v-focus` 属性，如下：

    <input v-focus>

### 钩子函数

一个指令定义对象可以提供如下几个钩子函数 (均为可选)：

- `bind`：只调用一次，指令第一次绑定到元素时调用。在这里可以进行一次性的初始化设置。

- `inserted`：被绑定元素插入父节点时调用 (仅保证父节点存在，但不一定已被插入文档中)。

- `update`：所在组件的 VNode 更新时调用，但是可能发生在其子 VNode 更新之前。指令的值可能发生了改变，也可能没有。
但是你可以通过比较更新前后的值来忽略不必要的模板更新 (详细的钩子函数参数见下)。

- `componentUpdated`：指令所在组件的 VNode 及其子 VNode 全部更新后调用。

- `unbind`：只调用一次，指令与元素解绑时调用。

接下来我们来看一下钩子函数的参数 (即 `el`、`binding`、`vnode` 和 `oldVnode`)。

### 钩子函数参数

指令钩子函数会被传入以下参数：

- `el`：指令所绑定的元素，可以用来直接操作 DOM 。
- `binding`：一个对象，包含以下属性：
    - `name`：指令名，不包括 `v-` 前缀。
    - `value`：指令的绑定值，例如：`v-my-directive="1 + 1"` 中，绑定值为 `2`。
    - `oldValue`：指令绑定的前一个值，仅在 `update` 和 `componentUpdated` 钩子中可用。无论值是否改变都可用。
    - `expression`：字符串形式的指令表达式。例如 `v-my-directive="1 + 1"` 中，表达式为 `"1 + 1"`。
    - `arg`：传给指令的参数，可选。例如 `v-my-directive:foo` 中，参数为 `"foo"`。
    - `modifiers`：一个包含修饰符的对象。例如：`v-my-directive.foo.bar` 中，修饰符对象为 `{ foo: true, bar: true }`。
- `vnode`：`Vue` 编译生成的虚拟节点。移步 [VNode API](https://cn.vuejs.org/v2/api/#VNode-%E6%8E%A5%E5%8F%A3) 来了解更多详情。
- `oldVnode`：上一个虚拟节点，仅在 `update` 和 `componentUpdated` 钩子中可用。

**注意**：除了 el 之外，其它参数都应该是只读的，切勿进行修改。如果需要在钩子之间共享数据，建议通过元素的 `dataset` 来进行。

这是一个使用了这些属性的自定义钩子样例：

```html
    <div id="hook-arguments-example" v-demo:foo.a.b="message"></div>
    <script>
        Vue.directive('demo', {
          bind: function (el, binding, vnode) {
            var s = JSON.stringify
            el.innerHTML =
              'name: '       + s(binding.name) + '<br>' +
              'value: '      + s(binding.value) + '<br>' +
              'expression: ' + s(binding.expression) + '<br>' +
              'argument: '   + s(binding.arg) + '<br>' +
              'modifiers: '  + s(binding.modifiers) + '<br>' +
              'vnode keys: ' + Object.keys(vnode).join(', ')
          }
        })
        
        new Vue({
          el: '#hook-arguments-example',
          data: {
            message: 'hello!'
          }
        })
    </script>
```
结果：

    name: "demo"
    value: "hello!"
    expression: "message"
    argument: "foo"
    modifiers: {"a":true,"b":true}
    vnode keys: tag, data, children, text, elm, ns, context, fnContext, fnOptions,
    fnScopeId, key, componentOptions, componentInstance, parent, raw, isStatic,
    isRootInsert, isComment, isCloned, isOnce, asyncFactory, asyncMeta,
    isAsyncPlaceholder

### 动态指令参数

指令的参数可以是动态的。例如，在 `v-mydirective:[argument]="value" `中，`argument` 参数可以根据组件实例数据进行更新！这使得自定义指令可以在应用中被灵活使用。

例如你想要创建一个自定义指令，用来通过固定布局将元素固定在页面上。我们可以像这样创建一个通过指令值来更新竖直位置像素值的自定义指令：
```html
    <div id="baseexample">
      <p>Scroll down the page</p>
      <p v-pin="200">Stick me 200px from the top of the page</p>
    </div>
    <script>
        Vue.directive('pin', {
          bind: function (el, binding, vnode) {
            el.style.position = 'fixed'
            el.style.top = binding.value + 'px'
          }
        })
        
        new Vue({
          el: '#baseexample'
        })
    </script>
```
这会把该元素固定在距离页面顶部 200 像素的位置。但如果场景是我们需要把元素固定在左侧而不是顶部又该怎么办呢？这时使用动态参数就可以非常方便地根据每个组件实例来进行更新。
```html
    <div id="dynamicexample">
      <h3>Scroll down inside this section ↓</h3>
      <p v-pin:[direction]="200">I am pinned onto the page at 200px to the left.</p>
    </div>
    <script>
    Vue.directive('pin', {
      bind: function (el, binding, vnode) {
        el.style.position = 'fixed'
        var s = (binding.arg == 'left' ? 'left' : 'top')
        el.style[s] = binding.value + 'px'
      }
    })
    
    new Vue({
      el: '#dynamicexample',
      data: function () {
        return {
          direction: 'left'
        }
      }
    })
    </script>
```
这样这个自定义指令现在的灵活性就足以支持一些不同的用例了。

### 函数简写

在很多时候，你可能想在 `bind` 和 `update` 时触发相同行为，而不关心其它的钩子。比如这样写:
```javascript
    Vue.directive('color-swatch', function (el, binding) {
      el.style.backgroundColor = binding.value
    })
```

### 对象字面量

如果指令需要多个值，可以传入一个 JavaScript 对象字面量。记住，指令函数能够接受所有合法的 JavaScript 表达式。
```html
    <div v-demo="{ color: 'white', text: 'hello!' }"></div>
    <script>
    Vue.directive('demo', function (el, binding) {
      console.log(binding.value.color) // => "white"
      console.log(binding.value.text)  // => "hello!"
    })
    </script>
```

## [vue实例的生命周期](https://cn.vuejs.org/v2/api/#%E9%80%89%E9%A1%B9-%E7%94%9F%E5%91%BD%E5%91%A8%E6%9C%9F%E9%92%A9%E5%AD%90)

- 什么是生命周期：从Vue实例创建、运行、到销毁期间，总是伴随着各种各样的事件，这些事件，统称为生命周期！
- 生命周期钩子：就是生命周期事件的别名而已；
- 生命周期钩子 = 生命周期函数 = 生命周期事件
- 主要的生命周期函数分类：
- 创建期间的生命周期函数：
    - `beforeCreate`：实例刚在内存中被创建出来，此时，还没有初始化好 data 和 methods 属性
    - `created`：实例已经在内存中创建OK，此时 data 和 methods 已经创建OK，此时还没有开始 编译模板
    - `beforeMount`：此时已经完成了模板的编译，但是还没有挂载到页面中
    - `mounted`：此时，已经将编译好的模板，挂载到了页面指定的容器中显示
- 运行期间的生命周期函数：
    - `beforeUpdate`：状态更新之前执行此函数， 此时 data 中的状态值是最新的，但是界面上显示的 数据还是旧的，因为此时还没有开始重新渲染DOM节点
    - `updated`：实例更新完毕之后调用此函数，此时 data 中的状态值 和 界面上显示的数据，都已经完成了更新，界面已经被重新渲染好了！
- 销毁期间的生命周期函数：
    - `beforeDestroy`：实例销毁之前调用。在这一步，实例仍然完全可用。
    - `destroyed`：Vue 实例销毁后调用。调用后，Vue 实例指示的所有东西都会解绑定，所有的事件监听器

![](../../res/lifecycle.png)

## [vue-resource 实现 get, post, jsonp请求](https://github.com/pagekit/vue-resource)

除了 vue-resource 之外，还可以使用 axios 的第三方包实现实现数据的请求
- 之前的学习中，如何发起数据请求？
- 常见的数据请求类型？ get post jsonp
- 测试的URL请求资源地址：
    - get请求地址： http://vue.studyit.io/api/getlunbo
    - post请求地址：http://vue.studyit.io/api/post
    - jsonp请求地址：http://vue.studyit.io/api/jsonp
- JSONP的实现原理
    - 由于浏览器的安全性限制，不允许AJAX访问 协议不同、域名不同、端口号不同的 数据接口，浏览器认为这种访问不安全；
    - 可以通过动态创建script标签的形式，把script标签的src属性，指向数据接口的地址，因为script标签不存在跨域限制，这种数据获取方式，称作JSONP（注意：根据JSONP的实现原理，知晓，JSONP只支持Get请求）；
    - 具体实现过程：
        - 先在客户端定义一个回调方法，预定义对数据的操作；
        - 再把这个回调方法的名称，通过URL传参的形式，提交到服务器的数据接口；
        - 服务器数据接口组织好要发送给客户端的数据，再拿着客户端传递过来的回调方法名称，拼接出一个调用这个方法的字符串，发送给客户端去解析执行；
        - 客户端拿到服务器返回的字符串之后，当作Script脚本去解析执行，这样就能够拿到JSONP的数据了；
    - 带大家通过 Node.js ，来手动实现一个JSONP的请求例子；
    ```javascript
       const http = require('http');
       // 导入解析 URL 地址的核心模块
       const urlModule = require('url');
    
       const server = http.createServer();
       // 监听 服务器的 request 请求事件，处理每个请求
       server.on('request', (req, res) => {
         const url = req.url;
    
         // 解析客户端请求的URL地址
         var info = urlModule.parse(url, true);
    
         // 如果请求的 URL 地址是 /getjsonp ，则表示要获取JSONP类型的数据
         if (info.pathname === '/getjsonp') {
           // 获取客户端指定的回调函数的名称
           var cbName = info.query.callback;
           // 手动拼接要返回给客户端的数据对象
           var data = {
             name: 'zs',
             age: 22,
             gender: '男',
             hobby: ['吃饭', '睡觉', '运动']
           };
           // 拼接出一个方法的调用，在调用这个方法的时候，把要发送给客户端的数据，序列化为字符串，作为参数传递给这个调用的方法：
           var result = `${cbName}(${JSON.stringify(data)})`;
           // 将拼接好的方法的调用，返回给客户端去解析执行
           res.end(result);
         } else {
           res.end('404');
         }
       });
    
       server.listen(3000, () => {
         console.log('server running at http://127.0.0.1:3000');
    ```

## [Vue.js Ajax(axios)](https://www.runoob.com/vue2/vuejs-ajax-axios.html)

Vue.js 2.0 版本推荐使用 axios 来完成 ajax 请求。

Axios 是一个基于 Promise 的 HTTP 库，可以用在浏览器和 node.js 中。

Github开源地址： https://github.com/axios/axios

### 安装方法

- 使用 cdn:

    `<script src="https://unpkg.com/axios/dist/axios.min.js"></script>`
或 `<script src="https://cdn.staticfile.org/axios/0.18.0/axios.min.js"></script>`

- 使用 npm:  `$ npm install axios`

- 使用 bower: `$ bower install axios`

- 使用 yarn:  `$ yarn add axios`

### GET方法
- 我们可以简单的读取 JSON 数据：
    ```javascript
        new Vue({
          el: '#app',
          data () {
            return {
              info: null
            }
          },
          mounted () {
            axios
              .get('https://www.runoob.com/try/ajax/json_demo.json')
              .then(response => (this.info = response))
              .catch(function (error) { // 请求失败处理
                console.log(error);
              });
          }
        })
    ```
- 使用 `response.data` 读取 JSON 数据：
    ```html
        <div id="app">
          <h1>网站列表</h1>
          <div
           v-for="site in info"
         >
            { { site.name } }
          </div>
        </div>
        <script type = "text/javascript">
        new Vue({
          el: '#app',
          data () {
            return {
              info: null
            }
          },
          mounted () {
            axios
              .get('https://www.runoob.com/try/ajax/json_demo.json')
              .then(response => (this.info = response.data.sites))
              .catch(function (error) { // 请求失败处理
                console.log(error);
              });
          }
        })
        </script>
    ```
    
- GET 方法传递参数格式如下：
    ```javascript
        // 直接在 URL 上添加参数 ID=12345
        axios.get('/user?ID=12345')
          .then(function (response) {
            console.log(response);
          })
          .catch(function (error) {
            console.log(error);
          });
        
        // 也可以通过 params 设置参数：
        axios.get('/user', {
            params: {
              ID: 12345
            }
          })
          .then(function (response) {
            console.log(response);
          })
          .catch(function (error) {
            console.log(error);
          });
    ```

### POST方法
- 示例
    ```javascript
        new Vue({
          el: '#app',
          data () {
            return {
              info: null
            }
          },
          mounted () {
            axios
              .post('https://www.runoob.com/try/ajax/demo_axios_post.php')
              .then(response => (this.info = response))
              .catch(function (error) { // 请求失败处理
                console.log(error);
              });
          }
        })
    ```
    
- POST 方法传递参数格式如下：
    ```javascript
        axios.post('/user', {
            firstName: 'Fred',        // 参数 firstName
            lastName: 'Flintstone'    // 参数 lastName
          })
          .then(function (response) {
            console.log(response);
          })
          .catch(function (error) {
            console.log(error);
          });
    ```

### 执行多个并发请求
- 示例
    ```javascript
        function getUserAccount() {
          return axios.get('/user/12345');
        }
        
        function getUserPermissions() {
          return axios.get('/user/12345/permissions');
        }
        axios.all([getUserAccount(), getUserPermissions()])
          .then(axios.spread(function (acct, perms) {
            // 两个请求现在都执行完成
          }));
    ```

### axios API
- 可以通过向axios传递相关配置来创建请求
    ```javascript
        axios(config)
        // 发送 POST 请求
        axios({
          method: 'post',
          url: '/user/12345',
          data: {
            firstName: 'Fred',
            lastName: 'Flintstone'
          }
        });
        //  GET 请求远程图片
        axios({
          method:'get',
          url:'http://bit.ly/2mTM3nY',
          responseType:'stream'
        })
          .then(function(response) {
          response.data.pipe(fs.createWriteStream('ada_lovelace.jpg'))
        });
        axios(url[, config])
        // 发送 GET 请求（默认的方法）
        axios('/user/12345');
    ```

#### 请求方法的别名
- 为方便使用，官方为所有支持的请求方法提供了别名，可以直接使用别名来发起请求：
    ```text
        axios.request(config)
        axios.get(url[, config])
        axios.delete(url[, config])
        axios.head(url[, config])
        axios.post(url[, data[, config]])
        axios.put(url[, data[, config]])
        axios.patch(url[, data[, config]])
    ```
    - 注意：在使用别名方法时， url、method、data 这些属性都不必在配置中指定。

#### 并发
- 处理并发请求的助手函数
    ```javascript
        axios.all(iterable)
        axios.spread(callback)
    ```

#### 创建实例
- 可以使用自定义配置创建一个axios实例：
    ```javascript
        axios.create([config])
        const instance = axios.create({
          baseURL: 'https://some-domain.com/api/',
          timeout: 1000,
          headers: {'X-Custom-Header': 'foobar'}
        });
    ```

#### 实例方法
- 以下是可以的实例方法，指定的配置将与实例的配置合并
    ```javascript
        axios#request(config)
        axios#get(url[, config])
        axios#delete(url[, config])
        axios#head(url[, config])
        axios#post(url[, data[, config]])
        axios#put(url[, data[, config]])
        axios#patch(url[, data[, config]])
    ```

#### 请求配置项
- 下面是创建请求时可用的配置选项，注意只有url是必需的。如果没有指定method，请求将默认使用get方法。
    ```text
        {
          // `url` 是用于请求的服务器 URL
          url: "/user",
        
          // `method` 是创建请求时使用的方法
          method: "get", // 默认是 get
        
          // `baseURL` 将自动加在 `url` 前面，除非 `url` 是一个绝对 URL。
          // 它可以通过设置一个 `baseURL` 便于为 axios 实例的方法传递相对 URL
          baseURL: "https://some-domain.com/api/",
        
          // `transformRequest` 允许在向服务器发送前，修改请求数据
          // 只能用在 "PUT", "POST" 和 "PATCH" 这几个请求方法
          // 后面数组中的函数必须返回一个字符串，或 ArrayBuffer，或 Stream
          transformRequest: [function (data) {
            // 对 data 进行任意转换处理
        
            return data;
          }],
        
          // `transformResponse` 在传递给 then/catch 前，允许修改响应数据
          transformResponse: [function (data) {
            // 对 data 进行任意转换处理
        
            return data;
          }],
        
          // `headers` 是即将被发送的自定义请求头
          headers: {"X-Requested-With": "XMLHttpRequest"},
        
          // `params` 是即将与请求一起发送的 URL 参数
          // 必须是一个无格式对象(plain object)或 URLSearchParams 对象
          params: {
            ID: 12345
          },
        
          // `paramsSerializer` 是一个负责 `params` 序列化的函数
          // (e.g. https://www.npmjs.com/package/qs, http://api.jquery.com/jquery.param/)
          paramsSerializer: function(params) {
            return Qs.stringify(params, {arrayFormat: "brackets"})
          },
        
          // `data` 是作为请求主体被发送的数据
          // 只适用于这些请求方法 "PUT", "POST", 和 "PATCH"
          // 在没有设置 `transformRequest` 时，必须是以下类型之一：
          // - string, plain object, ArrayBuffer, ArrayBufferView, URLSearchParams
          // - 浏览器专属：FormData, File, Blob
          // - Node 专属： Stream
          data: {
            firstName: "Fred"
          },
        
          // `timeout` 指定请求超时的毫秒数(0 表示无超时时间)
          // 如果请求花费了超过 `timeout` 的时间，请求将被中断
          timeout: 1000,
        
          // `withCredentials` 表示跨域请求时是否需要使用凭证
          withCredentials: false, // 默认的
        
          // `adapter` 允许自定义处理请求，以使测试更轻松
          // 返回一个 promise 并应用一个有效的响应 (查阅 [response docs](#response-api)).
          adapter: function (config) {
            /* ... */
          },
        
          // `auth` 表示应该使用 HTTP 基础验证，并提供凭据
          // 这将设置一个 `Authorization` 头，覆写掉现有的任意使用 `headers` 设置的自定义 `Authorization`头
          auth: {
            username: "janedoe",
            password: "s00pers3cret"
          },
        
          // `responseType` 表示服务器响应的数据类型，可以是 "arraybuffer", "blob", "document", "json", "text", "stream"
          responseType: "json", // 默认的
        
          // `xsrfCookieName` 是用作 xsrf token 的值的cookie的名称
          xsrfCookieName: "XSRF-TOKEN", // default
        
          // `xsrfHeaderName` 是承载 xsrf token 的值的 HTTP 头的名称
          xsrfHeaderName: "X-XSRF-TOKEN", // 默认的
        
          // `onUploadProgress` 允许为上传处理进度事件
          onUploadProgress: function (progressEvent) {
            // 对原生进度事件的处理
          },
        
          // `onDownloadProgress` 允许为下载处理进度事件
          onDownloadProgress: function (progressEvent) {
            // 对原生进度事件的处理
          },
        
          // `maxContentLength` 定义允许的响应内容的最大尺寸
          maxContentLength: 2000,
        
          // `validateStatus` 定义对于给定的HTTP 响应状态码是 resolve 或 reject  promise 。如果 `validateStatus` 返回 `true` (或者设置为 `null` 或 `undefined`)，promise 将被 resolve; 否则，promise 将被 rejecte
          validateStatus: function (status) {
            return status &gt;= 200 &amp;&amp; status &lt; 300; // 默认的
          },
        
          // `maxRedirects` 定义在 node.js 中 follow 的最大重定向数目
          // 如果设置为0，将不会 follow 任何重定向
          maxRedirects: 5, // 默认的
        
          // `httpAgent` 和 `httpsAgent` 分别在 node.js 中用于定义在执行 http 和 https 时使用的自定义代理。允许像这样配置选项：
          // `keepAlive` 默认没有启用
          httpAgent: new http.Agent({ keepAlive: true }),
          httpsAgent: new https.Agent({ keepAlive: true }),
        
          // "proxy" 定义代理服务器的主机名称和端口
          // `auth` 表示 HTTP 基础验证应当用于连接代理，并提供凭据
          // 这将会设置一个 `Proxy-Authorization` 头，覆写掉已有的通过使用 `header` 设置的自定义 `Proxy-Authorization` 头。
          proxy: {
            host: "127.0.0.1",
            port: 9000,
            auth: : {
              username: "mikeymike",
              password: "rapunz3l"
            }
          },
        
          // `cancelToken` 指定用于取消请求的 cancel token
          // （查看后面的 Cancellation 这节了解更多）
          cancelToken: new CancelToken(function (cancel) {
          })
        }
    ```

#### 响应结构
- axios请求的响应包含以下信息
    ```text
        {
          // `data` 由服务器提供的响应
          data: {},
        
          // `status`  HTTP 状态码
          status: 200,
        
          // `statusText` 来自服务器响应的 HTTP 状态信息
          statusText: "OK",
        
          // `headers` 服务器响应的头
          headers: {},
        
          // `config` 是为请求提供的配置信息
          config: {}
        }
    ```
- 使用then时，会接收下面这样的响应：
    ```javascript
        axios.get("/user/12345")
          .then(function(response) {
            console.log(response.data);
            console.log(response.status);
            console.log(response.statusText);
            console.log(response.headers);
            console.log(response.config);
          });
    ```
在使用 catch 时，或传递 [rejection callback](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Promise/then) 作为 then 的第二个参数时，响应可以通过 error 对象可被使用。

#### 配置的默认值
- 你可以指定将被用在各个请求的配置默认值。
    - 全局的axios默认值
        ```javascript
        axios.defaults.baseURL = 'https://api.example.com';
        axios.defaults.headers.common['Authorization'] = AUTH_TOKEN;
        axios.defaults.headers.post['Content-Type'] = 'application/x-www-form-urlencoded';
        ```
    - 自定义实例默认值
        ```javascript
        // 创建实例时设置配置的默认值
        var instance = axios.create({
          baseURL: 'https://api.example.com'
        });
        
        // 在实例已创建后修改默认值
        instance.defaults.headers.common['Authorization'] = AUTH_TOKEN;
        ```

#### 配置的优先顺序
- 配置会以一个优先顺序进行合并。这个顺序是：在 lib/defaults.js 找到的库的默认值，然后是实例的 defaults 属性，最后是请求的 config 参数。后者将优先于前者。这里是一个例子：
    
    ```javascript
        // 使用由库提供的配置的默认值来创建实例
        // 此时超时配置的默认值是 `0`
        var instance = axios.create();
        
        // 覆写库的超时默认值
        // 现在，在超时前，所有请求都会等待 2.5 秒
        instance.defaults.timeout = 2500;
        
        // 为已知需要花费很长时间的请求覆写超时设置
        instance.get('/longRequest', {
          timeout: 5000
        });
    ```

#### 拦截器
- 在请求或响应被 then 或 catch 处理前拦截它们。
    ```javascript
        // 添加请求拦截器
        axios.interceptors.request.use(function (config) {
            // 在发送请求之前做些什么
            return config;
          }, function (error) {
            // 对请求错误做些什么
            return Promise.reject(error);
          });
        
        // 添加响应拦截器
        axios.interceptors.response.use(function (response) {
            // 对响应数据做点什么
            return response;
          }, function (error) {
            // 对响应错误做点什么
            return Promise.reject(error);
          });
    ```

- 如果你想在稍后移除拦截器，可以这样：
    ```javascript
        var myInterceptor = axios.interceptors.request.use(function () {/*...*/});
        axios.interceptors.request.eject(myInterceptor);
    ```

- 可以为自定义 axios 实例添加拦截器。
    ```javascript
        var instance = axios.create();
        instance.interceptors.request.use(function () {/*...*/});
    ```

     - 错误处理：
         ```javascript
            axios.get('/user/12345')
              .catch(function (error) {
                if (error.response) {
                  // 请求已发出，但服务器响应的状态码不在 2xx 范围内
                  console.log(error.response.data);
                  console.log(error.response.status);
                  console.log(error.response.headers);
                } else {
                  // Something happened in setting up the request that triggered an Error
                  console.log('Error', error.message);
                }
                console.log(error.config);
              });
        ```
   
     - 可以使用 validateStatus 配置选项定义一个自定义 HTTP 状态码的错误范围。
        ```javascript
            axios.get('/user/12345', {
              validateStatus: function (status) {
                return status < 500; // 状态码在大于或等于500时才会 reject
              }
            })
        ```

#### 取消
使用 cancel token 取消请求。

Axios 的 cancel token API 基于[cancelable promises proposal](https://github.com/tc39/proposal-cancelable-promises)

可以使用 CancelToken.source 工厂方法创建 cancel token，像这样：

```javascript
    var CancelToken = axios.CancelToken;
    var source = CancelToken.source();
    
    axios.get('/user/12345', {
      cancelToken: source.token
    }).catch(function(thrown) {
      if (axios.isCancel(thrown)) {
        console.log('Request canceled', thrown.message);
      } else {
        // 处理错误
      }
    });
    
    // 取消请求（message 参数是可选的）
    source.cancel('Operation canceled by the user.');
```
还可以通过传递一个 executor 函数到 CancelToken 的构造函数来创建 cancel token：
```javascript
    var CancelToken = axios.CancelToken;
    var cancel;
    
    axios.get('/user/12345', {
      cancelToken: new CancelToken(function executor(c) {
        // executor 函数接收一个 cancel 函数作为参数
        cancel = c;
      })
    });
    
    // 取消请求
    cancel();
```
注意：可以使用同一个 cancel token 取消多个请求。

#### 请求时使用 application/x-www-form-urlencoded
axios 会默认序列化 JavaScript 对象为 JSON。 如果想使用 application/x-www-form-urlencoded 格式，你可以使用下面的配置。

- 浏览器

    在浏览器环境，你可以使用 URLSearchParams API：
    ```javascript
        const params = new URLSearchParams();
        params.append('param1', 'value1');
        params.append('param2', 'value2');
        axios.post('/foo', params);
    ```
    URLSearchParams 不是所有的浏览器均支持。

    除此之外，你可以使用 qs 库来编码数据:
    ```javascript
        const qs = require('qs');
        axios.post('/foo', qs.stringify({ 'bar': 123 }));
        
        // Or in another way (ES6),
        
        import qs from 'qs';
        const data = { 'bar': 123 };
        const options = {
          method: 'POST',
          headers: { 'content-type': 'application/x-www-form-urlencoded' },
          data: qs.stringify(data),
          url,
        };
        axios(options);
    ```

- Node.js 环境

    在 node.js里, 可以使用 querystring 模块:
    ```javascript
        const querystring = require('querystring');
        axios.post('http://something.com/', querystring.stringify({ foo: 'bar' }));
    ```
    当然，同浏览器一样，你还可以使用 qs 库。

#### Promises

axios 依赖原生的 ES6 Promise 实现而被支持。

如果你的环境不支持 ES6 Promise，你可以使用 polyfill。

#### TypeScript支持

axios 包含 TypeScript 的定义。

```javascript
    import axios from "axios";
    axios.get("/user?ID=12345");
```

## 案例

[跑马灯](../../code/JavaScript/vue-跑马灯.html)

[简易计算器](../../code/JavaScript/vue-简易计算器.html)

[品牌管理案例](../../code/JavaScript/vue-品牌管理案例.html)






