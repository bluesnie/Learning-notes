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

## [模板语法](https://cn.vuejs.org/v2/guide/syntax.html)

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
        {{ parentMessage }} - {{ index }} - {{ item.message }}
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
        {{ value }}
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
      {{ name }}: {{ value }}
    </div>
```
还可以用第三个参数作为索引：
```html
    <div v-for="(value, name, index) in object">
      {{ index }}. {{ name }}: {{ value }}
    </div>
```

### v-for迭代数字
```html
    <!--count从1开始-->
    <div id="app">
      <ul>
        <li v-for="n in 10">
         {{ n }}
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
    {{ message | capitalize }}
    
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

    {{ message | filterA | filterB }}
在这个例子中，filterA 被定义为接收单个参数的过滤器函数，表达式 message 的值将作为参数传入到函数中。然后继续调用同样被定义为接收单个参数的过滤器函数 filterB，将 filterA 的结果传递到 filterB 中。

过滤器是 JavaScript 函数，因此可以接收参数：

    {{ message | filterA('arg1', arg2) }}   
这里，filterA 被定义为接收三个参数的过滤器函数。其中 message 的值作为第一个参数，普通字符串 'arg1' 作为第二个参数，表达式 arg2 的值作为第三个参数。

## [事件处理](https://cn.vuejs.org/v2/guide/events.html)

### 监听事件

可以用 `v-on` 指令监听 DOM 事件，并在触发时运行一些 JavaScript 代码。

示例：
```html
    <div id="example-1">
      <button v-on:click="counter += 1">Add 1</button>
      <p>The button above has been clicked {{ counter }} times.</p>
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

## 案例

[跑马灯](../../code/JavaScript/vue-跑马灯.html)

[简易计算器](../../code/JavaScript/vue-简易计算器.html)

[品牌管理案例](../../code/JavaScript/vue-品牌管理案例.html)

















