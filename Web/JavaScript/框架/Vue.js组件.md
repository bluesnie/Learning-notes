###### datetime:2019/11/27 16:40
###### author:nzb

## [Vue.js组件](https://cn.vuejs.org/v2/guide/components.html#data-%E5%BF%85%E9%A1%BB%E6%98%AF%E4%B8%80%E4%B8%AA%E5%87%BD%E6%95%B0)

### 基础

#### 基本示例

简单示例
    
```html
<div id="components-demo">
  <button-counter></button-counter>
</div>

<script >
// 定义一个名为 button-counter 的新组件
Vue.component('button-counter', {
  data: function () {
    return {
      count: 0
    }
  },
  template: '<button v-on:click="count++">You clicked me {{ count }} times.</button>'
})


new Vue({ el: '#components-demo' })
</script>
```

组件是可复用的 Vue 实例，且带有一个名字：在这个例子中是 `<button-counter>`。我们可以在一个通过 `new Vue` 创建的 Vue 根实例中，把这个组件作为自定义元素来使用：

因为组件是可复用的 Vue 实例，所以它们与 new Vue 接收相同的选项，例如 data、computed、watch、methods 以及生命周期钩子等。仅有的例外是像 el 这样根实例特有的选项。

#### 组件的复用

你可以将组件进行任意次数的复用：

```html
<div id="components-demo">
    <button-counter></button-counter>
    <button-counter></button-counter>
    <button-counter></button-counter>
</div>  
```
注意当点击按钮时，每个组件都会各自独立维护它的 count。因为你每用一次组件，就会有一个它的新实例被创建。

##### data必须是一个函数

当我们定义这个 `<button-counter>` 组件时，你可能会发现它的 data 并不是像这样直接提供一个对象：
```text
data: {
    count: 0
}   
```
取而代之的是，一个组件的 data 选项必须是一个函数，因此每个实例可以维护一份被返回对象的独立的拷贝：

```javascript
data: function f() {
  return {
      count: 0
  }
}
```

#### [组件的注册](https://cn.vuejs.org/v2/guide/components-registration.html)

通常一个应用会以一颗嵌套的组件树的形式来组织
![](../../res/components.png)

例如，你可能会有页头、侧边栏、内容区等组件，每个组件又包含了其它的像导航链接、博文之类的组件。

为了能在模板中使用，这些组件必须先注册以便 Vue 能够识别。这里有两种组件的注册类型：全局注册和局部注册。

##### 组件名

在注册一个组件的时候，我们始终需要给它一个名字。比如在全局注册的时候我们已经看到了：

```javascript
Vue.component('my-component-name', { /* ... */ })
```
该组件名就是 Vue.component 的第一个参数。

你给予组件的名字可能依赖于你打算拿它来做什么。当直接在 DOM 中使用一个组件 (而不是在字符串模板或单文件组件) 的时候，我们强烈推荐遵循 W3C 规范中的自定义组件名 (字母全小写且必须包含一个连字符)。这会帮助你避免和当前以及未来的 HTML 元素相冲突。

你可以在风格指南中查阅到关于组件名的其它建议。

- 组件名大小写

    - 使用kebab-case
    
        ```javascript
        Vue.component('my-component-name', { /* ... */ })
        ```
        当使用 kebab-case (短横线分隔命名) 定义一个组件时，你也必须在引用这个自定义元素时使用 kebab-case，例如 `<my-component-name>`。

    - 使用 PascalCase
        
        ```javascript
        Vue.component('MyComponentName', { /* ... */ })
        ```
        当使用 PascalCase (首字母大写命名) 定义一个组件时，你在引用这个自定义元素时两种命名法都可以使用。也就是说 `<my-component-name>` 和 `<MyComponentName>` 都是可接受的。注意，尽管如此，直接在 DOM (即非字符串的模板) 中使用时只有 kebab-case 是有效的。

##### 全局注册

到目前为止，我们只用过 Vue.component 来创建组件：
```javascript
Vue.component('my-component-name', {
  // ... 选项 ...
})
```
这些组件是全局注册的。也就是说它们在注册之后可以用在任何新创建的 Vue 根实例 (new Vue) 的模板中。比如：

```html
<div id="app">
  <component-a></component-a>
  <component-b></component-b>
  <component-c></component-c>
</div>
<script >
Vue.component('component-a', { /* ... */ })
Vue.component('component-b', { /* ... */ })
Vue.component('component-c', { /* ... */ })

new Vue({ el: '#app' })
</script>

```
在所有子组件中也是如此，也就是说这三个组件在各自内部也都可以相互使用。

##### 局部注册

全局注册往往是不够理想的。比如，如果你使用一个像 webpack 这样的构建系统，全局注册所有的组件意味着即便你已经不再使用一个组件了，它仍然会被包含在你最终的构建结果中。这造成了用户下载的 JavaScript 的无谓的增加。

在这些情况下，你可以通过一个普通的 JavaScript 对象来定义组件：
```javascript
var ComponentA = { /* ... */ }
var ComponentB = { /* ... */ }
var ComponentC = { /* ... */ }
```
然后在 components 选项中定义你想要使用的组件：

```javascript
new Vue({
  el: '#app',
  components: {
    'component-a': ComponentA,
    'component-b': ComponentB
  }
})
```
对于 components 对象中的每个属性来说，其属性名就是自定义元素的名字，其属性值就是这个组件的选项对象。

注意**局部注册的组件在其子组件中不可用**。例如，如果你希望 ComponentA 在 ComponentB 中可用，则你需要这样写：
```javascript
var ComponentA = { /* ... */ }

var ComponentB = {
  components: {
    'component-a': ComponentA
  },
  // ...
}
```
或者如果你通过 Babel 和 webpack 使用 ES2015 模块，那么代码看起来更像：
```javascript
import ComponentA from './ComponentA.vue'

export default {
  components: {
    ComponentA
  },
  // ...
}
```
注意在 ES2015+ 中，在对象中放一个类似 ComponentA 的变量名其实是 ComponentA: ComponentA 的缩写，即这个变量名同时是：

- 用在模板中的自定义元素的名称
- 包含了这个组件选项的变量名

#### 通过 Prop 向子组件传递数据

早些时候，我们提到了创建一个博文组件的事情。问题是如果你不能向这个组件传递某一篇博文的标题或内容之类的我们想展示的数据的话，它是没有办法使用的。这也正是 prop 的由来。

Prop 是你可以在组件上注册的一些自定义特性。当一个值传递给一个 prop 特性的时候，它就变成了那个组件实例的一个属性。为了给博文组件传递一个标题，我们可以用一个 props 选项将其包含在该组件可接受的 prop 列表中：
```javascript
Vue.component('blog-post', {
  props: ['title'],
  template: '<h3>{{ title }}</h3>'
})
```
一个组件默认可以拥有任意数量的 prop，任何值都可以传递给任何 prop。在上述模板中，你会发现我们能够在组件实例中访问这个值，就像访问 data 中的值一样。

一个 prop 被注册之后，你就可以像这样把数据作为一个自定义特性传递进来：
```html
<blog-post title="My journey with Vue"></blog-post>
<blog-post title="Blogging with Vue"></blog-post>
<blog-post title="Why Vue is so fun"></blog-post>
```
然而在一个典型的应用中，你可能在 data 里有一个博文的数组：
```javascript
new Vue({
  el: '#blog-post-demo',
  data: {
    posts: [
      { id: 1, title: 'My journey with Vue' },
      { id: 2, title: 'Blogging with Vue' },
      { id: 3, title: 'Why Vue is so fun' }
    ]
  }
})
```
并想要为每篇博文渲染一个组件：
```html
<blog-post
  v-for="post in posts"
  v-bind:key="post.id"
  v-bind:title="post.title"
></blog-post>
```
如上所示，你会发现我们可以使用 `v-bind` 来动态传递 prop。这在你一开始不清楚要渲染的具体内容，比如[从一个 API 获取博文列表](https://jsfiddle.net/chrisvfritz/sbLgr0ad)的时候，是非常有用的。

到目前为止，关于 prop 你需要了解的大概就这些了，如果你阅读完本页内容并掌握了它的内容，我们会推荐你再回来把 [prop](https://cn.vuejs.org/v2/guide/components-props.html) 读完。

#### 单个根元素

当构建一个 `<blog-post>` 组件时，你的模板最终会包含的东西远不止一个标题：
```html
<h3>{{ title }}</h3>
```

最最起码，你会包含这篇博文的正文：
```html
<h3>{{ title }}</h3>
<div v-html="content"></div>
```

然而如果你在模板中尝试这样写，Vue 会显示一个错误，并解释道 every component must have a single root element (每个组件必须只有一个根元素)。你可以将模板的内容包裹在一个父元素内，来修复这个问题，例如：
```html
<div class="blog-post">
  <h3>{{ title }}</h3>
  <div v-html="content"></div>
</div>
```

看起来当组件变得越来越复杂的时候，我们的博文不只需要标题和内容，还需要发布日期、评论等等。为每个相关的信息定义一个 prop 会变得很麻烦：
```html
<blog-post
  v-for="post in posts"
  v-bind:key="post.id"
  v-bind:title="post.title"
  v-bind:content="post.content"
  v-bind:publishedAt="post.publishedAt"
  v-bind:comments="post.comments"
></blog-post>
```

所以是时候重构一下这个 `<blog-post>` 组件了，让它变成接受一个单独的 `post` prop：
```html
<blog-post
  v-for="post in posts"
  v-bind:key="post.id"
  v-bind:post="post"
></blog-post>
Vue.component('blog-post', {
  props: ['post'],
  template: `
    <div class="blog-post">
      <h3>{{ post.title }}</h3>
      <div v-html="post.content"></div>
    </div>
  `
})
```
**注意**：上述的这个和一些接下来的示例使用了 JavaScript 的模板字符串来让多行的模板更易读。它们在 IE 下并没有被支持，所以如果你需要在不 (经过 Babel 或 TypeScript 之类的工具) 编译的情况下支持 IE，请使用折行转义字符取而代之。

现在，不论何时为 `post` 对象添加一个新的属性，它都会自动地在 `<blog-post>` 内可用。

#### 动态组件(组件切换)

有点时候，在不同组件之间进行动态切换是非常有用的，比如在一个多标签的界面里：
![](../../res/vue-动态组件.png)

上述内容可以通过 Vue 的 `<component>` 元素加一个特殊的 `is` 特性来实现：
```html
<!-- 组件会在 `currentTabComponent` 改变时改变 -->
<component v-bind:is="currentTabComponent"></component>
```
在上述示例中，`currentTabComponent` 可以包括

- 已注册组件的名字，或
- 一个组件的选项对象

你可以在[这里](https://jsfiddle.net/chrisvfritz/o3nycadu/)查阅并体验完整的代码，或在[这个版本](https://jsfiddle.net/chrisvfritz/b2qj69o1/)了解绑定组件选项对象，而不是已注册组件名的示例。

到目前为止，关于动态组件你需要了解的大概就这些了，如果你阅读完本页内容并掌握了它的内容，我们会推荐你再回来把[动态和异步组件](https://cn.vuejs.org/v2/guide/components-dynamic-async.html)读完。

#### 多个组件的过渡动画

多个组件的过渡简单很多 - 我们不需要使用 `key` 特性。相反，我们只需要使用[动态组件](https://cn.vuejs.org/v2/guide/components.html#%E5%8A%A8%E6%80%81%E7%BB%84%E4%BB%B6)。
其中mode="out-in"属性是先出后进
```html
<transition name="component-fade" mode="out-in">
  <component v-bind:is="view"></component>
</transition>
```
```javascript
new Vue({
  el: '#transition-components-demo',
  data: {
    view: 'v-a'
  },
  components: {
    'v-a': {
      template: '<div>Component A</div>'
    },
    'v-b': {
      template: '<div>Component B</div>'
    }
  }
})
```
```css
.component-fade-enter-active, .component-fade-leave-active {
  transition: opacity .3s ease;
}
.component-fade-enter, .component-fade-leave-to
/* .component-fade-leave-active for below version 2.1.8 */ {
  opacity: 0;
}
```

#### 父组件向子组件传值
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="./lib/vue.js"></script>
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <title>Document</title>
</head>
<body>
    <div id="app">
        <!--父组件，可以在引用子组件的时候，通过属性绑定（v:bind）的形式，把需要传递给子组件的数据
        传递到子组件内部-->
        <com1 :parentmsg="msg">

        </com1>
    </div>
    <script>
        var vm = new Vue({
            el: '#app',
            data:{
                msg: '123父组件中的数据' 
            },
            methods: {
                
            },
            components:{
                'com1':{
                    //子组件中，默认无法访问到父组件中的data和methods
                    template: '<h1 @click="change"> 这是子组件 {{parentmsg}}</h1>',
                    //注意，组件中的所有props中的数据都是通过父组件传递给子组件的
                    //propes中的数据是只可读
                    props: ['parentmsg'] ,// 把父组件传递过来的parentmsg属性， 数组中，定义一下，这样才能用这个数据,
                    //注意子组件中的data数据，并不是通过父组件传递过来的，而是子组件字有的，比如：子组件通过Ajax请求回来的值，可以放到data中
                    //dat a中的数据可读可写
                    data(){
                        return {
                            title: '123',
                            content: 'qqq'
                        }
                    },
                    methods: {
                        change(){
                            this.parentmsg='被修改'
                        }
                    },
                }
            }
        })
    </script>
</body>
</html>
```

#### 父组件把方法传递给子组件
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <title>Document</title>
    <script src="./lib/vue.js"></script>
    
</head>
<body>
    <div id="app">
        <!--父组件想子组件传递方法使用的是事件绑定机制: v-on 当我们自定义了一个事件属性之后，
        子组件就能过通过某型方法，来调用传进去的这个方法了-->
        <com2 @func="show">
        </com2>
    </div>
    <template id='temp1'>
        <div>
            <h1>子组件</h1>
            <input type="button" @click="myclick" value="这个是子组件的按钮，点击它，触发父组件传过来的func方法 ">
        </div>
    </template>
    <script>
        //定义了一个字面类型的组件模板对象
        var com2 = {
            template: '#temp1' ,
            methods: {
                myclick(){
                    //当点击子组件方法的时候，如何拿到父组件传递过来的func方法
                    //emit是触发的意思
                    this.$emit('func', this.sonmsg)
                }
            },  
            data() {
                return {
                    sonmsg: {
                        name: 'aaa',
                        age: 6
                    }
                 }
            }
        }
        var vm = new Vue({
            el: '#app',
            data: {
                datamsgFromSon: null
            },
            methods: {
                show(data){
                   // console.log('调用了父组件身上的show方法'+data)
                   console.log(data)
                   this.datamsgFromSon = data
                },
            },
            components:{
                com2  
            },
        })
    </script>
</body>
</html>
```

#### ref获取DOM元素和组件
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <script src="./lib/vue.js"></script>
    <title>Document</title>
</head>
<body>
    <div id="app">
        <input type="button" value="获取元素" @click="getElement">
        <h3 ref='myh3'>
            哈哈哈  拉拉    
        </h3>
        <login ref="myLogin">

        </login>
    </div>
    <script>
        var login = {
            template: '<h1> 登陆组件 </h1>',
            data(){
                return {
                    msg: 'template msg'
                }
            },
            methods: {
                show(){
                    console.log('调用了子组件的方法')
                }
            },
        }
        var vm = new Vue({
            el: '#app',
            data: {
            },
            methods: {
                getElement(){
                    console.log(this.$refs.myh3.innerText)
                    console.log(this.$refs.myLogin.msg)
                    this.$refs.myLogin.show()
                }
            },
            components:{
                login
            }
        })
    </script>
</body>
</html>
```








