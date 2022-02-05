###### datetime:2022/02/05 16:58

###### author:nzb

# 7天用Go从零实现Web框架Gee教程

## 设计一个框架

大部分时候，我们需要实现一个 Web 应用，第一反应是应该使用哪个框架。不同的框架设计理念和提供的功能有很大的差别。比如 Python 语言的 `django`和`flask`，前者大而全，后者小而美。Go语言/golang 也是如此，新框架层出不穷，比如`Beego`，`Gin`，`Iris`等。那为什么不直接使用标准库，而必须使用框架呢？在设计一个框架之前，我们需要回答框架核心为我们解决了什么问题。只有理解了这一点，才能想明白我们需要在框架中实现什么功能。

我们先看看标准库`net/http`如何处理一个请求。

```go
func main() {
    http.HandleFunc("/", handler)
    http.HandleFunc("/count", counter)
    log.Fatal(http.ListenAndServe("localhost:8000", nil))
}

func handler(w http.ResponseWriter, r *http.Request) {
    fmt.Fprintf(w, "URL.Path = %q\n", r.URL.Path)
}
```

`net/http`提供了基础的Web功能，即监听端口，映射静态路由，解析HTTP报文。一些Web开发中简单的需求并不支持，需要手工实现。

- 动态路由：例如`hello/:name`，`hello/*`这类的规则。
- 鉴权：没有分组/统一鉴权的能力，需要在每个路由映射的handler中实现。
- 模板：没有统一简化的HTML机制。
- ...

当我们离开框架，使用基础库时，需要频繁手工处理的地方，就是框架的价值所在。但并不是每一个频繁处理的地方都适合在框架中完成。Python有一个很著名的Web框架，名叫[`bottle`](https://github.com/bottlepy/bottle)，整个框架由`bottle.py`一个文件构成，共4400行，可以说是一个微框架。那么理解这个微框架提供的特性，可以帮助我们理解框架的核心能力。

- 路由(Routing)：将请求映射到函数，支持动态路由。例如`'/hello/:name`。
- 模板(Templates)：使用内置模板引擎提供模板渲染机制。
- 工具集(Utilites)：提供对 cookies，headers 等处理机制。
- 插件(Plugin)：Bottle本身功能有限，但提供了插件机制。可以选择安装到全局，也可以只针对某几个路由生效。
- ...

## Gee 框架

这个教程将使用 Go 语言实现一个简单的 Web 框架，起名叫做`Gee`，[`geektutu.com`](https://geektutu.com)的前三个字母。我第一次接触的 Go 语言的 Web 框架是`Gin`，`Gin`的代码总共是14K，其中测试代码9K，也就是说实际代码量只有**5K**。`Gin`也是我非常喜欢的一个框架，与Python中的`Flask`很像，小而美。

`7天实现Gee框架`这个教程的很多设计，包括源码，参考了`Gin`，大家可以看到很多`Gin`的影子。

时间关系，同时为了尽可能地简洁明了，这个框架中的很多部分实现的功能都很简单，但是尽可能地体现一个框架核心的设计原则。例如`Router`的设计，虽然支持的动态路由规则有限，但为了性能考虑匹配算法是用`Trie树`实现的，`Router`最重要的指标之一便是性能。

希望这个教程能够对你有所启发，如果对 Gee 有任何好的建议，欢迎提[issues - Github](https://github.com/geektutu/7days-golang/issues) 和 PR。教程中的任何问题，可以直接在文章末尾评论。

## 目录

- 第一天：[前置知识(http.Handler接口)](./gee-day1.md)，[Code - Github](../day1-http-base)
- 第二天：[上下文设计(Context)](./gee-day2.md)，[Code - Github](../day2-context)
- 第三天：[Trie树路由(Router)](./gee-day3.md)，[Code - Github](../day3-router)
- 第四天：[分组控制(Group)](./gee-day4.md)，[Code - Github](../day4-group)
- 第五天：[中间件(Middleware)](./gee-day5.md)，[Code - Github](../day5-middleware)
- 第六天：[HTML模板(Template)](./gee-day6.md)，[Code - Github](../day6-template)
- 第七天：[错误恢复(Panic Recover)](./gee-day7.md)，[Code - Github](../day7-panic-recover)

## 推荐阅读

- [Go 语言简明教程](../../../Go简明教程/01-Go语言简明教程.md)
- [Go Test 单元测试简明教程](../../../Go简明教程/07-Go-Test单元测试简明教程.md)
- [Go Gin 简明教程](../../../Go简明教程/02-Go-Gin-简明教程.md)