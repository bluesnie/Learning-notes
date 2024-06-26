###### datetime:2022/02/01 15::02
###### author:nzb

# Go Mock (gomock)简明教程

## 1 gomock 简介
上一篇文章 Go Test 单元测试简明教程 介绍了 Go 语言中单元测试的常用方法，包括子测试(subtests)、表格驱动测试(table-driven tests)、帮助函数(helpers)、网络测试和基准测试(Benchmark)等。这篇文章介绍一种新的测试方法，mock/stub 测试，当待测试的函数/对象的依赖关系很复杂，并且有些依赖不能直接创建，例如数据库连接、文件I/O等。这种场景就非常适合使用 mock/stub 测试。简单来说，就是用 mock 对象模拟依赖项的行为。

> GoMock is a mocking framework for the Go programming language. It integrates well with Go’s built-in testing package, but can be used in other contexts too.

[gomock](https://github.com/golang/mock) 是官方提供的 mock 框架，同时还提供了 mockgen 工具用来辅助生成测试代码。

使用如下命令即可安装：
```text
go get -u github.com/golang/mock/gomock
go get -u github.com/golang/mock/mockgen
```
## 2 一个简单的 Demo

```text
// db.go
type DB interface {
	Get(key string) (int, error)
}

func GetFromDB(db DB, key string) int {
	if value, err := db.Get(key); err == nil {
		return value
	}

	return -1
}
```

假设 `DB` 是代码中负责与数据库交互的部分(在这里用 map 模拟)，测试用例中不能创建真实的数据库连接。这个时候，如果我们需要测试 `GetFromDB` 这个函数内部的逻辑，就需要 mock 接口 `DB`。

第一步：使用 `mockgen` 生成 `db_mock.go`。一般传递三个参数。包含需要被mock的接口得到源文件`source`，生成的目标文件`destination`，包名`package`。
```text
$ mockgen -source=db.go -destination=db_mock.go -package=main
```
第二步：新建 `db_test.go`，写测试用例。
```text
func TestGetFromDB(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish() // 断言 DB.Get() 方法是否被调用

	m := NewMockDB(ctrl)
	m.EXPECT().Get(gomock.Eq("Tom")).Return(100, errors.New("not exist"))

	if v := GetFromDB(m, "Tom"); v != -1 {
		t.Fatal("expected -1, but got", v)
	}
}
```
- 这个测试用例有2个目的，一是使用 `ctrl.Finish()` 断言 `DB.Get()` 被是否被调用，如果没有被调用，后续的 `mock` 就失去了意义；
- 二是测试方法 `GetFromDB()` 的逻辑是否正确(如果 `DB.Get()` 返回 `error`，那么 `GetFromDB()` 返回 -1)。
- `NewMockDB()` 的定义在 `db_mock.go` 中，由 `mockgen` 自动生成。

最终的代码结构如下：
```text
project/
    |--db.go
    |--db_mock.go // generated by mockgen
    |--db_test.go
```
执行测试：
```text
$ go test . -cover -v
=== RUN   TestGetFromDB
--- PASS: TestGetFromDB (0.00s)
PASS
coverage: 81.2% of statements
ok      example 0.008s  coverage: 81.2% of statements
```
## 3 打桩(stubs)

在上面的例子中，当 `Get()` 的参数为 `Tom`，则返回 error，这称之为`打桩(stub)`，有明确的参数和返回值是最简单打桩方式。除此之外，检测调用次数、调用顺序，动态设置返回值等方式也经常使用。

### 3.1 参数(Eq, Any, Not, Nil)
```text
m.EXPECT().Get(gomock.Eq("Tom")).Return(0, errors.New("not exist"))
m.EXPECT().Get(gomock.Any()).Return(630, nil)
m.EXPECT().Get(gomock.Not("Sam")).Return(0, nil) 
m.EXPECT().Get(gomock.Nil()).Return(0, errors.New("nil")) 
```
- `Eq(value)` 表示与 value 等价的值。
- `Any()` 可以用来表示任意的入参。
- `Not(value)` 用来表示非 value 以外的值。
- `Nil()` 表示 None 值

### 3.2 返回值(Return, DoAndReturn)
```text
m.EXPECT().Get(gomock.Not("Sam")).Return(0, nil)
m.EXPECT().Get(gomock.Any()).Do(func(key string) {
    t.Log(key)
})
m.EXPECT().Get(gomock.Any()).DoAndReturn(func(key string) (int, error) {
    if key == "Sam" {
        return 630, nil
    }
    return 0, errors.New("not exist")
})
```
- `Return` 返回确定的值
- `Do` Mock 方法被调用时，要执行的操作吗，忽略返回值。
- `DoAndReturn` 可以动态地控制返回值。

### 3.3 调用次数(Times)
```text
func TestGetFromDB(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	m := NewMockDB(ctrl)
	m.EXPECT().Get(gomock.Not("Sam")).Return(0, nil).Times(2)
	GetFromDB(m, "ABC")
	GetFromDB(m, "DEF")
}
```
- `Times()` 断言 Mock 方法被调用的次数。
- `MaxTimes()` 最大次数。
- `MinTimes()` 最小次数。
- `AnyTimes()` 任意次数（包括 0 次）。

### 3.4 调用顺序(InOrder)
```text
func TestGetFromDB(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish() // 断言 DB.Get() 方法是否被调用

	m := NewMockDB(ctrl)
	o1 := m.EXPECT().Get(gomock.Eq("Tom")).Return(0, errors.New("not exist"))
	o2 := m.EXPECT().Get(gomock.Eq("Sam")).Return(630, nil)
	gomock.InOrder(o1, o2)
	GetFromDB(m, "Tom")
	GetFromDB(m, "Sam")
}
```
## 4 如何编写可 mock 的代码

写可测试的代码与写好测试用例是同等重要的，如何写可 mock 的代码呢？

- mock 作用的是接口，因此将依赖抽象为接口，而不是直接依赖具体的类。
- 不直接依赖的实例，而是使用依赖注入降低耦合性。

> 在软件工程中，依赖注入的意思为，给予调用方它所需要的事物。 “依赖”是指可被方法调用的事物。依赖注入形式下，调用方不再直接指使用“依赖”，取而代之是“注入” 。“注入”是指将“依赖”传递给调用方的过程。在“注入”之后，调用方才会调用该“依赖”。传递依赖给调用方，而不是让让调用方直接获得依赖，这个是该设计的根本需求。
> [依赖注入 - Wikipedia](https://zh.wikipedia.org/zh-cn/%E4%BE%9D%E8%B5%96%E6%B3%A8%E5%85%A5)

如果 `GetFromDB()` 方法长这个样子
```text
func GetFromDB(key string) int {
	db := NewDB()
	if value, err := db.Get(key); err == nil {
		return value
	}

	return -1
}
```
对 `DB` 接口的 mock 并不能作用于 `GetFromDB()` 内部，这样写是没办法进行测试的。那如果将接口 `db DB` 通过参数传递到 `GetFromDB()`，那么就可以轻而易举地传入 `Mock` 对象了。










