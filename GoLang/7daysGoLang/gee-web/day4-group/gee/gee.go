package gee

import (
	"log"
	"net/http"
)

// HandlerFunc defines the request handler used by gee
type HandlerFunc func(*Context)

// 这里为什么是 Engine 嵌套（继承） RouterGroup，而不是 RouterGroup 嵌套（继承） Engine？
// Go语言的嵌套在其他语言中类似于继承，子类必然是比父类有更多的成员变量和方法。
// RouterGroup 仅仅是负责分组路由，Engine 除了分组路由外，还有很多其他的功能。
// RouterGroup 继承 Engine 的 Run()，ServeHTTP 等方法是没有意义的。

// Engine implement the interface of ServeHTTP
type (
	RouterGroup struct {
		prefix      string
		middlewares []HandlerFunc // support middleware
		// parent 之前设计是用来拼接 prefix 的，每个 group 只记录自己的部分，最后通过 parent 层层回溯拼接。不过后来改用 group.prefix + prefix 的方式 group 初始化时已经拼接了完整的 prefix，所以不需要 parent 了，可以删除。
		parent *RouterGroup // support nesting
		engine *Engine      // all groups share a Engine instance
	}
	Engine struct {
		*RouterGroup
		router *router
		groups []*RouterGroup // store all groups
	}
)

// New is the constructor of gee.Engine
func New() *Engine {
	engine := &Engine{router: newRouter()} // 初始化相关信息
	engine.RouterGroup = &RouterGroup{engine: engine}
	engine.groups = []*RouterGroup{engine.RouterGroup}
	return engine
}

// Group is defined to create a new RouterGroup
// remember all groups share the same Engine instance
func (group *RouterGroup) Group(prefix string) *RouterGroup {
	engine := group.engine
	newGroup := &RouterGroup{
		prefix: group.prefix + prefix,
		parent: group,
		engine: engine,
	}
	engine.groups = append(engine.groups, newGroup)
	return newGroup
}

func (group *RouterGroup) addRoute(method string, comp string, handler HandlerFunc) {
	pattern := group.prefix + comp
	log.Printf("Route %4s - %s", method, pattern)
	group.engine.router.addRoute(method, pattern, handler)
}

// GET defines the method to add GET request
func (group *RouterGroup) GET(pattern string, handler HandlerFunc) {
	// 这里错了，Group 方法返回的最新实例，里面是有 prefix属性，而 group.engine是上一个，初始化的时候的，prefix为空字符串，所以注册路由没有分组
	//group.engine.addRoute("GET", pattern, handler)
	group.addRoute("GET", pattern, handler) // 正确

}

// POST defines the method to add POST request
func (group *RouterGroup) POST(pattern string, handler HandlerFunc) {
	group.addRoute("POST", pattern, handler)
}

// Run defines the method to start a http server
func (engine *Engine) Run(add string) (err error) {
	return http.ListenAndServe(add, engine)
}

// 相当于 flask 的钩子函数，所有请求都经过该方法
func (engine *Engine) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	c := newContext(w, req) // 上下文
	engine.engine.router.handle(c)
}
