package gee

import (
	"html/template"
	"log"
	"net/http"
	"path"
	"strings"
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
		router        *router
		groups        []*RouterGroup     // store all groups
		htmlTemplates *template.Template // for html render
		funcMap       template.FuncMap   // for html render
	}
)

// New is the constructor of gee.Engine
func New() *Engine {
	engine := &Engine{router: newRouter()} // 初始化相关信息
	engine.RouterGroup = &RouterGroup{engine: engine}
	engine.groups = []*RouterGroup{engine.RouterGroup}
	return engine
}

// Default use Logger() & Recovery middlewares
func Default() *Engine {
	engine := New()
	engine.Use(Logger(), Recovery())
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

// Use is defined to add middleware to the group
func (group *RouterGroup) Use(middlewares ...HandlerFunc) {
	group.middlewares = append(group.middlewares, middlewares...)
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

// create static handler
func (group *RouterGroup) createStaticHandler(relativePath string, fs http.FileSystem) HandlerFunc {
	absolutePath := path.Join(group.prefix, relativePath)
	fileServer := http.StripPrefix(absolutePath, http.FileServer(fs))
	return func(c *Context) {
		file := c.Param("filepath")
		// Check if file exists and/or if we have permission to access it
		if _, err := fs.Open(file); err != nil {
			c.Status(http.StatusNotFound)
			return
		}
		fileServer.ServeHTTP(c.Writer, c.Req)
	}
}

// Static serve static files
func (group *RouterGroup) Static(relativePath string, root string) {
	handler := group.createStaticHandler(relativePath, http.Dir(root))
	urlPattern := path.Join(relativePath, "/*filepath")
	// Register GET handlers
	group.GET(urlPattern, handler)
}

// SetFuncMap for custom render function
func (engine *Engine) SetFuncMap(funcMap template.FuncMap) {
	engine.funcMap = funcMap
}

func (engine *Engine) LoadHTMLGlob(pattern string) {
	engine.htmlTemplates = template.Must(template.New("").Funcs(engine.funcMap).ParseGlob(pattern))
}

// Run defines the method to start a http server
func (engine *Engine) Run(add string) (err error) {
	return http.ListenAndServe(add, engine)
}

// 相当于 flask 的钩子函数，所有请求都经过该方法
func (engine *Engine) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	var middlerwares []HandlerFunc
	// 根据路由查找中间件
	for _, group := range engine.groups {
		if strings.HasPrefix(req.URL.Path, group.prefix) {
			middlerwares = append(middlerwares, group.middlewares...)
		}
	}
	c := newContext(w, req) // 上下文
	c.handlers = middlerwares
	c.engine = engine // 上下文添加一个引擎指针成员
	engine.engine.router.handle(c)
}
