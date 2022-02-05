package gee

import (
	"net/http"
	"strings"
)

type router struct {
	roots    map[string]*node
	handlers map[string]HandlerFunc
}

// roots key eg, roots['GET'] roots['POST']
// handlers key eg, handlers['GET-/p/:lang/doc'], handlers['POST-/p/book']

func newRouter() *router {
	return &router{
		roots:    make(map[string]*node),
		handlers: make(map[string]HandlerFunc),
	}
}

// 字符串分割， Only one * is allowed
//
// eg: /p/*name/* -> []string{"p", "*name"}
//
// eg: /p/:name -> []string{"p", ":name"}
//
// eg: /p/* -> []string{"p", "*"}
func parsePattern(pattern string) []string {
	vs := strings.Split(pattern, "/")
	parts := make([]string, 0)
	for _, item := range vs {
		if item != "" {
			parts = append(parts, item)
			if item[0] == '*' {
				break
			}
		}
	}
	return parts
}

// 添加路由（生成前缀树）
//
// 最后 router 示例：&{map[GET:0xc000148880] map[GET-/:<nil> GET-/assets/*filepath:<nil> GET-/hello/:name:<nil> GET-/hello/b/c:<nil> GET-/hi/:name:<nil>]}
func (r *router) addRoute(method string, pattern string, handler HandlerFunc) {
	parts := parsePattern(pattern)
	key := method + "-" + pattern
	_, ok := r.roots[method]
	if !ok { // 不存在则报错，则初始化节点
		r.roots[method] = &node{}
	}
	// 插入子节点
	// method 可能为： GET、POST等
	r.roots[method].insert(pattern, parts, 0)
	// 映射处理函数
	// key 例子：map[GET-/:<nil> GET-/assets/*filepath:<nil> GET-/hello/:name:<nil> GET-/hello/b/c:<nil> GET-/hi/:name:<nil>]}
	r.handlers[key] = handler
}

// 路由匹配和设置 Params
func (r *router) getRoute(method string, path string) (*node, map[string]string) {
	searchParts := parsePattern(path) // URL 切分，eg: /hello/nzb  -> ["hello", "nzb"]
	params := make(map[string]string)
	root, ok := r.roots[method]
	if !ok {
		return nil, nil
	}
	// 路由节点搜索
	// [node{pattern=/hello/:name, part=:name, isWild=true}]
	n := root.search(searchParts, 0)

	if n != nil {
		parts := parsePattern(n.pattern) // 切分 pattern, 设置参数
		for index, part := range parts {
			// 模糊匹配,然后设置对应的参数值
			if part[0] == ':' {
				params[part[1:]] = searchParts[index]
			}
			if part[0] == '*' && len(part) > 1 {
				params[part[1:]] = strings.Join(searchParts[index:], "/")
				break
			}
		}
		return n, params // node{pattern=/hello/:name, part=:name, isWild=true},map[name:nzb]
	}
	return nil, nil
}

// 获取某一方法(GET, POST等)的路由树
//
// eg: [node{pattern=/, part=, isWild=false} node{pattern=/hello/:name, part=:name, isWild=true}
//node{pattern=/hello/b/c, part=c, isWild=false} node{pattern=/hi/:name, part=:name, isWild=true}
//node{pattern=/assets/*filepath, part=*filepath, isWild=true}]
func (r *router) getRoutes(method string) []*node {
	root, ok := r.roots[method]
	if !ok {
		return nil
	}
	nodes := make([]*node, 0)
	root.travel(&nodes)
	return nodes
}

// 路由处理函数
func (r *router) handle(c *Context) {
	n, params := r.getRoute(c.Method, c.Path) // node{pattern=/hello/:name, part=:name, isWild=true},map[name:nzb]
	if n != nil {
		c.Params = params
		key := c.Method + "-" + n.pattern // GET-/hello/:name
		//r.handlers[key](c)
		c.handlers = append(c.handlers, r.handlers[key])
	} else { // 路由节点不存在404
		c.handlers = append(c.handlers, func(c *Context) {
			c.String(http.StatusNotFound, "404 NOT FOUND: %s\n", c.Path)
		})
	}
	c.Next() // 开始调用 c.handlers 里面的函数（包括中间件和视图）
}
