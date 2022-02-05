package gee

import (
	"log"
	"reflect"
	"testing"
)

func newTestRouter() *router {
	r := newRouter()
	r.addRoute("GET", "/", nil)
	r.addRoute("GET", "/hello/:name", nil)
	r.addRoute("GET", "/hello/b/c", nil)
	r.addRoute("GET", "/hi/:name", nil)
	r.addRoute("GET", "/assets/*filepath", nil)
	return r
}

func TestParsePattern(t *testing.T) {
	ok := reflect.DeepEqual(parsePattern("/p/:name"), []string{"p", ":name"})
	ok = ok && reflect.DeepEqual(parsePattern("/p/*"), []string{"p", "*"})
	ok = ok && reflect.DeepEqual(parsePattern("/p/*name/*"), []string{"p", "*name"})
	if !ok {
		t.Fatal("test parasePattern failed")
	}
}

func TestGetRoute(t *testing.T) {
	log.SetFlags(log.Lshortfile | log.LstdFlags)
	r := newTestRouter()
	// &{map[GET:0xc000148880] map[GET-/:<nil> GET-/assets/*filepath:<nil>
	// GET-/hello/:name:<nil> GET-/hello/b/c:<nil> GET-/hi/:name:<nil>]}
	// 第一层：node{pattern=/, part=, isWild=false}
	// 第二层 children：[node{pattern=, part=hello, isWild=false} node{pattern=, part=hi, isWild=false} node{pattern=, part=assets, isWild=false}]
	// 第三次children：[node{pattern=/hello/:name, part=:name, isWild=true}]
	//a, _ := r.roots["GET"]
	//log.Println(a.children[0].children)
	log.Println("travel", r.getRoutes("GET"))
	log.Println(r)
	// example1
	n, ps := r.getRoute("GET", "/hello/nzb")
	log.Printf("TestGetRoute getRoute node:%v \n results:%v \n", n, ps)
	if n == nil {
		t.Fatal("nil shouldn't be returned")
	}

	if n.pattern != "/hello/:name" {
		t.Fatal("should math /hello/:name")
	}

	if ps["name"] != "nzb" {
		t.Fatal("name should be equal to 'nzb'")
	}
	log.Printf("matched path:%s, params['name']:%s\n", n.pattern, ps["name"])

	// example2
	n, ps = r.getRoute("GET", "/hello/b/c")
	log.Printf("TestGetRoute getRoute node:%v \n results:%v \n", n, ps)
	if n == nil {
		t.Fatal("nil shouldn't be returned")
	}

	if n.pattern != "/hello/b/c" {
		t.Fatal("should math /hello/b/c")
	}
	log.Printf("matched path:%s\n", n.pattern)

	// example3
	n, ps = r.getRoute("GET", "/assets/css/t.css")
	log.Printf("TestGetRoute getRoute node:%v \n results:%v \n", n, ps)
	if n == nil {
		t.Fatal("nil shouldn't be returned")
	}

	if n.pattern != "/assets/*filepath" {
		t.Fatal("should math /assets/*filepath")
	}
	if ps["filepath"] != "css/t.css" {
		t.Fatal("filepath should be equal to 'css/t.css'")
	}

	log.Printf("matched path:%s, params['filepath']:%s\n", n.pattern, ps["filepath"])
}
