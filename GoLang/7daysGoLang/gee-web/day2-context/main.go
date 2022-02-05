package main

/*
$ curl -i http://localhost:9999/
HTTP/1.1 200 OK
Date: Mon, 12 Aug 2019 16:52:52 GMT
Content-Length: 18
Content-Type: text/html; charset=utf-8
<h1>Hello Gee</h1>

$ curl "http://localhost:9999/hello?name=world"
hello world, you're at /hello

$ curl "http://localhost:9999/login" -X POST -d "username=hello&password=1234"
{"password":"1234","username":"hello"}

$ curl "http://localhost:9999/xxx"
404 NOT FOUND: /xxx
*/

import (
	"gee"
	"net/http"
)

func main() {
	r := gee.New()
	r.GET("/", func(c *gee.Context) {
		c.HTML(http.StatusOK, "<h1>Hello World<h1>")
	})

	r.GET("/hello", func(c *gee.Context) {
		// expect /hello?name=world
		c.String(http.StatusOK, "hello %s, you're at %s\n", c.Query("name"), c.Path)
	})

	r.POST("/login", func(c *gee.Context) {
		c.JSON(http.StatusOK, gee.H{
			"username": c.PostForm("username"),
			"password": c.PostForm("password"),
		})
	})

	r.Run(":9999")
}
