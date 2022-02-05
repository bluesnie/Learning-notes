package gee

import (
	"fmt"
	"strings"
)

type node struct {
	pattern  string  // 待匹配路由，例如 /p/:lang
	part     string  // 路由中的一部分，例如 :lang
	children []*node // 子节点，例如 [doc, tutorial, intro]
	isWild   bool    // 是否模糊匹配，part 含有 : 或 * 时为true
}

func (n *node) String() string {
	return fmt.Sprintf("node{pattern=%s, part=%s, isWild=%t}", n.pattern, n.part, n.isWild)
}

// 一边匹配一边插入的方法
func (n *node) insert(pattern string, parts []string, height int) {
	// 如果已经匹配完了，那么将pattern赋值给该node，表示它是一个完整的url
	// 这是递归的终止条件
	if len(parts) == height {
		n.pattern = pattern
		return
	}

	part := parts[height]
	child := n.matchChild(part)
	if child == nil {
		// 没有匹配上，那么进行生成，放到n节点的子列表中
		child = &node{part: part, isWild: part[0] == ':' || part[0] == '*'}
		n.children = append(n.children, child)
	}
	// 接着插入下一个part节点
	child.insert(pattern, parts, height+1)
}

// 路由节点搜索，根路由进来，path=/hello/nzb, parts = ["hello", "nzb"]
func (n *node) search(parts []string, height int) *node {
	if len(parts) == height || strings.HasPrefix(n.part, "*") {
		// 递归终止条件，找到末尾了或者通配符
		if n.pattern == "" {
			// pattern为空字符串表示它不是一个完整的url，匹配失败
			return nil
		}
		return n
	}

	part := parts[height]
	// 获取所有可能的子路径
	// 第一次查找：children [node{pattern=, part=hello, isWild=false}]
	// 第二次查找：children [node{pattern=/hello/:name, part=:name, isWild=true}]
	children := n.matchChildren(part)
	for _, child := range children {
		// 对于每条路径接着用下一part去查找
		result := child.search(parts, height+1)
		if result != nil {
			// 找到了即返回,eg:node{pattern=/hello/:name, part=:name, isWild=true}
			return result
		}
	}

	return nil
}

// 查找所有完整的url，保存到列表中
func (n *node) travel(list *([]*node)) {
	if n.pattern != "" {
		// 递归终止条件
		*list = append(*list, n)
	}
	for _, child := range n.children {
		// 一层一层的递归找pattern是非空的节点
		child.travel(list)
	}
}

// 第一个匹配成功的节点，用于插入
// 找到匹配的子节点，场景是用在插入时使用，找到1个匹配的就立即返回
func (n *node) matchChild(part string) *node {
	// 遍历n节点的所有子节点，看是否能找到匹配的子节点，将其返回
	for _, child := range n.children {
		// 如果有模糊匹配的也会成功匹配上
		if child.part == part || child.isWild {
			return child
		}
	}
	return nil
}

// 这个函数跟matchChild有点像，但它是返回所有匹配的子节点，原因是它的场景是用以查找
// 它必须返回所有可能的子节点来进行遍历查找，用于查找
func (n *node) matchChildren(part string) []*node {
	nodes := make([]*node, 0)
	for _, child := range n.children {
		if child.part == part || child.isWild {
			nodes = append(nodes, child)
		}
	}
	return nodes
}
