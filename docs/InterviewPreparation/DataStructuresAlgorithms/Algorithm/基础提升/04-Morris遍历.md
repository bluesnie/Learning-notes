###### datetime:2024/2/28 19:02

###### author:nzb

# Morris遍历

> 面试用，笔试尽量用其他，出错率比较高

一种遍历二叉树的方式，并且时间复杂度`O(N)`，额外空间复杂度`O(1)`

通过利用原树中大量空闲指针(叶子节点左右节点都是空)的方式，达到节省空间的目的

如果规定不能改变数据结构，则不能用Morris遍历

## Morris遍历细节

- 假设来到当前节点`cur`，开始时`cur`来到头节点位置
- 1）如果`cur`没有左孩子，`cur`向右移动`(cur = cur.right)`
- 2）如果`cur`有左孩子，找到左子树上最右的节点`mostRight`：
    - a.如果`mostRight`的右指针指向空，让其指向`cur`，然后`cur`向左移动`(cur = cur.left)`
    - b.如果`mostRight`的右指针指向`cur`，让其指向`null`，然后cur向右移动`(cur = cur.right)`
- 3）`cur`为空时遍历停止

## Morris遍历的实质

建立一种机制，对于没有左子树的节点只到达一次，对于有左子树的节点会到达两次，递归版本每个节点一定会到达三次

```text

    1
   / \
  2   3
 /\   /\
4  5 6  7
cur=1, most_right=5, most_right.right->1
cur=2, most_right=4, most_right.right->2
cur=4, 无左树, cur = cur.right->2
cur=2, most_right=4, 因为most_right.right->cur 所以重置most_right.right->None, cur = cur.right
cur=5, 无左树, cur = cur.right->1
cur=1, most_right=5, 因为most_right.right->cur 所以重置most_right.right->None, cur = cur.right
cur=3, most_right=6, most_right.right->3
cur=6 , 无左树, cur = cur.right->3
cur=3, most_right=3, 因为most_right.right->cur 所以重置most_right.right->None, cur = cur.right
cur=7 左右都为空停

Morris遍历顺序：1 2 4 2 5 1 3 6 3 7
```

### Morris遍历

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left: TreeNode = left
        self.right: TreeNode = right


# Morris遍历
def morris(head: TreeNode):
    if not head:
        return
    cur = head
    while cur:
        most_right = cur.left
        # 有左子树
        if most_right:
            # 留在最右节点
            # 2个停止条件，1、右节点为空，2、右节点是当前节点(第二次到达)
            while most_right.right and most_right.right != cur:
                most_right = most_right.right
            if not most_right.right:  # 左子树最右节点, 第一次来到cur
                most_right.right = cur
                cur = cur.left
                continue
            else:  # 第二次，most_right.right == cur, 重置右节点，最后移到右子树
                most_right.right = None
        # 如果左子树为空，直接移到右子树
        cur = cur.right
```

### 先序遍历

```python
# 先序遍历
def morris_pre(head: TreeNode):
    """
    先序遍历
    只到达一次，直接打印
    到达两次的，第一次到达的时候打印
    :param head:
    :return:
    """
    if not head:
        return
    cur = head
    while cur:
        most_right = cur.left
        # 有左子树
        if most_right:
            # 留在最右节点
            # 2个停止条件，1、右节点为空，2、右节点是当前节点(第二次到达)
            while most_right.right and most_right.right != cur:
                most_right = most_right.right
            if not most_right.right:  # 左子树最右节点, 第一次来到cur
                print(cur.val, end=" ")
                most_right.right = cur
                cur = cur.left
                continue
            else:  # 第二次，most_right.right == cur, 重置右节点
                most_right.right = None
        else:
            print(cur.val, end=" ")
        # 如果左子树为空，直接移到右子树
        cur = cur.right


# 创建二叉树

"""
    1
   / \
  2   3
 / \
4   5
"""
root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)

morris_pre(root)  # 1 2 4 5 3
```

### 中序遍历

```python
# 中序遍历
def morris_in(head: TreeNode):
    """
    中序遍历
    只到达一次，直接打印
    到达两次的，第二次到达的时候打印
    :param head:
    :return:
    """
    if not head:
        return
    cur = head
    while cur:
        most_right = cur.left
        # 有左子树
        if most_right:
            # 留在最右节点
            # 2个停止条件，1、右节点为空，2、右节点是当前节点(第二次到达)
            while most_right.right and most_right.right != cur:
                most_right = most_right.right
            if not most_right.right:  # 左子树最右节点, 第一次来到cur
                most_right.right = cur
                cur = cur.left
                continue
            else:  # 第二次，most_right.right == cur, 重置右节点
                most_right.right = None
        print(cur.val, end=" ")  # 没有左子树只到达一次，有左子树但是，第二次的时候也会到达这，因为上面else没有continue，这里就是到达2次的第二次
        # 如果左子树为空，直接移到右子树
        cur = cur.right


# 创建二叉树

"""
    1
   / \
  2   3
 / \
4   5
"""
root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)

morris_in(root)  # 4 2 5 1 3

```

### 后序遍历

```python
# 后序遍历
def morris_pos(head: TreeNode):
    """
    后序遍历
        到达两次的，第二次到达的时候逆序打印左树右边界
        最后，单独打印整棵树的右边界(逆序)
    :param head:
    :return:
    """
    if not head:
        return
    cur = head
    while cur:
        most_right = cur.left
        # 有左子树
        if most_right:
            # 留在最右节点
            # 2个停止条件，1、右节点为空，2、右节点是当前节点(第二次到达)
            while most_right.right and most_right.right != cur:
                most_right = most_right.right
            if not most_right.right:  # 左子树最右节点, 第一次来到cur
                most_right.right = cur
                cur = cur.left
                continue
            else:  # 第二次，most_right.right == cur, 重置右节点
                most_right.right = None
                print_node(cur.left)  # 第二次到达的时候逆序打印左树右边界, 注意不能重置之前，不然右边界会回去
        # 如果左子树为空，直接移到右子树
        cur = cur.right
    # 单独打印整棵树右边界
    print_node(head)


def print_node(head: TreeNode):
    tail = reverse(head)  # 逆序
    cur = tail
    while cur:
        # 打印
        print(cur.val, end=" ")
        cur = cur.right
    # 逆序回去
    reverse(tail)


def reverse(head: TreeNode):
    """逆序右边界"""
    cur = head
    prev = None
    while cur:
        right = cur.right
        cur.right = prev
        prev = cur
        cur = right
    return prev


# 创建二叉树

"""
    1
   / \
  2   3
 / \
4   5
"""
root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)

morris_pos(root)  # 4 5 2 3 1

```

### 是否搜索二叉树

```python
# 搜索二叉树
def is_bst(head: TreeNode):
    if not head:
        return True
    cur = head
    prev_val = float("-inf")
    while cur:
        most_right = cur.left
        # 有左子树
        if most_right:
            # 留在最右节点
            # 2个停止条件，1、右节点为空，2、右节点是当前节点(第二次到达)
            while most_right.right and most_right.right != cur:
                most_right = most_right.right
            if not most_right.right:  # 左子树最右节点, 第一次来到cur
                most_right.right = cur
                cur = cur.left
                continue
            else:  # 第二次，most_right.right == cur, 重置右节点，最后移到右子树
                most_right.right = None
        # 如果左子树为空，直接移到右子树
        if cur.val <= prev_val:
            return False
        prev_val = cur.val
        cur = cur.right
    return True


root = TreeNode(5)
root.left = TreeNode(3)
root.right = TreeNode(7)
root.left.left = TreeNode(2)
root.left.right = TreeNode(4)
root.right.left = TreeNode(6)
root.right.right = TreeNode(8)
root.left.left.left = TreeNode(1)

print(is_bst(root))
```

## Morris遍历时间复杂度的证明

## 总结

- Morris遍历的解决了最本质的问题，所以很多问题都是以Morris遍历为最优解
- 如何判断使用Morris遍历作为最优解还是使用二叉树递归套路作为最优解
    - 如果你的方法必须做第三次的数据强整合，就需要二叉树的递归套路
    - 否则使用Morris遍历