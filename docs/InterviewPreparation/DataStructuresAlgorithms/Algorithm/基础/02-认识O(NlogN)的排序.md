###### datetime:2023-06-03 15:30

###### author:nzb

# 认识O(NlogN)的排序

剖析递归行为和递归行为时间复杂度的估算

用递归方法找一个数组中的最大值，系统上到底是怎么做的？

- `master`公式：`T(N) = a*T(N/b) + O(N^d)`
- 解释
    - `T(N)`：母问题的数据量是`N`级别的，数据量规模为`N`
    - `aT(N/b)`：子问题的规模都是`N/b`，子问题的规模都是等量的，`a`是子问题调用次数
    - `O(N^d)`：除了子问题的调用之外，剩下的时间复杂度
    - 这样一类的递归都可以使用 `master`公式来算时间复杂度

- 结论
    - `log(b,a) > d` -> 复杂度为`O(N^log(b,a))`
    - `log(b,a) = d` -> 复杂度为`O(N^d * logN)`
    - `log(b,a) < d` -> 复杂度为`O(N^d)`

- [算法的复杂度与Master定理](www.gocalf.com/blog/algorithm-complexity-and-master-theorem.html)

- 示例(求最大值)

```python
# idx= [0, 1, 2, 3, 4, 5]
data = [3, 2, 5, 6, 7, 4]

"""
正常解法，重头遍历到尾，找最大值，时间复杂度 O(N)
用递归找最大值，递归结构图
                     p(0, 5)：0~5找最大值
                    /                  \
                p(0, 2)              p(3, 5)
              /         \           /       \
          p(0, 1)    p(2, 2)    p(3, 4)   p(5, 5)
          /      \              /     \ 
      p(0, 0)  p(1, 1)     p(3, 3)  p(4, 4) 
多叉树后序遍历，每个节点都需要子节点汇总得到结果才能得出最大值
"""


# master公式：T(N) = 2 * T(N/2) + O(1)
# a = 2, b = 2, d=0
# log(2, 2) > 0，所以时间复杂度：O(N^log(2,2) = O(N) 等价于重头遍历到尾，找最大值

def max_value(d, left, right):
    if left == right:  # O(1)
        return d[left]
    mid = left + ((right - left) >> 1)
    left_d = max_value(d, left, mid)  # T(N/2)
    right_d = max_value(d, mid + 1, right)  # T(N/2)
    return max(left_d, right_d)


print(max_value(data, 0, len(data) - 1))
```

- 示例2：假设一个数组
    - 前面1/3调用一次递归获取最大值，中间1/3调用一次递归获取最大值，后面1/3调用一次递归获取最大值，最后获取最大值，符合master公式：`T(N) = 3 * T(N/3) + O(1)`
    - 左边2/3调用一次递归获取最大值，右边2/3调用一次递归获取最大值，最后获取最大值，虽然有重叠部分，但是也符合master公式：`T(N) = 2 * T(N/(3/2)) + O(1)`
    - 左边1/3调用一次递归获取最大值，右边2/3调用一次递归获取最大值，最后获取最大值，不符合master公式，因为子问题规模不等
    - 前面1/3调用一次递归获取最大值，中间1/3调用一次递归获取最大值，后面1/3调用一次递归获取最大值，最后再打印一遍数据，然后获取最大值，符合master公式：`T(N) = 3 * T(N/3) + O(N)`

## 归并排序

- 整体就是一个简单递归，左边排好序、右边排好序、让其整体有序2）让其整体有序的过程里用了排外序方法
- 利用master公式来求解时间复杂度，`master`公式：`T(N) = 2 * T(N/2) + O(N)`，log(2, 2) = 1，所以时间复杂度：`O(N * logN)`
- 归并排序的实质
- 时间复杂度：`O(N * logN)`，额外空间复杂度`O(N)`

```python
# idx= [0, 1, 2, 3, 4, 5]
data = [3, 2, 1, 5, 6, 2]


def merge_sort(arr):
    if len(arr) < 2:
        return arr
    mid = len(arr) >> 1  # len(arr) // 2
    left = merge_sort(arr[:mid])  # T(N/2)
    right = merge_sort(arr[mid:])  # T(N/2)
    return merge(left, right)


def merge(left, right):
    tmp = []  # 数据排序到新数据，left和right，要么left要么right，拷贝到tmp，最后一次拷贝是整个数组长度，所以O(N)
    while left and right:  # 遍历一遍 O(N)
        if left[0] <= right[0]:
            tmp.append(left.pop(0))
        else:
            tmp.append(right.pop(0))
    tmp += left[:]
    tmp += right[:]
    return tmp


print(merge_sort(data))
```

- 归并排序是如何做到时间复杂度从 `O(N^2)`到 `O(N*logN)`?
    - 冒泡、选择排序，浪费了大量的比较行为，比如选择排序，0~N-1范围上，比较了N次才知道了放到0位置，只搞定了一个数，0~N-2范围上，比较了N-1次才搞定一个数，以此类推，每一轮的比较都是独立的，浪费了多次比较才搞定一个数
    - 归并排序没有浪费比较行为，左侧部分有序，右侧部分有序，接下来比较是左侧部分的指针和右侧部分的指针，依次从左到右，左侧跟右侧的比，这个比较行为信息没有浪费，
      变成了一个整体有序的部分，下一回轮到这个大部分继续跟另一个大部分继续merge出来一个更长的有序部分，依次往下传递，所以时间复杂度更优
















