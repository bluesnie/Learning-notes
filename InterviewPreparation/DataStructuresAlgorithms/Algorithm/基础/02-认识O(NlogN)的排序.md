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

- 整体就是一个简单递归，左边排好序、右边排好序、让其整体有序
- 让其整体有序的过程里用了排外序方法
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
    # mid = len(arr) >> 1  # len(arr) // 2
    # return  merge(merge_sort(arr[:mid]), merge_sort(arr[mid:]))


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

- 归并排序实质，归并排序是如何做到时间复杂度从 `O(N^2)`到 `O(N*logN)`?
    - 冒泡、选择排序，浪费了大量的比较行为，比如选择排序，0~N-1范围上，比较了N次才知道了放到0位置，只搞定了一个数，0~N-2范围上，比较了N-1次才搞定一个数，以此类推，每一轮的比较都是独立的，浪费了多次比较才搞定一个数
    - 归并排序没有浪费比较行为，左侧部分有序，右侧部分有序，接下来比较是左侧部分的指针和右侧部分的指针，依次从左到右，左侧跟右侧的比，这个比较行为信息没有浪费，
      变成了一个整体有序的部分，下一回轮到这个大部分继续跟另一个大部分继续merge出来一个更长的有序部分，依次往下传递，所以时间复杂度更优

### 归并排序的扩展

- 小和问题：在一个数组中，每一个数左边比当前数小的数累加起来，叫做这个数组的小和。求一个数组的小和。
    - 例子:`[1,3,4,2,5]` 1左边比1小的数，没有; 3左边比3小的数，1; 4左边比4小的数，1、3; 2左边比2小的数，1; 5左边比5小的数，1、3、4、2; 所以小和为`1+1+3+1+1+3+4+2=16`
    - 暴力解法：O(N^2)
    - 转换思路，求一个数前面的小和，比如对于1来说，右边有多少个数比1大，就产生多少个数乘以1的小和，右边有多少个数比3大，就产生多少个数乘以3的小和
        - `1: 4 * 1`
        - `3: 2 * 3`
        - `4: 4 * 1`
        - `2: 2 * 1`

```python
# idx = [0, 1, 2, 3, 4]
data = [1, 3, 4, 2, 5]


def small_sum(arr):
    if not arr or len(arr) < 2:
        return 0
    return merge_sort(arr, 0, len(arr) - 1)


def merge_sort(arr, left, right):
    if left == right:
        return 0
    mid = left + ((right - left) >> 1)  # len(arr) // 2
    # 左组在排序的时候求小和，右组在排序的时候求小和，左组和右组合并的时候也要求小和
    return merge_sort(arr, left, mid) + merge_sort(arr, mid + 1, right) + merge(arr, left, mid, right)


def merge(arr, left, mid, right):
    # tmp = [0] * (right - left + 1)  # 长度：right - left + 1
    # i = 0
    tmp = []
    p1 = left
    p2 = mid + 1
    res = 0
    while p1 <= mid and p2 <= right:  # 都不越界
        if arr[p1] < arr[p2]:
            res += (right - p2 + 1) * arr[p1]  # 求小和
            # tmp[i] = arr[p1]
            tmp.append(arr[p1])
            p1 += 1
        else:  # 跟普通归并排序不同的点，左组的数和右组的数相等的时候，需要先拷贝右组的数，并且不产生小和，否则不知道右组有多少个小和：[1,1,1,2,2,3,4,5]  [1,1,2,3,4,4,5,5]
            # tmp[i] = arr[p2]
            tmp.append(arr[p2])
            p2 += 1
        # i += 1
    # while p1 <= mid:
    #     tmp[i] = arr[p1]
    #     i += 1
    #     p1 += 1
    # while p2 <= right:
    #     tmp[i] = arr[p2]
    #     i += 1
    #     p2 += 1
    # 归并排序，排序不可少，不然不清楚右边有多少个数比左边大
    tmp += arr[p1:mid + 1]
    tmp += arr[p2:right + 1]
    for k, v in enumerate(tmp):
        arr[left + k] = v
    return res


print(small_sum(data))
```    

- 逆序对问题：在一个数组中，左边的数如果比右边的数大，则这两个数构成一个逆序对，请打印所有逆序对(请找到逆序对)。
    - 例子:`[3,2,4,5,0]`
        - `3,2`
        - `3,0`
        - `2,0`
        - `4,0`
        - `5,0`

```python
# idx = [0, 1, 2, 3, 4]
data = [3, 2, 4, 5, 0]


def reverse_pair(arr):
    if not arr or len(arr) < 2:
        return 0
    return merge_sort(arr, 0, len(arr) - 1)


def merge_sort(arr, left, right):
    if left == right:
        return 0
    mid = left + ((right - left) >> 1)
    return merge_sort(arr, left, mid) + merge_sort(arr, mid + 1, right) + merge(arr, left, mid, right)


def merge(arr, left, mid, right):
    tmp = []
    cnt = 0
    p1 = left
    p2 = mid + 1
    while p1 <= mid and p2 <= right:
        if arr[p1] > arr[p2]:
            print(arr[p1], arr[p2])
            tmp.append(arr[p1])
            cnt += right - p2 + 1
            p1 += 1
        else:  # 跟普通归并排序不同的点，左组的数和右组的数相等的时候，需要先拷贝右组的数，并且不产生逆序对[1,1,1,2,2,3,4,5]  [1,1,2,3,4,4,5,5]
            tmp.append(arr[p2])
            p2 += 1

    # 归并排序，排序不可少，不然不清楚右边有多少个数比左边大
    tmp += arr[p1:mid + 1]
    tmp += arr[p2:right + 1]
    for k, v in enumerate(tmp):
        arr[left + k] = v
    return cnt


print(reverse_pair(data))
```

## 快速排序

荷兰国旗问题

- 问题一

  给定一个数组`arr`，和一个数`num`，请把小于等于`num`的数放在数组的左边，大于`num`的数放在数组的右边。要求额外空间复杂度`O(1)`，时间复杂度`O(N)`

```python
"""
         |
 <=区    |  [3, 5, 6, 7, 4, 3, 5, 8], num = 5
         |  i位置，指向3
1、[i] <= num, 把当前数[i]和小于等于区的下一个数做交换，小于等于区往右扩，i++
2、[i] > num, 小于等于区不动, i++

流程：
arr[i] = 3 <= num
               |
 <=区      [3, | 5, 6, 7, 4, 3, 5, 8]
               | i位置，指向5

arr[i] = 5 <= num                  
                  |
 <=区      [3, 5, | 6, 7, 4, 3, 5, 8]
                  | i位置，指向6

arr[i] = 6 > num，小于等于区不动, i++               
                  |
 <=区      [3, 5, | 6, 7, 4, 3, 5, 8]
                  |    i位置，指向7
                  
arr[i] = 7 > num，小于等于区不动, i++                  
                  |
 <=区      [3, 5, | 6, 7, 4, 3, 5, 8]
                  |       i位置，指向4

arr[i] = 4 <= num  
                     |
 <=区      [3, 5, 4, | 7, 6, 3, 5, 8]
                     |       i位置，指向3

arr[i] = 3 <= num  
                        |
 <=区      [3, 5, 4, 3, | 6, 7, 5, 8]
                        |       i位置，指向5
                        
arr[i] = 5 <= num  
                           |
 <=区      [3, 5, 4, 3, 5, | 7, 6, 8]
                           |       i位置，指向8

arr[i] = 8 > num  
                           |
 <=区      [3, 5, 4, 3, 5, | 7, 6, 8]
                           |          i位置越界停止
"""
```

- 问题二(荷兰国旗问题)

  给定一个数组`arr`，和一个数`num`，请把小于`num`的数放在数组的左边，等于`num`的数放在数组的中间，大于`num`的数放在数组的右边。要求额外空间复杂度`O(1)`，时间复杂度`O(N)`

```python
"""
         |                                            |      
 <区     |  [3, 5, 6, 3, 4, 5, 2, 6, 9, 0], num = 5   |  > 区   
         |   i位置，指向3                              |
1、[i] < num, 把当前数[i]和小于区的下一个数做交换，小于区往右扩，i++
2、[i] = num, i++
3、[i] > num, 把当前数[i]和大于区的前一个数做交换，大于区往左扩，i原地不变

流程：
arr[i] = 3 < num
               |                               |   
 <=区      [3, |  5, 6, 3, 4, 5, 2, 6, 9, 0]   |  > 区   
               | i位置，指向5                    |

arr[i] = 5 = num                  
               |                               |   
 <=区      [3, |  5, 6, 3, 4, 5, 2, 6, 9, 0]   |  > 区   
               |     i位置，指向6               |

arr[i] = 6 > num，把当前数[i]和大于区的前一个数做交换，大于区往左扩，i原地不变，为什么不变，因为它是右边过来的没作比较过                        
               |                         |   
 <=区      [3, | 5, 0, 3, 4, 5, 2, 6, 9, | 6]     > 区   
               |     i位置，指向0          |
                  
arr[i] = 0 < num           
                 |                      |   
 <=区      [3, 0, | 5, 3, 4, 5, 2, 6, 9, | 6]     > 区   
                 |    i位置，指向3       |

arr[i] = 3 < num  
                     |                   |   
 <=区      [3, 0, 3, | 5, 4, 5, 2, 6, 9, | 6]     > 区   
                     |    i位置，指向4     |

arr[i] = 4 < num  
                        |               |   
 <=区      [3, 0, 3, 4, | 5, 5, 2, 6, 9, | 6]     > 区   
                        |    i位置，指向5  |
                        
arr[i] = 5 = num  
                        |               |   
 <=区      [3, 0, 3, 4, | 5, 5, 2, 6, 9, | 6]     > 区   
                        |       i位置，指向2  |

arr[i] = 2 < num  
                           |            |   
 <=区      [3, 0, 3, 4, 2, | 5, 5, 6, 9, | 6]     > 区   
                          |        i位置，指向6

arr[i] = 6 > num  
                           |         |   
 <=区      [3, 0, 3, 4, 2, | 5, 5, 9, | 6, 6]     > 区   
                          |        i位置，指向9

arr[i] = 9 > num, 自己跟自己换，大于区左扩，大于区域和i相等时停止
                           |      |   
 <=区      [3, 0, 3, 4, 2, | 5, 5,| 9, 6, 6]     > 区   
                          |         i位置，指向9
"""
```

### 不改进的快速排序

- 1）把数组范围中的最后一个数作为划分值，然后把数组通过荷兰国旗问题分成三个部分：左侧<划分值、中间==划分值、右侧>划分值
- 2）对左侧范围和右侧范围，递归执行

- 分析
    - 1）划分值越靠近两侧，复杂度越高；划分值越靠近中间，复杂度越低
    - 2）可以轻而易举的举出最差的例子，所以不改进的快速排序时间复杂度为`O(N^2)`












