## Pandas 连接合并追加操作

### concat

```python
concat(objs, axis=0, join='outer',
       ignore_index: bool = False, keys = None, levels = None, names = None, verify_integrity: bool = False, sort: bool = False, copy: bool = True)
```

- 连接 2 个`Series`
    ```python
    s1 = pd.DataFrame([1,2,3])
    s2 = pd.DataFrame([4,6,5])
  
    s1
       0
    0  1
    1  2
    2  3
  
    s2
       0
    0  4
    1  6
    2  5
  
    pd.concat([s1, s2])
       0
    0  1
    1  2
    2  3
    0  4
    1  6
    2  5
  
    # 忽略索引
    pd.concat([s1, s2], ignore_index=True)
       0
    0  1
    1  2
    2  3
    3  4
    4  6
    5  5
    ```
- 连接 2个 `DataFrame`
  ```python
  # 普通连接(行)
  df1 = pd.DataFrame([['a', 1], ['b', 2]], columns=['letter', 'number'])
  df2 = pd.DataFrame([['c', 3], ['d', 4]], columns=['letter', 'number'])
  
  df1
    letter  number
  0      a       1
  1      b       2
  
  df2
    letter  number
  0      c       3
  1      d       4
  
  pd.concat([df1, df2])
    letter  number
  0      a       1
  1      b       2
  0      c       3
  1      d       4
  
  # 普通连接(列)
  pd.concat([df1, df2], axis=1)
    letter  number letter  number
  0      a       1      c       3
  1      b       2      d       4
  
  # 如果字段不相同，填充 `Nan`
  df3 = pd.DataFrame([['c', 3, 'cat'], ['d', 4, 'dog']], columns=['letter', 'number', 'animal'])
  
  df3
    letter  number animal
  0      c       3    cat
  1      d       4    dog
  
  pd.concat([df1, df3], sort=False)
    letter  number animal
  0      a       1    NaN
  1      b       2    NaN
  0      c       3    cat
  1      d       4    dog
  
  # 内连接(只连接相同字段)
  pd.concat([df1, df3], join='inner')
    letter  number
  0      a       1
  1      b       2
  0      c       3
  1      d       4
  
  # 排序后，列拼接
  pd.concat([df1, df2], axis=1)
    letter  number letter  number
  0      a       1      c       3
  1      b       2      d       4
  
  df2.sort_values('number', ascending=False, inplace=True)
  # 这一步至关重要
  df2.reset_index(drop=True, inplace=True)
  pd.concat([df1, df2], axis=1)
    letter  number letter  number
  0      a       1      d       4
  1      b       2      c       3
  ```

### merge

```python
merge(left, right, how: str = 'inner', on = None, left_on = None, right_on = None, left_index: bool = False,
        right_index: bool = False, sort: bool = False, suffixes = ('_x', '_y'), copy: bool = True, 
        indicator: bool = False, validate = None)
```

- 普通合并
  ```python
  df1 = pd.DataFrame({'lkey': ['foo', 'bar', 'baz', 'foo'], 'value': [1, 2, 3, 5]})
  df2 = pd.DataFrame({'rkey': ['foo', 'bar', 'baz', 'foo'], 'value': [5, 6, 7, 8]})
  
  df1
    lkey  value
  0  foo      1
  1  bar      2
  2  baz      3
  3  foo      5
  
  df2
    rkey  value
  0  foo      5
  1  bar      6
  2  baz      7
  3  foo      8
  
  df1.merge(df2, left_on='lkey', right_on='rkey')
    lkey  value_x rkey  value_y
  0  foo        1  foo        5
  1  foo        1  foo        8
  2  foo        5  foo        5
  3  foo        5  foo        8
  4  bar        2  bar        6
  5  baz        3  baz        7
  
  df1.merge(df2, left_on='lkey', right_on='rkey',suffixes=('_left', '_right'))
    lkey  value_left rkey  value_right
  0  foo           1  foo            5
  1  foo           1  foo            8
  2  foo           5  foo            5
  3  foo           5  foo            8
  4  bar           2  bar            6
  5  baz           3  baz            7
  ```

- 内、左、右连接
  ```python
  df1 = pd.DataFrame({'a': ['foo', 'bar'], 'b': [1, 2]})
  df2 = pd.DataFrame({'a': ['foo', 'baz'], 'c': [3, 4]})
  
  df1
       a  b
  0  foo  1
  1  bar  2
  
  df2
       a  c
  0  foo  3
  1  baz  4
  
  df1.merge(df2, how='inner', on='a')
       a  b  c
  0  foo  1  3
  
  df1.merge(df2, how='left', on='a')
       a  b    c
  0  foo  1  3.0
  1  bar  2  NaN
  
  df1.merge(df2, how='right', on='a')
       a    b  c
  0  foo  1.0  3
  1  baz  NaN  4
  # 相同值，但是不同的字段名，左连接
  df3 = pd.DataFrame({'d': ['foo', 'baz'], 'c': [3, 4]})
  df3
       d  c  
  0  foo  3
  1  baz  4
  
  df1.merge(df3, how='left', left_on='a', right_on='d')
       a  b    d    c
  0  foo  1  foo  3.0
  1  bar  2  NaN  NaN
  ```
- 笛卡尔积
  ```python
  df1 = pd.DataFrame({'left': ['foo', 'bar']})
  df2 = pd.DataFrame({'right': [7, 8]})
  
  df1
    left
  0  foo
  1  bar
  
  df2
     right
  0      7
  1      8
  
  df1.merge(df2, how='cross')
    left  right
  0  foo      7
  1  foo      8
  2  bar      7
  3  bar      8
  ```

### append

```python
append(other, ignore_index=False, verify_integrity=False, sort=False)
```

- 普通追加
  ```python
  df = pd.DataFrame([[1, 2], [3, 4]], columns=list('AB'))
  
  df
     A  B
  0  1  2
  1  3  4
  
  df2 = pd.DataFrame([[5, 6], [7, 8]], columns=list('AB'))
  
  df2
     A  B
  0  5  6
  1  7  8
  
  df.append(df2)
     A  B
  0  1  2
  1  3  4
  0  5  6
  1  7  8
  
  df.append(df2, ignore_index=True)
     A  B
  0  1  2
  1  3  4
  2  5  6
  3  7  8
  # 通过 for 循环追加
  df = pd.DataFrame(columns=['A'])
  for i in range(5):
      df = df.append({'A': i}, ignore_index=True)
  df
     A
  0  0
  1  1
  2  2
  3  3
  4  4
  # 等价于 concat 连接，如下
  pd.concat([pd.DataFrame([i], columns=['A']) for i in range(5)],ignore_index=True)
     A
  0  0
  1  1
  2  2
  3  3
  4  4
  ```