###### datetime:2025/04/18 16:07
###### author:nzb

# MoveMeanFilter(移动平均滤波器)


```python
import numpy as np

class MoveMeanFilter:
    def __init__(self, x0: float, length=15):
        super().__init__()
        self._x = x0 * np.ones(length)

    def filter(self, x):
        self._x[: - 1] = self._x[1:]
        self._x[-1] = x
        return np.sum(self._x) / self._x.size
```