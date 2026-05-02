# Python 常用小技巧

## 自定义排序：`cmp_to_key`

Python 3 的 `sorted` 只有 `key` 参数，没有 `cmp`。如果需要自定义比较函数：

```python
from functools import cmp_to_key

def compare(a, b):
    if a < b:
        return -1
    elif a > b:
        return 1
    else:
        return 0

sorted([3, 1, 2], key=cmp_to_key(compare))
```

**注意**：`key` 对每个元素调用一次（O(n) 次），`cmp` 对每次比较调用（O(n log n) 次），所以优先用 `key`。

---

## 组合数计算：`math.comb`

Python 3.8+ 内置：

```python
from math import comb

comb(5, 2)   # 10
comb(10, 3)  # 120
```

替代方案：

```python
# scipy（支持大数、浮点数）
from scipy.special import comb
comb(10, 3, exact=True)  # 120

# 手写高效版（避免大阶乘溢出）
def my_comb(n, k):
    if k > n - k:
        k = n - k
    res = 1
    for i in range(k):
        res = res * (n - i) // (i + 1)
    return res
```