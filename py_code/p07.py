# -*- coding: utf-8 -*-
import numpy as np
print(np.__version__)

#array zeros ones empty arange linspace logspace random

a = np.array([[2, 3, 4], [5, 6, 7]], dtype=np.float)
print(a)
print(a.dtype)

b = np.zeros((3, 4, 2), dtype=np.int)
print(b)

c = np.ones((2, 3))
print(c)

d = np.empty((3, 3))
print(d)

#随机数
print(np.arange(1, 10, 2))
print(np.array(range(1, 10, 2)))

#等差数列
e = np.linspace(1, 10, 7, endpoint=False)
print(e)
print(np.linspace(1, 10, 8))

#等比数列
f = np.logspace(1, 8, 4, endpoint=False)
print(f)
print(np.logspace(1, 8, 5))


#[0, 1)
g = np.random.random((2, 3, 4))
print('g:', g)
print('属性：', g.dtype, g.shape, g.size, g.ndim)
print(np.random.randint(10, 20, (2, 3, 4)))
