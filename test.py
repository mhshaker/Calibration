import numpy as np
x = np.array([1,2,3,4,4,5])
y = np.array([1,2,30,4,40,5])

c = np.where(x == y)
same = x[c[0]]

print(c)

print(same)