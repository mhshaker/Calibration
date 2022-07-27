import numpy as np


x = np.array([1,2,np.NaN, 0])
y = np.nan_to_num(x)
print(x)
print(y)