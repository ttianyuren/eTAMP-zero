import numpy as np
import math

import random

b = [1, 2]
b[1] = 4
# print(b)

d = {}

d[None] = 4 - np.inf

d[None] = 55 - np.inf


d["a"] = -np.inf

d["b"] = -np.inf + 10000000000000000000000000000000000


# d['d']=100

# d['c']=100


print(list(d.values())[0])
