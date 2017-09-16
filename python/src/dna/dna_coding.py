import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import src.mylib.mfile as mfile
from matplotlib import style

stop =int('101010', 2) #101010 I Ching 63 After Completion
befo =int('010101', 2) #101010 I Ching 64 Before Completion

guai =int('111110', 2) #101010 I Ching 43

qian =int('111111', 2) #101010 I Ching 01
kun  =int('000000', 2) #101010 I Ching 02

df = mfile.loadOneSymbol("JPY=X", "../db/forex.db")
df = df.reset_index(drop=True)
df = df['Close']
df = df[-1000:]
df = df.diff()
df = df.dropna()
fn = lambda x: (1.0 if x > 0 else 0.0)
xx = df.apply(fn)
xx = xx.values
ln = len(xx)
sz = (ln // 6) * 6
xx = xx[:sz]
print(xx)
#L0 = xx[:-2]
#L1 = xx[1:-1]
#L2 = xx[2:]
#yy = L0 * 4 + L1 * 2 + L2
def my_func(arr, num):
    sum = 0
    for i in range(num):
        sum += arr[i] * (2**(num-i-1))
    return sum
xx = np.reshape(xx, (-1, 6))
yy = np.apply_along_axis(my_func, 1, xx, 6)
i, = np.where(yy == stop)
zz = np.copy(yy)
zz[zz != stop] = np.nan
ss = yy
sp = range(0, len(ss))
style.use('ggplot')
plt.plot(ss)
plt.plot(zz, 'bo')
print(ss)
plt.show()
