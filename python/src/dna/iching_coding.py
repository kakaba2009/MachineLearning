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
def dateToDouble(d):
    return (d.year * 10000.0 + d.month * 100.0 + d.day) / 10000.0
#df.index = df.index.map(dateToDouble)
df = df[['Close']] #return data frame
df = df.head(100)
df = df.diff()
#df = df.dropna()
fn = lambda x: (1.0 if x > 0.0 else 0.0)
xx = df['Close'].apply(fn)
L0 = xx
print(L0)
L1 = xx.shift(-1)
print(L1)
L2 = xx.shift(-2)
L3 = xx.shift(-3)
L4 = xx.shift(-4)
L5 = xx.shift(-5)
yy = L0 * 32 + L1 * 16 + L2 * 8 + L3 * 4 + L4 * 2 + L5
#zz = np.copy(yy)
#zz[zz != stop] = np.nan
style.use('ggplot')
plt.scatter(yy.index, yy)
print(yy)
plt.show()
