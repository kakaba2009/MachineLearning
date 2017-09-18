import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

stop =int('101010', 2) #101010 I Ching 63 After Completion
befo =int('010101', 2) #101010 I Ching 64 Before Completion

guai =int('111110', 2) #101010 I Ching 43

qian =int('111111', 2) #101010 I Ching 01
kun  =int('000000', 2) #101010 I Ching 02

def dateToDouble(d):
    return (d.year * 10000.0 + d.month * 100.0 + d.day) / 10000.0

def getHexgram(dat, p = False):
    # df.index = df.index.map(dateToDouble)
    dat = dat[['Close']]  # return data frame
    # dat = dat.head(200)
    df = dat.diff()
    # df = df.dropna()
    fn = lambda x: (1.0 if x > 0.0 else 0.0)
    xx = df['Close'].apply(fn)
    L0 = xx
    #print(L0)
    L1 = xx.shift(-1)
    #print(L1)
    L2 = xx.shift(-2)
    L3 = xx.shift(-3)
    L4 = xx.shift(-4)
    L5 = xx.shift(-5)
    yy = L0 * 32 + L1 * 16 + L2 * 8 + L3 * 4 + L4 * 2 + L5
    if p:
        print(yy.values)
    return yy

def plot(yy, dat, ratio = 0.5):
    zz = yy.copy()
    zz[zz != stop] = np.nan
    style.use('ggplot')
    plt.plot(yy.index, yy)
    plt.plot(zz.index, zz, "bo")
    plt.plot(dat.index, dat['Close'] * ratio)
    plt.show()

def searchPattern(yy, pattern, length, psize):
    def fMatch(a, **kwargs):
        for i in range(0, length - psize + 1, 1):
            p1 = pattern[i:i + psize]
            rv = np.sqrt(np.sum((a - p1) ** 2))
            if rv == 0:
                print(a)
                return 0
        return 1

    rv = yy.rolling(psize).apply(fMatch)
    print(rv.values)
