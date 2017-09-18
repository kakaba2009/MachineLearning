import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import src.mylib.mfile as mfile
import src.mylib.mcalc as mcalc
import src.mylib.iching as iching

#pattern = [19.,  38.,  12.,  24.,  49.,  35.,  6.,  12.,  25.,  50.,  36., 9.,  19.,  39.,  15.,  30.,  60.]
psize   = 9

dat = mfile.loadOneSymbol("JPY=X", "../db/forex.db")

yy = iching.getHexgram(mcalc.m_sample_y(dat))
pattern = yy.dropna().values
num     = len(pattern)

#dat = mcalc.m_sample_w(dat)
yy = iching.getHexgram(dat)
#iching.plot(y1, dat)

def fMatch(a, **kwargs):
    for i in range(0, num-psize+1, 1):
        p1 = pattern[i:i+psize]
        rv = np.sqrt(np.sum((a - p1)**2))
        if rv == 0:
            print(a)
            return 0
    return 1

rv = yy.rolling(psize).apply(fMatch)
print(rv.values)
