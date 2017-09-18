import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import src.mylib.mfile as mfile
import src.mylib.mcalc as mcalc
import src.mylib.iching as iching

#pattern = [19.,  38.,  12.,  24.,  49.,  35.,  6.,  12.,  25.,  50.,  36., 9.,  19.,  39.,  15.,  30.,  60.]
psize   = 3

dat = mfile.loadOneSymbol("CHRIS/CME_GC1", "../db/Metals.db", 'Metal')
ydt = mcalc.m_sample_y(dat)

yy = iching.getHexgram(ydt, 'Last')
#iching.plot(yy, ydt, 'Last', 0.025)
pattern = yy.dropna().values
length  = len(pattern)

mdt = mcalc.m_sample_w(dat)
mmm = iching.getHexgram(mdt, 'Last')

for i in range(0, length-psize+1, 1):
    pp = pattern[i:i+psize]
    print(pp)
    p2  = mmm.isin(pp)
    rrr = mmm[p2]
    plt.plot(rrr.index, rrr, 'o')
    #iching.searchPattern(mmm, pattern, length, psize)

plt.show()
