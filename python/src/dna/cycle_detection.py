import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import src.mylib.mfile as mfile
import src.mylib.mcalc as mcalc
import src.mylib.iching as iching

start= 12

dat = mfile.loadOneSymbol("CHRIS/CME_GC1", "../db/Metals.db", 'weekly')
dat = dat.reset_index()
dat = dat[['Close']]
#dat = mcalc.m_pct(dat, False)
dat = dat.iloc[start:96+start]

#fn = lambda x: (1.0 if x > 0.0 else 0.0)
#xx = dat['Close'].apply(fn)
xx = dat['Close']
yy = xx.copy()
yy[ (yy.index+start+1) % 16 != 0 ] = np.nan

print(xx)
plt.plot(xx.index, xx, color='blue')
plt.scatter(yy.index, yy, s=50, edgecolors='none')
plt.show()
