import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import src.mylib.mfile as mfile
import src.mylib.mcalc as mcalc
import src.mylib.iching as iching

psize   = 16

dat = mfile.loadOneSymbol("CHRIS/CME_GC1", "../db/Metals.db", 'weekly')
dat = dat[['Close']]
dat = dat.iloc[:32]
print(dat)
plt.plot(dat.index, dat)
plt.show()
