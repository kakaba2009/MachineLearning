import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import src.mylib.mfile as mfile
import src.mylib.mcalc as mcalc
from matplotlib.finance import candlestick2_ohlc

dat = mfile.loadOneSymbol("CHRIS/CME_GC1", "../db/Metals.db", 'monthly')
dat = dat[dat.index > '2015-01-01']
y0  = dat['Close'].min()
idx = dat['Close'].idxmin()
print(idx, y0)
fig, ax = plt.subplots()
candlestick2_ohlc(ax, dat['Open'], dat['High'], dat['Low'], dat['Close'], width=0.6)
a = 0.00
s = 30.0
L = len(dat['Close'])
y = y0
for i in range(30):
    yy = (np.sqrt(y) + a / 180.0)**2
    ax.plot([0, L], [yy, yy], 'k-')
    a += s
    print(yy)

plt.show()
