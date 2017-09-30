import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import src.mylib.mfile as mfile
import src.mylib.mcalc as mcalc
from matplotlib.finance import candlestick2_ohlc

dat = mfile.loadOneSymbol("JPY=X", "../db/forex.db", "FX")
dat = dat[dat.index > '2011-01-01']
dat = dat.reset_index()
piv = dat['Close'].min()
x0 = dat['Close'].idxmin()
print(x0, piv)
y0 = piv
print(y0)
piv = dat['Close'].max()
x1  = dat['Close'].idxmax()
print(x1, piv)
y1 = piv
print(y1)
fig, ax = plt.subplots()
candlestick2_ohlc(ax, dat['Open'], dat['High'], dat['Low'], dat['Close'], width=0.6)
a = 0.00
s = 15
L = len(dat['Close'])
y = y0
for i in range(31):
    yy = (np.sqrt(y) + a / 180.0)**2
    ax.plot([0, L], [yy, yy], 'blue')
    a += s

P = y1
t = np.sqrt(P)
print(t)
ax.plot([x0, x0, x1, x1, x0], [y0, y1, y1, y0, y0], 'green')

tt = x0
for i in range(20):
    ax.plot([tt, tt], [y0, y1], 'yellow')
    tt += t
    #print(tt)

plt.show()
