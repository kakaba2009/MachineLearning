import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import src.mylib.mfile as mfile
import src.mylib.reversal as reversal
from matplotlib.finance import candlestick2_ohlc

dat = mfile.loadOneSymbol("CHRIS/CME_GC1", "../db/Metals.db", 'weekly')
flt = '2011-01-01'
dat = dat[dat.index > flt]
L   = len(dat['Close'])

fig, ax = plt.subplots()
candlestick2_ohlc(ax, dat['Open'], dat['High'], dat['Low'], dat['Close'], width=0.6)

#d0  = reversal.getPivit(flt, 'daily', 'h')
w0  = reversal.getPivit(flt, 'weekly', 'h')
m0  = reversal.getPivit(flt, 'monthly', 'h')
#q0  = reversal.getPivit(flt, 'quarterly', 'h')
#y0  = reversal.getPivit(flt, 'annual', 'h')

#reversal.drawReversal(ax, d0, L, 90.0, 20, -1.0)
reversal.drawReversal(ax, w0, L, 90.0, 20, -1.0)
#reversal.drawReversal(ax, m0, L, 90.0, 20, -1.0)
#reversal.drawReversal(ax, q0, L, 90.0, 20, -1.0)
#reversal.drawReversal(ax, y0, L, 90.0, 20, -1.0)

plt.show()
