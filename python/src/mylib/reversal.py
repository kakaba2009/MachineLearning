import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import src.mylib.mfile as mfile
import src.mylib.mcalc as mcalc

def getPivit(datStr, feq, hl='h'):
    dat = mfile.loadOneSymbol("CHRIS/CME_GC1", "../db/Metals.db", feq)
    dat = dat[dat.index > datStr]
    if hl == 'l':
        piv = dat['Close'].min()
        idx = dat['Close'].idxmin()
    else:
        piv = dat['Close'].max()
        idx = dat['Close'].idxmax()
    print(idx, piv)
    return piv

def drawReversal(ax, y0, L, s=30.0, n=30, hl=1.0):
    a = 0.00
    y = y0
    for i in range(n):
        yy = (np.sqrt(y) + hl * a / 180.0) ** 2
        ax.plot([0, L], [yy, yy], 'k-')
        a += s
        print(yy)
