import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import src.mylib.mfile as mfile
import src.mylib.mcalc as mcalc

def getPivit(datStr, feq):
    dat = mfile.loadOneSymbol("CHRIS/CME_GC1", "../db/Metals.db", feq)
    dat = dat[dat.index > datStr]
    piv = dat['Close'].min()
    idx = dat['Close'].idxmin()
    print(idx, piv)
    return piv

def drawReversal(ax, y0, L):
    a = 0.00
    s = 30.0
    y = y0
    for i in range(30):
        yy = (np.sqrt(y) + a / 180.0) ** 2
        ax.plot([0, L], [yy, yy], 'k-')
        a += s
        print(yy)
