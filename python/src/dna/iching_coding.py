import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import src.mylib.mfile as mfile
import src.mylib.mcalc as mcalc
import src.mylib.iching as iching

#pattern = [19.,  38.,  12.,  24.,  49.,  35.,  6.,  12.,  25.,  50.,  36., 9.,  19.,  39.,  15.,  30.,  60.]
psize   = 9

dat = mfile.loadOneSymbol("AAPL", "../db/Dow.db", 'Stocks')
ydt = mcalc.m_sample_y(dat)

yy = iching.getHexgram(ydt)
iching.plot(yy, ydt, 1)
pattern = yy.dropna().values
length  = len(pattern)

#dat = mcalc.m_sample_w(dat)
yy = iching.getHexgram(dat)

iching.searchPattern(yy, pattern, length, psize)
