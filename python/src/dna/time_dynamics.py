import numpy as np
import pandas as pd
import src.mylib.mfile as mfile
import src.mylib.mcalc as mcalc
import matplotlib.pyplot as plt

dat = mfile.loadOneSymbol("JPY=X", "../db/forex.db", 'FX')
dat = dat[['Close']]
dat = mcalc.m_pct(dat, True)
#dat = dat.iloc[:1000]
#print(dat)
T = mcalc.vector_delay_embed(dat, 6, 1)
#print(T)

def distance(a, b):
    return np.linalg.norm(a - b)

D = np.array([])

#for i in range(0, len(T)-1, 1):
#    D = np.append(D, np.linalg.norm(T[i] - T[i+1]))

#for i in range(0, len(T)-1, 1):
#    D = np.append(D, np.inner(T[i], T[i+1]))

#for i in range(0, len(T)-1, 1):
#    x = np.inner(T[i], T[i+1])
#    x = x / np.linalg.norm(T[i])
#    x = x / np.linalg.norm(T[i+1])
#    D = np.append(D, x)

#D = np.arccos(D) * (180.0 / np.pi)

#R = D[ D > 90.0 ]

for i in range(0, len(T), 1):
    x = (T[i][0])**2 + (T[i][1])**2 + (T[i][2])**2 + (T[i][3])**2 + (T[i][4])**2 + (T[i][5])**2
    D = np.append(D, x)

D = np.log(D)

plt.scatter(range(len(D)), D, s=10, edgecolors='none')
plt.show()
