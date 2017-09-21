import src.mylib.mfile as mfile
import src.mylib.mcalc as mcalc
import src.mylib.iching as iching
import matplotlib.pyplot as plt
import numpy as np

dat = mfile.loadOneSymbol("JPY=X", "../db/forex.db", 'FX')
#ydt = mcalc.m_sample_y(dat)
dat = dat[['Close']]
P = mcalc.m_pct(dat, False)
for i in range(1, 10):
    x = P['Close'][i]
    F = P[ np.abs(P['Close'] - x) < 0.0001 ]
    #T = mcalc.vector_delay_embed(P, 6, 1)
    #plt.scatter(P.index, P, s=10, edgecolors='none')
    plt.scatter(F.index, F, s=10, edgecolors='none')
"""
period = 0.5

f = np.arange(len(yy.values))  # Data range
z = yy.values  # Data

a = f*np.sin(period*f)
b = f*np.cos(period*f)

fig = plt.figure()
ax = plt.subplot(111)
fig.add_subplot(ax)
ax.scatter(a, b, c=z, s=10, edgecolors='none')
"""
plt.show()
