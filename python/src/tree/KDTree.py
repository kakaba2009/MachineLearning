import matplotlib.pyplot as plt
import sklearn.neighbors as nb
import src.lib.mfile  as mfile
import src.lib.mcalc  as mcalc
import src.lib.mplot  as mplot
import src.lib.mlearn as mlearn
import numpy as np
import pickle

np.random.seed(0)

df = mfile.loadOneSymbol("JPY=X", "../db/forex.db")

df = df[-9000:]
df = df[['High']]
df = mcalc.m_pct(df, True)

X = mcalc.vector_delay_embed(df, 4, 32)

print(X.shape)

tree = nb.KDTree(X, leaf_size=32)

mete = []

for i in range(X.shape[0]):
    R = X[i]
    R = R.reshape(1,-1) #it contains a single sample
    dist, ind = tree.query(R, k=2)     
    #print(ind[0,1], dist[0,1])  # indices of 3 closest neighbors
    mete.append(dist[0,1])
    
m = np.mean(mete)
print(m)