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

df = df[-1000:]
df = df[['High']]
df = mcalc.m_pct(df, True)

d = mlearn.calc_best_dimension(df, n_neighbors=5)
print("Dimension->", d)
X = mcalc.vector_delay_embed(df, d, 1)

print(X.shape)

clrs = ["b","g","r","c","m","y","k"]
tree = nb.BallTree(X, leaf_size=32)
rads = mlearn.calc_tree_dist(X, tree, np.min) * 5.0
print("radius->", rads)
f, ax = plt.subplots()
for i in range(X.shape[0]):
    R = X[i]
    R = R.reshape(1,-1) #it contains a single sample
    
    ind = tree.query_radius(R, r=rads)
    if(len(ind) >= 1):
        t2 = ind[0]
        if(t2[0] == i): #index 0 is always the same as i
            t2 = t2[1:]
        t1 = np.repeat(np.array([i]), t2.shape[0])
        if(len(t1) >= 1):
            pr = np.abs(t2 - t1)
            cr = clrs[i % 7]
            ax.scatter(t1, t2, color = cr)
            #ax.scatter(i, np.average(pr), color = cr)
            #ax.text(t1[0], t2[0], str(pr), color=cr)
            #plt.pause(0.001)
        
plt.grid(True)
plt.show()