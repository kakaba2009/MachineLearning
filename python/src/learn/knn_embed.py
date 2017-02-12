from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.manifold import *
from sklearn import neighbors
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import src.mylib.mfile as mfile
import src.mylib.mcalc as mcalc
import src.mylib.mplot as mplot
import src.mylib.mlearn as mlearn

df = mfile.loadOneSymbol("JPY=X", "../db/forex.db")
#df = mcalc.m_sample_y(df)
df = df[-50:]
df = mcalc.computeTimeDiff(df)
df = df[['TimeDiff', 'Open', 'Close']]
df = df.values
#df = mcalc.m_pct(df, dropna=True)
lag = 1
dim = 3 #best knn score is dim = 3 or 4, for percent change, dim = 6 or 7 or 8, yearly change dim = 2
n_neighbors = dim + 1
tim = []
pre = []
while dim <= 10:
    X  = df #mcalc.vector_delay_embed(df, dim, lag)
    T  = X[-1:,  ]
    A  = X[:-1,  ]
    B  = X[1:,   ]
    #nbr = neighbors.NearestNeighbors(n_neighbors=n_neighbors, algorithm='kd_tree')
    #nbr.fit(X)
    #distances, indices = nbr.kneighbors(X)
    knn = neighbors.KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance', algorithm='ball_tree', 
                                        p=2, metric='pyfunc', metric_params={'func':mcalc.hyperdist})
    y_  = knn.fit(A[:-1,], B[:-1,]).predict(T)
    cv1 = KFold(n_splits=2, shuffle=False)
    res = cross_val_score(knn,A,B,cv=cv1)
    print(dim, '=>', y_[0,0], res.mean())
    tim.append(dim)
    pre.append(res.mean())
    dim += 1
    
f, ax = plt.subplots()
ax.plot(tim, pre)
plt.grid(True)
plt.show()