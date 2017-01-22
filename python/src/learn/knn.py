from sklearn.model_selection import *
from sklearn.manifold import *
from sklearn import neighbors
from sklearn import metrics
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import src.mylib.mfile as mfile
import src.mylib.mcalc as mcalc
import src.mylib.mplot as mplot
import src.mylib.mlearn as mlearn

df = mfile.loadOneSymbol("JPY=X", "../db/forex.db")
df = mcalc.m_sample_w(df)
df = df[-250:]
df = df[['High']]
df = mcalc.m_pct(df, dropna=True)
n_neighbors = 5
lag = 1
dim = 7 #best knn score is dim = 3 or 4
tim = []
pre = []
if lag <= 31:
    X  = mcalc.vector_delay_embed(df, dim, lag)
    T  = X[-1:,  ]
    A  = X[:-1,  ]
    B  = X[1:,   ]
    #nbr = neighbors.NearestNeighbors(n_neighbors=n_neighbors, algorithm='kd_tree')
    #nbr.fit(X)
    #distances, indices = nbr.kneighbors(X)
    clf = Isomap(n_neighbors, n_components=2)
    #clf = LocallyLinearEmbedding(n_neighbors, n_components=2,method='standard')
    #clf = LocallyLinearEmbedding(n_neighbors, n_components=2,method='modified')
    #clf = MDS(n_components=2)
    #clf = SpectralEmbedding(n_components=2, eigen_solver="arpack")
    #clf = TSNE(n_components=2, init='pca')
    lle = clf.fit_transform(X)
    #print("Done. Reconstruction error: %g" % clf.reconstruction_error())
    mlearn.plot_embedding(lle)
    #f, ax = plt.subplots()
    #ax.scatter(lle[:,0], lle[:,1])
    #ax.set_title("lag=" + str(lag))
    lag += 10
    
#plt.grid(True)
#plt.axis('tight')
#plt.legend()
#plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors, 'distance'))
plt.show()