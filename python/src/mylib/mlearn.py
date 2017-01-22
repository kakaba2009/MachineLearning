import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import *
from sklearn.manifold import *
from sklearn import neighbors
from sklearn import metrics
import src.mylib.mcalc as mcalc
# Scale and visualize the embedding vectors
def plot_embedding(X):    
    f, ax = plt.subplots()
    
    Set1 = plt.get_cmap("cool")
    
    for i in range(X.shape[0]-1):
        ax.scatter(X[i,0], X[i,1], color=Set1(i))
        #ax.arrow(X[i,0], X[i,1], X[i+1,0]-X[i,0], X[i+1,1]-X[i,1], color=plt.cm.Set1(i))
        ax.text(X[i,0], X[i,1], str(i), color=Set1(i))
        plt.pause(0.001)
        
def calc_tree_dist(X, tree, func=np.mean):
    mete = []
    
    for i in range(X.shape[0]):
        R = X[i]
        R = R.reshape(1,-1) #it contains a single sample
        dist, ind = tree.query(R, k=2)     
        #print(ind[0,1], dist[0,1])
        mete.append(dist[0,1])
        
    m = func(mete)
    return m

def calc_best_dimension(df, n_neighbors=5):
    lag = 1
    dim = 1 #best knn score is dim = 3 or 4, for percent change, dim = 6 or 7 or 8, yearly change dim = 2
    tim = []
    scr = []
    while dim <= 10:
        X  = mcalc.vector_delay_embed(df, dim, lag)
        T  = X[-1:,  ]
        A  = X[:-1,  ]
        B  = X[1:,   ]
        knn = neighbors.KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance', algorithm='ball_tree', p=2)
        #y_  = knn.fit(A[:-1,], B[:-1,]).predict(T)
        cv1 = KFold(n_splits=2, shuffle=False)
        res = cross_val_score(knn,A,B,cv=cv1)
        print(dim, '=>', res.mean())
        tim.append(dim)
        scr.append(res.mean())
        dim += 1
        
    max = np.argmax(scr)
        
    return tim[max]