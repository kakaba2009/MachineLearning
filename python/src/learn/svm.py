import numpy
import pandas
from sklearn.svm import SVR
import src.mylib.mfile as mfile
import src.mylib.mcalc as mcalc
import src.mylib.mplot as mplot
import src.mylib.mlearn as mlearn
import matplotlib.pyplot as plt

lag = 60
dim = 7

dataset = mfile.loadOneSymbol("JPY=X", "../db/forex.db")
dataset = dataset[-1000:]
dataset = dataset[['High']]
#dataset = m_pct(dataset, True)

T = mcalc.vector_delay_embed(dataset, dim, lag)
X, y = mcalc.split_x_y(T)

y = y[ : , 0 ] #Get 1st one for 2nd dimension

svr_rbf  = SVR(kernel='rbf',    C=1.0, gamma=0.1) # gaussian kernel
svr_lin  = SVR(kernel='linear', C=1.0)
y_rbf  = svr_rbf.fit(X, y).predict(X)
y_lin  = svr_lin.fit(X, y).predict(X)
s_rbf  = svr_rbf.score(X, y)
s_lin  = svr_lin.score(X, y)
print("rbf score:", s_rbf)
print("lin score:", s_lin)
# Plot the results
lw = 2
X  = X[ : , 0 ] #Get 1st one for 2nd dimension
plt.scatter(X, y, color='darkorange', label='data')
plt.hold('on')
plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()