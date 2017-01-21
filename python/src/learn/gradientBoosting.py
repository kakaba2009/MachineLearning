import src.lib.mfile as mfile
import src.lib.mcalc as mcalc
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

lag = 100
dim = 7

dataset = mfile.loadOneSymbol("JPY=X", "../db/forex.db")
dataset = dataset[-1000:]
dataset = dataset[['High']]
#dataset = m_pct(dataset, True)

T = mcalc.vector_delay_embed(dataset, dim, lag)
X, y = mcalc.split_x_y(T)
y = y[ : , 0 ] #Get 1st one for 2nd dimension

X_train, X_test, y_train, y_test = mcalc.splitTrainTest(X, y, ratio=0.80)

clf = ensemble.GradientBoostingRegressor(loss='ls',n_estimators=500,max_depth=30,min_samples_split=2,learning_rate=0.01)

clf.fit(X_train, y_train)
gb = clf.predict(X_test)
mse = mean_squared_error(y_test, gb)
print("MSE: %.4f" % mse)
score = clf.score(X, y)
print("score:", score)
# Plot the results
X = X[ : , 0 ] #Get 1st one for 2nd dimension
plt.scatter(X, y, color='darkorange', label='data')
plt.hold('on')
plt.plot(y_test, gb, color='navy', lw=2, label='score='+str(score))
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()