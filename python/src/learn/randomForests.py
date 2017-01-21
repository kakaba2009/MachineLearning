import numpy
import pandas
import src.lib.mfile  as mfile
import src.lib.mcalc  as mcalc
import src.lib.mplot  as mplot
import src.lib.mlearn as mlearn
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

lag = 7
dim = 7

dataset = mfile.loadOneSymbol("JPY=X", "../db/forex.db")
dataset = dataset[-2000:]
dataset = dataset[['High']]
#dataset = m_pct(dataset, True)

T = mcalc.vector_delay_embed(dataset, dim, lag)
X, y = mcalc.split_x_y(T)

X_train, X_test, y_train, y_test = mcalc.splitTrainTest(X, y, ratio=0.80)

max_depth = 30
regr_multirf = MultiOutputRegressor(RandomForestRegressor(max_depth=max_depth, random_state=0))
regr_multirf.fit(X_train, y_train)

regr_rf = RandomForestRegressor(max_depth=max_depth, random_state=2)
regr_rf.fit(X_train, y_train)

# Predict on new data
y_multirf = regr_multirf.predict(X_test)
y_rf = regr_rf.predict(X_test)

# Plot the results
plt.figure()
s = 50 #marker size
a = 0.5
plt.scatter(y_test[:, 0], y_test[:, 1], c="navy", s=s, marker="s", alpha=a, label="Data")

plt.scatter(y_multirf[:, 0], y_multirf[:, 1],
            c="cornflowerblue", s=s, alpha=a,
            label="Multi RF score=%.2f" % regr_multirf.score(X_test, y_test))
plt.scatter(y_rf[:, 0], y_rf[:, 1],
            c="c", s=s, marker="^", alpha=a,
            label="RF score=%.2f" % regr_rf.score(X_test, y_test))
#plt.xlim([-6, 6])
#plt.ylim([-6, 6])
plt.xlabel("target 1")
plt.ylabel("target 2")
plt.title("Comparing random forests and the multi-output meta estimator")
plt.legend()
plt.show()
