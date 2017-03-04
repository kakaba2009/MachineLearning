import numpy as np
import src.mylib.mcalc as mcalc
import src.mylib.mlstm as mlstm
from sklearn.preprocessing import normalize
from sklearn.model_selection import KFold

loaded = False
modelSaved = "../model/JPYRMSPropLinear12x6xD6.h5"

batch_size = 1000
epochs_num = 1

lag = 1
dim = seqn_size = 6
output_dim = 6

np.random.seed(6) # fix random seed for reproducibility

dataset = mlstm.loadFXData('JPY=X', '../db/forex.db', 1000)
dataset = dataset[['Close']]
dataset = mcalc.m_pct(dataset, True)
SCALE   = np.amax(dataset.values.flatten())

T = mcalc.vector_delay_embed(dataset, dim, lag)

X, y = mcalc.split_x_y(T)

def mshape(X):
    return np.reshape(X, (X.shape[0],  -1, X.shape[1])) 

kf = KFold(n_splits=2, shuffle=False, random_state=None)

for train_index, test_index in kf.split(X):
    trainX, testX = X[train_index], X[test_index]
    trainY, testY = y[train_index], y[test_index]
    
    trainX = normalize(trainX, norm='l2')
    trainY = normalize(trainY, norm='l2')
    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], -1, trainX.shape[1]))
    # create and fit the LSTM network
    if(loaded == False):
        model = mlstm.loadModel(modelSaved, batch_size, trainX.shape, loss='mean_squared_error', 
                      opt="RMSProp", stack=1, state=False, od=output_dim, act='linear', neurons=12)
        loaded = True

    mlstm.trainModel(model, trainX, trainY, batch_size, epochs_num, modelSaved)

# make predictions
predict = model.predict(mshape(X), batch_size=batch_size)

mlstm.plot_result_2F(y, predict, None, seqn_size)