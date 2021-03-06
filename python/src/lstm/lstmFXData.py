import os
import warnings
import numpy as np
import src.mylib.mcalc as mcalc
import src.mylib.mlstm as mlstm
import tensorflow as tf
import theano.d3viz as d3v
from sklearn.preprocessing import normalize
from sklearn.model_selection import KFold
from keras.utils.visualize_util import plot
from keras import backend as K
from keras import callbacks

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings

tb = callbacks.TensorBoard(log_dir='../log', histogram_freq=1, write_graph=True, write_images=True)

loaded = False
modelSaved = "../model/JPYRMSPropLinear12x6xD6.h5"
graphSaved = "../model/JPYRMSPropLinear12x6xD6.html"

batch_size = 1000
epochs_num = 1
output_dim = 6

np.random.seed(6) # fix random seed

ds = mlstm.loadFXData('JPY=X', '../db/forex.db', 1000)
ds = ds[['Close']]

P = mcalc.m_pct(ds, True)

T = mcalc.vector_delay_embed(P, output_dim, 1)

X, y = mcalc.split_x_y(T)

def mshape(X):
    # reshape input to be [samples, time steps, features]
    return np.reshape(X, (X.shape[0],  -1, X.shape[1])) 

kf = KFold(n_splits=3, shuffle=False, random_state=None)

for train_index, test_index in kf.split(X):
    trainX, testX = X[train_index], X[test_index]
    trainY, testY = y[train_index], y[test_index]
    
    #trainX = normalize(trainX, norm='l2')
    #trainY = normalize(trainY, norm='l2')
    #testX  = normalize(testX,  norm='l2')
    #testY  = normalize(testY,  norm='l2')
    
    trainX = mshape(trainX)
    # create and fit the LSTM network
    if(loaded == False):
        model = mlstm.loadModel(modelSaved, batch_size, trainX.shape, loss='mean_squared_error', 
                      opt="RMSProp", stack=1, state=False, od=output_dim, act='linear', neurons=12)
        loaded = True

    mlstm.trainModel(model, trainX, trainY, batch_size, epochs_num, modelSaved, validation=(mshape(testX), testY), cb=[tb])

layer = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])
print(layer)
#plot(model, to_file=graphSaved)
#d3v.d3viz(get_last_layer_output, graphSaved)
tfGraph = K.get_session().graph
print(tfGraph)

# make predictions
predict = model.predict(mshape(X), batch_size=batch_size)
#print(predict.shape, predict[:,-1])
mlstm.plot_results(predict[:,-1], y[:,-1])