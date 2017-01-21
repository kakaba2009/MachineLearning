#LSTM for problem with window regression framing
import numpy
import pandas
import math
import src.lib.mfile  as mfile
import src.lib.mcalc  as mcalc
import src.lib.mlstm  as mlstm
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

batch_size = 1000
look_back  = 6
output_dim = 6

numpy.random.seed(7) # fix random seed for reproducibility

LOG     = False
dataset = mlstm.loadFXData('JPY=X', '../db/forex.db', 1000)
dataset = dataset[['High']]
dataset = mcalc.m_pct(dataset, True)
dataset = dataset.values
dataset = dataset.astype('float32')    
SCALE   = 1 #np.amax(dataset.flatten())
print("SCALE=>", SCALE)

print(dataset[-look_back:])

dataset = mlstm.normalize(dataset, SCALE, LOG)

trainX, trainY, testX, testY = mlstm.setupTrainTest(dataset, look_back, SCALE, LOG)
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], -1, trainX.shape[1]))
testX1 = numpy.reshape(testX,  (testX.shape[0],  -1, testX.shape[1]))
# create and fit the LSTM network
modelSaved = "../model/JPYRMSPropLinear12x6xD6.h5"
model = mlstm.loadModel(modelSaved, batch_size, trainX.shape, loss='mean_squared_error', 
                  opt="RMSProp", stack=1, state=False, od=output_dim, act='linear', neurons=12)

mlstm.trainModel(model, trainX, trainY, batch_size, 5, modelSaved)

# make predictions
trainPredict = model.predict(trainX, batch_size=batch_size)
#model.reset_states()
testPredict  = model.predict(testX1, batch_size=batch_size)
#model.reset_states()
lastX,lastX1 = mlstm.lastInputSeq(dataset, look_back)
lastPredict  = model.predict(lastX1,  batch_size=batch_size)
#model.reset_states()

mlstm.printScore(model, trainX, trainY)
mlstm.printScore(model, testX1,  testY)
#inverse scale
dataset      = mlstm.inverse_transform(dataset, SCALE, LOG)
#print(dataset[-look_back:])

testX        = mlstm.inverse_transform(testX, SCALE, LOG)
lastX        = mlstm.inverse_transform(lastX, SCALE, LOG)
testPredict  = mlstm.inverse_transform(testPredict,  SCALE, LOG)
lastPredict  = mlstm.inverse_transform(lastPredict,  SCALE, LOG)
trainPredict = mlstm.inverse_transform(trainPredict, SCALE, LOG)
mlstm.printX_Y(testX, testPredict)

print("Tomorrow:", lastX1, "=>", lastPredict)

X, Y = mlstm.create_dataset(dataset, look_back)
mlstm.plot_result_2F(X, trainPredict, testPredict, look_back)