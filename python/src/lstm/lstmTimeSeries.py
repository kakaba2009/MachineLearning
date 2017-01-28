import os.path
import numpy as np
import pandas as pd
from matplotlib import style
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.layers import Activation

def loadCSV(symbol):
    filename = "./" + symbol + '.csv'
    
    df = pd.read_csv(filename, index_col=False)
    
    return df

def normalize(dataset, scale=1, e=False):
    # normalize the dataset
    dataset = dataset / scale

    if(e == True):
        dataset = np.log(X)
    
    return dataset

def inverse_transform(matrix, scale=1, e=False):
    X = matrix * scale
        
    if(e == True):
        X = np.exp(X)
        
    return X

def create_dataset(dataset, seqn_size=1):
    dataX, dataY = [], []
    
    for i in range(dataset.shape[0]-seqn_size):
        a = dataset[i:(i+seqn_size),    :]
        a = np.ndarray.flatten(a)
        dataX.append(a)
        
        b = dataset[i+1:(i+seqn_size+1),:]
        b = np.ndarray.flatten(b)
        dataY.append(b)
    
    return np.array(dataX), np.array(dataY)

def setupTrainTest(dataset, seqn_size):
    # split into train and test sets
    test_size  = seqn_size + 5
    train_size = dataset.shape[0] - test_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    # reshape into X=t and Y=t+1
    trainX, trainY = create_dataset(train, seqn_size)
    testX,  testY  = create_dataset(test, seqn_size)
    
    return trainX, trainY, testX, testY

def loadModel(filepath,batch_size,iShape,loss='categorical_crossentropy',opt='adam',stack=1,state=False,od=1,act='softmax',neurons=8):
    if(os.path.exists(filepath)):
        # returns a compiled model identical to the previous one
        model = load_model(filepath)
        
        if(state == True):
            model.reset_states()
            
        print(model.summary())
    else:
        model = createModel(batch_size, iShape, loss, opt, stack, state, od, act, neurons)

    jsonModel = model.to_json()

    print(jsonModel)

    return model

def createModel(batch_size,iShape,obj='mean_squared_error',opt='adam',stack=1,state=False,od=1,act='softmax',neurons=8):
    model = Sequential()
    #shape input to be [samples, time steps, features]
    for i in range(stack):
        if(i == (stack -1)):
            if(state == True):
                model.add(LSTM(output_dim=neurons, batch_input_shape=(batch_size, iShape[1], iShape[2]), stateful=state, return_sequences=False))
            else:
                model.add(LSTM(output_dim=neurons, input_dim=iShape[2], stateful=state, return_sequences=False))
                
            print("Added LSTM Layer@", i, "return_sequences=False")
        else:
            if(state == True):
                model.add(LSTM(output_dim=neurons, batch_input_shape=(batch_size, iShape[1], iShape[2]), stateful=state, return_sequences=True))
            else:
                model.add(LSTM(output_dim=neurons, input_dim=iShape[2], stateful=state, return_sequences=True))
                          
            print("Added LSTM Layer@", i, "return_sequences=True")
    
    model.add(Dense(output_dim=od))

    print("Added Dense Layer")
    
    model.add(Activation(act)) 
    print("Added Activation Layer")
    
    print(model.summary())
    
    if(opt == "adam"):
        opt = Adam(lr=0.001)
    elif(opt == "RMSProp"):        
        opt = RMSprop(lr=0.001)
        
    model.compile(loss=obj, optimizer=opt, metrics=["accuracy"])
    
    return model

def saveModel(model, filepath):
    model.save(filepath)

def trainModel(model, X, Y, batch_size, epochs, modelSaved):
    for i in range(1):
        hist = model.fit(X, Y, nb_epoch=epochs, batch_size=batch_size, verbose=2, shuffle=False, validation_split=0.05)
        print(hist.history)
        
        if(i > 1 and i % 2 == 0):
            saveModel(model, modelSaved)
            print("Model saved at middle point")

        model.reset_states()
        print("Model State Reset")

    saveModel(model, modelSaved)
    print("Model saved at complete point")
    
def printX_Y(X, Y):
    for i in range(len(X)):
        a = X[i]
        b = Y[i]
        print(a, "->", b)
        
def printScore(model, X, Y):
    scores = model.evaluate(X, Y, verbose=0)
    print('Evaluate Score:', scores)
    
def lastInputSeq(X, lag):
    lX = X[-1 * lag:, :]
    # reshape input to be [samples, time steps, features]
    lY = np.reshape(lX, (1, -1, lag))
    
    return lX, lY    

def plot_all_feature(X, c):
    features = X.shape[1]
    
    for i in range(features):
        F = X[:,i]
        plt.scatter(np.arange(F.shape[0]), F, color=c)

def plot_result_2F(dataset, train, test, seqn_size):
    #input array shape is (, 2)
    style.use('ggplot')
    
    # shift train predictions for plotting
    trainPr       = np.empty_like(dataset)
    trainPr[:, :] = np.nan
    trainPr[seqn_size:len(train)+seqn_size, :] = train
    # shift test predictions for plotting
    testPre       = np.empty_like(dataset)
    testPre[:, :] = np.nan
    testPre[-1*len(test):, :] = test
    
    # plot baseline and predictions
    plot_all_feature(dataset, "blue")
    plot_all_feature(trainPr, "green")
    plot_all_feature(testPre, "red")
        
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()

batch_size = 1000
epochs_num = 5

seqn_size  = 6
output_dim = 6

LOG     = False
dataset = loadCSV('USDJPY')
dataset = dataset[['Close']]
dataset = dataset.values
dataset = dataset.astype('float32')    
SCALE   = np.amax(dataset.flatten())
print("SCALE=>", SCALE)

print(dataset[-seqn_size:])

dataset = normalize(dataset, SCALE, LOG)

trainX, trainY, testX, testY = setupTrainTest(dataset, seqn_size)
# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], -1, trainX.shape[1]))
testX1 = np.reshape(testX,  (testX.shape[0],  -1, testX.shape[1]))
# create and fit the LSTM network
modelFile = "./USDJPYRMSProp.h5"
model = loadModel(modelFile, batch_size, trainX.shape, loss='mean_squared_error', 
                  opt="RMSProp", stack=1, state=False, od=output_dim, act='linear', neurons=12)

trainModel(model, trainX, trainY, batch_size, epochs_num, modelFile)

# make predictions
trainPredict = model.predict(trainX, batch_size=batch_size)
testPredict  = model.predict(testX1, batch_size=batch_size)

lastX,lastX1 = lastInputSeq(dataset, seqn_size)
lastPredict  = model.predict(lastX1,  batch_size=batch_size)

printScore(model, trainX, trainY)
printScore(model, testX1,  testY)
#inverse scale
dataset      = inverse_transform(dataset, SCALE, LOG)

testX        = inverse_transform(testX, SCALE, LOG)
lastX        = inverse_transform(lastX, SCALE, LOG)
testPredict  = inverse_transform(testPredict,  SCALE, LOG)
lastPredict  = inverse_transform(lastPredict,  SCALE, LOG)
trainPredict = inverse_transform(trainPredict, SCALE, LOG)
printX_Y(testX, testPredict)

print("Tomorrow:", lastX, "=>", lastPredict)

X, Y = create_dataset(dataset, seqn_size)
plot_result_2F(X, trainPredict, testPredict, seqn_size)