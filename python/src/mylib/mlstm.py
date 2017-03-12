import math
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import src.mylib.mfile as mfile
import src.mylib.mcalc as mcalc
from matplotlib import style
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.layers import Activation
from keras.layers import TimeDistributed
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# convert an array of values into a dataset matrix
def create_dataset(dataset, seqn_size=1):
    dataX, dataY = [], []
    
    for i in range(dataset.shape[0]-seqn_size):
        a = dataset[i:(i+seqn_size),    :]
        a = np.ndarray.flatten(a)
        dataX.append(a)
        
        b = dataset[i+1:(i+seqn_size+1),:]
        b = np.ndarray.flatten(b)
        dataY.append(b)
        #print( a, '->', b )
    
    return np.array(dataX), np.array(dataY)

def create_dataset_class(dataset, seqn_size=1):
    dataX, dataY = [], []
    
    for i in range(dataset.shape[0]-seqn_size):
        a = dataset[i:(i+seqn_size),   :]
        dataX.append(a)

        b = dataset[i + seqn_size,     :] 
        p = dataset[i + seqn_size - 1, :] 
        if(b > p):        
            dataY.append([1, 0, 0])
        elif(b == p):
            dataY.append([0, 1, 0])
        else:
            dataY.append([0, 0, 1])
        #print( a, '->', b )
    
    return np.array(dataX), np.array(dataY)

def printX_Y(X, Y):
    for i in range(len(X)):
        a = X[i]
        b = Y[i]
        print(a, "->", b)
        
def printX_YScaler(X, Y, scale, e=False):
    X = inverse_transform(X, scale, e)
    
    Y = inverse_transform(Y, scale, e)
    
    printX_Y(X, Y)

def loadExample():
    dataframe = pd.read_csv('db/international-airline-passengers.csv', usecols=[1], engine='python')
    
    dataset   = dataframe.values
    
    return dataset

def saveModel(model, filepath):
    model.save(filepath)
    
def loadModel(filepath,batch_size,iShape,loss='categorical_crossentropy',opt='adam',stack=1,state=False,od=1,act='softmax',neurons=8):
    if(os.path.exists(filepath)):
        # returns a compiled model
        # identical to the previous one
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
    #model.add(TimeDistributed(Dense(output_dim=od)))
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

def updateModel(model, opt):
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=["accuracy"])
    

def minmaxScaler():
    # normalize the dataset
    scaler  = MinMaxScaler(feature_range=(0, 1))
    
    return scaler

def normalize(dataset, scale=1, e=False):
    # normalize the dataset
    dataset = dataset / scale
    
    return dataset

def inverse_transform(matrix, scale=1, e=False):
    X = matrix * scale
        
    if(e == True):
        X = np.exp(X)
        
    return X

def log_scale(X, Y):
    lx = np.log(X)
    ly = np.log(Y)
    
    return lx, ly

def exp_scale(X, Y):
    ex = np.exp(X)
    ey = np.exp(Y)
    
    return ex, ey

def printScore(model, X, Y):
    #calculate root mean squared error
    #scores = math.sqrt(mean_squared_error(X, Y))
    scores = model.evaluate(X, Y, verbose=0)
    print('Evaluate Score:', scores)
    
def plot_result_1F(dataset, train, test, seqn_size):
    #input array shape is (, 1)
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
    X1 = np.arange(dataset.shape[0])
    Y1 = dataset.flatten()
    plt.scatter(X1, Y1, color="blue")
    
    plt.scatter(np.arange(trainPr.shape[0]), trainPr.flatten(), color="green")
    plt.scatter(np.arange(testPre.shape[0]), testPre.flatten(), color="red")
    
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()

def trainModel(model, X, Y, batch_size, epochs, modelSaved, validation=None, cb=None):
    for i in range(1):
        hist = model.fit(X, Y, nb_epoch=epochs, batch_size=batch_size, verbose=2, shuffle=False, 
                         validation_split=0.0, validation_data=validation, callbacks=cb)
        print(hist.history)
        
        if(i > 1 and i % 2 == 0):
            saveModel(model, modelSaved)
            print("Model saved at middle point")

        model.reset_states()
        print("Model State Reset")

    saveModel(model, modelSaved)
    print("Model saved at complete point")

def lastInputSeq(X, lag):
    lX = X[-1 * lag:, :]
    # reshape input to be [samples, time steps, features]
    lY = np.reshape(lX, (1, -1, lag))
    
    return lX, lY

def calUpDown(X):
    Y = np.zeros_like(X)
    
    siz = len(X)
    
    for i in range(siz):
        if(i > 0):
            if(X[i] > X[i-1]):
                Y[i] = 1
            elif(X[i] == X[i-1]):
                Y[i] = 0
            else:
                Y[i] = -1

    return Y

def classMap():
    map = { 0 : [ 1 ], 1 : [ 0 ], 2 : [-1 ] }
    
    return map

def inverseMap(X):
    map = classMap()
    
    Y = np.zeros((X.shape[0], 1))
    
    for i in range(X.shape[0]):
        index = np.argmax(X)
        
        Y[i] = map[index]
        
    print(Y)    
        
    return Y

def to_class(X):
    Y = np_utils.to_categorical(X)
    
    return Y

def plot_all_feature(X, c):
    features = X.shape[1]
    
    for i in range(features):
        F = X[:,i]
        plt.scatter(np.arange(F.shape[0]), F, color=c)
        
    
def plot_result_2F(dataset, predict, test=None, seqn_size=1):
    #input array shape is (, 2)
    style.use('ggplot')
    
    # plot baseline and predictions
    plot_all_feature(dataset, "blue")
    plot_all_feature(predict, "green")
    
    if(test != None):
        # shift test predictions for plotting
        testPre       = np.empty_like(dataset)
        testPre[:, :] = np.nan
        testPre[-1*len(test):, :] = test
        plot_all_feature(testPre, "red")
        
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()

def loadFXData(Symbol, db, num):
    # load the dataset
    dataframe = mfile.loadOneSymbol(Symbol, db)
    dataframe = mcalc.c_lastn(dataframe, num)
    
    return dataframe

def setupTrainTest(dataset, seqn_size, SCALE, LOG):
    # split into train and test sets
    test_size  = seqn_size + 5
    train_size = dataset.shape[0] - test_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    # reshape into X=t and Y=t+1
    trainX, trainY = create_dataset(train, seqn_size)
    testX,  testY  = create_dataset(test, seqn_size)
    
    printX_YScaler(testX, testY, SCALE, LOG)
    
    return trainX, trainY, testX, testY

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Value')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Value')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()