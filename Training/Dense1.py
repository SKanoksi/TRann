"""
Temperature & Rainfall Hourly Forecast using Artificial Neural Network

Fully connected ANN -> 1 hidden layer
Input argument 1 = Number of hidden nodes
Input argument 2 = Activation function
Input argument 3 = Dataset
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras import backend as kf

import sys
import os

if len(sys.argv) != 3+1 :
    print('Invalid:: require 3 input arguments, i.e.,  Number of hidden nodes, Activation function, TRann experimental point.')
    sys.exit()

Frac = 0.01
colIn  = ['Lat','z','IR','VIS','WV','Wind_Strength','Wind_Direction','d2m','t2m','msl',\
          'land1','land2','land3','land4','land5','land6'] #,'land7'] # msl <-> sp
colOut = ['t2m_Next']
nameOut = 'T'
colDrop = ['Unnamed: 0','Year','wasRain']
Shape = [len(colIn),int(sys.argv[1]),len(colOut)]

Init  = 'glorot_uniform'
Acti  = sys.argv[2]
Loss  = 'mean_squared_error'
Opti  = keras.optimizers.RMSprop(0.0005)
#Opti  = keras.optimizers.RMSprop(0.001)
#Opti  = keras.optimizers.Adam(0.001)
#Opti  = keras.optimizers.SGD(lr=0.0001, momentum=0.8, decay=0.0, nesterov=False)
Metrics = ['mean_absolute_error']
EPOCHS  = 500
Patience = 50
Times   = 1 # Repeat -> Save the best

# 1) Input -----------------------------------------------------------
filePath = sys.argv[3]
if not os.path.isfile(filePath):
    print('Dataset not found')
    sys.exit()
dataset = pd.read_csv(filePath)
dataset = dataset.drop(colDrop,axis=1)
dataset = dataset.sample(frac=Frac)
dataset = dataset.dropna()
if len(dataset)<3000 :
    print('Too few data')
    sys.exit()

# 2) Normalize -------------------------------------------------------
stats = dataset.describe()
stats = stats.transpose()
stats['range'] = stats['max'] - stats['min']
Min   = np.array(stats['min'])
Range = np.array(stats['range'])
def XtoNorm(x):
    for i in range(x.shape[0]):
        x[i,:] = (x[i,:]-Min)/Range
    return x
def NormtoX(x):
    for i in range(x.shape[0]):
        x[i,:] = Range*x[i,:]+Min
    return x
dataset = pd.DataFrame( XtoNorm(dataset.values) ,columns=dataset.columns)

filePath = './Result/Stats/' + sys.argv[3] + '_Stats_'+ nameOut + '_Dense1.csv'
stats.to_csv(filePath)


# 3) Seperate to Train =0.6, Valid=0.2, Test=0.2 ---------------------
train = dataset.sample(frac=0.6)
test  = dataset.drop(train.index)
valid = test.sample(frac=0.5)
test  = test.drop(valid.index)


# 4) ANN -------------------------------------------------------------
def FullyConnectedAnn(Nodes,Acti,Init):
    annIn = layers.Input(shape=(Nodes[0],))
    annX = layers.Dense(Nodes[1],activation=Acti,use_bias=True,kernel_initializer=Init)(annIn)
    for numNode in Nodes[2:] :
        annX = layers.Dense(numNode,activation=Acti,use_bias=True,kernel_initializer=Init)(annX)
    return [annIn, annX]

def reset_weights(model):
    session = kf.get_session()
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)

annInput, annOutput = FullyConnectedAnn(Shape,Acti,Init)
model = models.Model(inputs=annInput, outputs=annOutput)
model.compile(loss=Loss,optimizer=Opti,metrics=Metrics)


# 5) Train and Test -----------------------------------------------------------
# Callback
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 10 == 0: print(epoch)
    print('.', end='')
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=Patience)

mse = 1000
history  = None
annStats = None
filePath = './Result/Weights/' + sys.argv[3] + '_' + sys.argv[1] + '_' + sys.argv[2] + '_Weights_' + nameOut + '_Dense1.h5'
for i in range(Times):
    # Fit
    his = model.fit(train.loc[:,colIn], train.loc[:,colOut], epochs=EPOCHS, validation_data=(train.loc[:,colIn],train.loc[:,colOut]), verbose=0, callbacks=[early_stop, PrintDot()])
    # Test
    Out = model.evaluate(test.loc[:,colIn], test.loc[:,colOut], verbose=0)
    # Save the best
    if mse > Out[0] :
        mse = Out[0]
        history  = his
        annStats = Out
        # Save weights
        model.save_weights(filePath)
    # Progress
    print('Train',i,'::',annStats)
    # Reset weights of ANN
    reset_weights(model)


# 6) Save ------------------------------------------------------------
# Plot Train
filePath = './Result/Train/' + sys.argv[3] + '_' + sys.argv[1] + '_' + sys.argv[2] + '_Train_' + nameOut + '_Dense1.png'
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
plt.figure(0)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(hist['epoch'], hist['loss'],
       label='Train Error')
plt.plot(hist['epoch'], hist['val_loss'],
       label = 'Val Error')
plt.legend()
plt.grid(True)
plt.savefig(filePath)


# Plot Test
filePath = './Result/Test/' + sys.argv[3] + '_' + sys.argv[1] + '_' + sys.argv[2] + '_Test1_' + nameOut + '_Dense1.png'
result = test.copy()
result.loc[:,colOut] = model.predict(test.loc[:,colIn])
test   = pd.DataFrame( NormtoX(test.values),   columns=test.columns)
result = pd.DataFrame( NormtoX(result.values), columns=result.columns)
plt.figure(1)
plt.scatter(test.loc[:,colOut] , result.loc[:,colOut] )
plt.xlabel('True Values ')
plt.ylabel('Predictions ')
plt.axis('square')
plt.grid(True)
plt.savefig(filePath)

filePath = './Result/Test/' + sys.argv[3] + '_' + sys.argv[1] + '_' + sys.argv[2] + '_Test2_' + nameOut + '_Dense1.png'
plt.figure(2)
error = test.loc[:,colOut] - result.loc[:,colOut]
plt.hist(error.values, bins = 10)
plt.xlabel("Error")
plt.ylabel("Count")
plt.savefig(filePath)

print('\nTest::')
stat = error.describe()
print(stat)

# Experimental log
filePath = './Result/Log_All.csv'
#LogCol = ['Point','Model','Shape','Activation','Train_result','Test_result']
newLog = [int(-1),nameOut + '_Dense1',Shape,Acti,annStats,[stat.loc['mean'][0],stat.loc['std'][0]]]
Log = pd.read_csv(filePath)
Log.loc[len(Log)] = newLog
Log.to_csv(filePath,index=False)

