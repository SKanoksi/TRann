"""
Temperature & Rainfall Hourly Forecast using Artificial Neural Network

Predict hourly total precipitation
Input argument 1 = Number of hidden nodes
Input argument 2 = Dataset
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

if len(sys.argv) != 2+1 :
    print('Invalid:: require 2 input arguments, i.e.,  Number of hidden nodes and TRann experimental point.')
    sys.exit()

# -------------------------------------

#colIn  = ['Lat','z','IR','VIS','WV','Wind_Strength','Wind_Direction','d2m','t2m','msl',\
#          'land1','land2','land3','land4','land5','land6'] #,'land7'] # msl <-> sp
colIn  = ['Month','Lat','Lon','z','IR','VIS','WV','Wind_Strength','Wind_Direction','d2m','t2m','msl',\
          'land1','land2','land3','land4','land5','land6','land7','sp'] # msl <-> sp
#colIn  = ['IR','VIS','WV','Wind_Strength','Wind_Direction','d2m','t2m','msl', 'sp','cp','lsp','tp'] # msl <-> sp
colOut = ['tp_Next']

nameOut = 'g-TP-H'
colDrop = ['Year','NoRain','HeavyRain']
Shape = [len(colIn),int(sys.argv[1]),len(colOut)]

# -------------------------------------

Frac_Train = 0.025
Frac_Valid = 0.025
Init  = 'glorot_uniform'
Acti  = 'My_Custom'
Loss  = 'mean_squared_error'
Opti  = keras.optimizers.RMSprop(0.0005)
#Opti  = keras.optimizers.RMSprop(0.001)
#Opti  = keras.optimizers.Adam(0.001)
#Opti  = keras.optimizers.SGD(lr=0.0001, momentum=0.8, decay=0.0, nesterov=False)
Metrics = ['mean_absolute_error']
EPOCHS  = 500
Patience = 20

# 1) Input -----------------------------------------------------------
filePath = sys.argv[2]
if not os.path.isfile(filePath):
    print('Dataset not found')
    sys.exit()
dataset = pd.read_csv(filePath)
# dataset = dataset.loc[ dataset['NoRain'].isin([0]) & dataset['HeavyRain'].isin([0]) ] # #####
dataset = dataset.loc[ dataset['HeavyRain'].isin([1]) ] # #####
dataset = dataset.drop(colDrop,axis=1)
dataset = dataset.dropna()
if len(dataset)<3000 :
    print('Too few data')
    sys.exit()

# 2) Normalize -------------------------------------------------------
stats = dataset.describe()
stats = stats.transpose()
stats['range'] = stats['max'] - stats['min']
stats = stats.transpose()
stats_colIn  = stats[colIn]
stats_colOut = stats[colOut]

stats = stats.transpose()
stats_colIn  = stats_colIn.transpose()
stats_colOut = stats_colOut.transpose()
def XtoNorm(x,statsIn):
    Min   = np.array(statsIn['min'])
    Range = np.array(statsIn['range'])
    for i in range(x.shape[0]):
        x[i,:] = (x[i,:]-Min)/Range
    return x
def NormtoX(x,statsIn):
    Min   = np.array(statsIn['min'])
    Range = np.array(statsIn['range'])
    for i in range(x.shape[0]):
        x[i,:] = Range*x[i,:]+Min
    return x

dataset = pd.DataFrame( XtoNorm(dataset.to_numpy(copy=False),stats) ,columns=dataset.columns)

def LtoS(list):
    s = [str(i) for i in list]
    return "".join(s)

filePath = './Result/Stats/' + sys.argv[1] + '_' + LtoS(Shape) + '_' + LtoS(Acti) + '_Stats_'+ nameOut + '_Mine.csv'
stats.to_csv(filePath)


# 3) Seperate to Train =0.6, Valid=0.2, Test=0.2 ---------------------
train = dataset.sample(frac=Frac_Train)
test  = dataset.drop(train.index)
valid = test.sample(frac=Frac_Valid/(1-Frac_Train))
test  = test.drop(valid.index)
#test  = test.sample(frac=0.1) ###
test_Out = pd.DataFrame( NormtoX(test[colOut].to_numpy(copy=True),stats_colOut), columns=colOut)
test  = test.drop(colOut,axis=1)
print('Num data = ',len(train),len(valid),len(test))


# 4) ANN -------------------------------------------------------------
class MyLayer(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer,self).__init__(**kwargs)

    def build(self, input_shape):
        self.I = int(input_shape[-1])
        self.H = self.output_dim
        self.A  = self.add_weight(name="A" , shape=(self.H,self.I), initializer='glorot_uniform', trainable=True)
        self.A2  = self.add_weight(name="A2" , shape=(self.H,self.I), initializer='glorot_uniform', trainable=True)
        self.X0 = self.add_weight(name="X0", shape=(self.H,self.I), initializer=keras.initializers.RandomUniform(minval=0, maxval=1.0, seed=None), trainable=True)
        self.B  = self.add_weight(name="B" , shape=(self.H,) , initializer=keras.initializers.RandomUniform(minval=0, maxval=1, seed=None), trainable=True)
        self.D  = self.add_weight(name="D" , shape=(self.H,self.I), initializer=keras.initializers.Ones(), trainable=True) # Unit
        super(MyLayer,self).build(input_shape)

    def call(self, xIn):
        xIn = tf.tile(xIn,[1,self.H])
        xIn = tf.reshape(xIn,[-1,self.H,self.I])
        dx  = tf.subtract(xIn,self.X0)

        T = tf.multiply(dx,self.A)
        T = T + tf.multiply(tf.multiply(dx,dx),self.A2)
        T = kf.dot(T,tf.ones([self.I,1]))
        T = tf.reshape(T,[-1,self.H])
        T = T + self.B

        # Inverse of distance --> D
        G = tf.multiply(dx,self.D)
        G = tf.multiply(G,G)
        G = kf.dot(G,tf.ones([self.I,1]))
        G = tf.reshape(G,[-1,self.H])
        G = tf.reciprocal(G)

        S = kf.dot(G,tf.ones([self.H,1]))
        S = tf.tile(S,[1,self.H])
        S = tf.reciprocal(S)
        G = tf.multiply(G,S)

        return tf.multiply(T,G)

    def compute_output_shape(self, input_shape):
        return self.output_dim


class SumLayer(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(SumLayer,self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_dim = int(input_shape[-1])
        super(SumLayer,self).build(input_shape)

    def call(self, x):
        return kf.dot(x,tf.ones([self.input_dim,1])) #tf.matmul(x,tf.ones([self.input_dim,1]))

    def compute_output_shape(self, input_shape):
        return self.output_dim

def reset_weights(model):
    session = kf.get_session()
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)

annInput  = layers.Input(shape=(Shape[0],))
annOutput = MyLayer(Shape[1])(annInput)
annOutput = SumLayer(Shape[2])(annOutput)
model = models.Model(inputs=annInput, outputs=annOutput)
model.compile(loss=Loss,optimizer=Opti,metrics=Metrics)
model.summary()


# 5) Train -----------------------------------------------------------
# Callback
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 10 == 0: print(epoch)
    print('.', end='')
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=Patience)

history = model.fit(train.loc[:,colIn], train.loc[:,colOut], epochs=EPOCHS, validation_data=(train.loc[:,colIn],train.loc[:,colOut]), verbose=0, callbacks=[early_stop, PrintDot()])


# 6) Test ------------------------------------------------------------
# Test
result_Out = test_Out.copy()
result_Out = model.predict(test.loc[:,colIn])
result_Out = pd.DataFrame( NormtoX(result_Out,stats_colOut), columns=colOut)
error  = test_Out - result_Out
stat   = error.describe()

# Experimental log
filePath = './Result/Log_All.csv'
#LogCol = ['Point','Model','Shape','Activation','Result']
newLog = [int(-1),nameOut + '_Mine',Shape,Acti,[stat.loc['mean'][0],stat.loc['std'][0]]]
Log = pd.read_csv(filePath)
Log.loc[len(Log)] = newLog
Log.to_csv(filePath,index=False)


# 7) Save ------------------------------------------------------------
# Save weights
filePath = './Result/Weights/' + sys.argv[2] + '_' + sys.argv[1] + '_Weights_' + nameOut + '_Mine.h5'
model.save_weights(filePath)

# Plot Train
filePath = './Result/Train/' + sys.argv[2] + '_' + sys.argv[1] + '_Train_' + nameOut + '_Mine.png'
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
plt.figure(0)
plt.clf()
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
filePath = './Result/Test/' + sys.argv[2] + '_' + sys.argv[1] + '_Test1_' + nameOut + '_Mine.png'
plt.figure(1)
plt.clf()
plt.scatter(test_Out, result_Out, s=1)
plt.xlabel('True Values ')
plt.ylabel('Predictions ')
plt.axis('square')
plt.grid(True)
plt.savefig(filePath)

filePath = './Result/Test/' + sys.argv[2] + '_' + sys.argv[1] + '_Test2_' + nameOut + '_Mine.png'
plt.figure(2)
plt.clf()
plt.hist(error.values, bins = 50)
plt.xlabel("Error")
plt.ylabel("Count")
plt.savefig(filePath)
plt.show()

# Progress
print('Finish :: Mean =',stat.loc['mean'][0],' Std =',stat.loc['std'][0])











