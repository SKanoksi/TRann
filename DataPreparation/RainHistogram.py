import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import sys
import os

filePath = sys.argv[1] 
dataset = pd.read_csv(filePath)
dataset = dataset.dropna()

dataset_N = dataset.loc[ dataset['tp_Next']<0.5 ]
dataset_L = dataset.loc[ (0.5<dataset['tp_Next']) & (dataset['tp_Next']<20.0) ]
dataset_H = dataset.loc[ 20.0<dataset['tp_Next'] ]

plt.figure(0)
plt.clf()
plt.hist( dataset_N['tp_Next'], bins=1, color='gray')
plt.hist( dataset_L['tp_Next'], bins=39, color='red')
plt.hist( dataset_H['tp_Next'], bins=40, color='blue')
plt.xlabel("Precipitation (mm)")
plt.ylabel("Count")
plt.xlim(-0.5,35.0)
#plt.ylim(0,1500000)
plt.grid(True)
plt.show()




