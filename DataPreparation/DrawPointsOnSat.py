import numpy as np
import imageio
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import os

# Load image
VIS = imageio.imread('VIS.JPG')
VIS = VIS[600:1000,500:750]
VIS = VIS.astype(int)

# Load point
filename = './Points'
pos = pd.read_csv(filename)
xS  = np.array( pos['xSat'].tolist() ).astype(int)
yS  = np.array( pos['ySat'].tolist() ).astype(int)
filename = './NumData'
pos = pd.read_csv(filename)
num = np.array( pos.iloc[:,1].tolist() ).astype(int)

# Select points
threshold = 100
for n in range(0,len(num)):
    if num[n]<threshold :     # The condition ###
        xS[n] = 0
        yS[n] = 0

# Draw
def drawPoint(x,y,color):
    for i,j in zip(x,y):
        VIS[j,i,0] = color[0]
        VIS[j,i,1] = color[1]
        VIS[j,i,2] = color[2]

drawPoint(xS-1,yS-1,[255,0,0])
drawPoint(xS  ,yS-1,[255,0,0])
drawPoint(xS+1,yS-1,[255,0,0])
drawPoint(xS-1,yS  ,[255,0,0])
#drawPoint(xS  ,yS  ,[255,0,0])
drawPoint(xS+1,yS  ,[255,0,0])
drawPoint(xS-1,yS+1,[255,0,0])
drawPoint(xS  ,yS+1,[255,0,0])
drawPoint(xS+1,yS+1,[255,0,0])

plt.figure(1)
plt.imshow(VIS)
plt.show()


