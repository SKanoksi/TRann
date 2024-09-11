"""
Add data to each representative points which satisfy
1) 1 point per ERA5 grid
2) their corresponding points on the satellite images are reliable (acceptable noise, 3x3)
3) Close to Thailand (roughly the same weather system)
4) Average over 3x3
[MISSING_VALUE = -1]
"""

import numpy as np
import imageio
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import os

HH  = range(0,24,1)
DD  = range(1,31+1,1)
OUT = '/home/somrath/Work/Dataset/12_2018/'  # Path to directory
MM  = '12'      # Month 01-12
YY   = '2018'   # Year XXXX
DICT = '/home/somrath/Work/SatelliteData_All/'  # Path to directory
threshold = 20
TYPE = ['IR1','VIS','WV']



def toSTR(x):
	if x < 10 :
		return '0'+str(x)
	else:
		return str(x)

def isAvaliable(typE,day,hr):
	filePath = DICT + TYPE[typE] + '_ASIA.'+YY+'_'+MM +toSTR(day)+'_' + toSTR(hr) + 'XX.JPG'
	return os.path.isfile(filePath)

def readImage(typE,day,hr):
	filePath = DICT + TYPE[typE] + '_ASIA.'+YY+'_'+MM +toSTR(day)+'_' + toSTR(hr) + 'XX.JPG'
	return imageio.imread(filePath)

def cleanImage(image):
    out = np.zeros(image.shape[0:2]) # Grey scale
    for j in range(image.shape[0]):
        for i in range(image.shape[1]):
            R = int( image[j,i,0] )
            G = int( image[j,i,1] )
            B = int( image[j,i,2] )
            if (abs(R-G) > threshold) or \
               (abs(R-B) > threshold) or \
               (abs(G-B) > threshold) :
                out[j,i] = -9999   # 9999>9*mean(255,255,255)
            else:
                out[j,i] = np.mean(image[j,i,:])

    return out

def setSatData(day,hour,x,y):
    IR  = readImage(0,day,hour)
    VIS = readImage(1,day,hour)
    WV  = readImage(2,day,hour)

    # 600,1000 -> lat 21,5
    # 500,750  -> long 97,107
    IR  = IR[600:1000,500:750,:]
    IR  = IR.astype(float)
    IR  = cleanImage(IR)
    VIS = VIS[600:1000,500:750,:]
    VIS = VIS.astype(float)
    VIS = cleanImage(VIS)
    WV  = WV[600:1000,500:750,:]
    WV  = WV.astype(float)
    WV  = cleanImage(WV)

    # Get Sat data
    out = np.zeros([len(x),3])
    for n in range(len(x)):
        out[n,0] = np.mean(  IR[y[n]-1:y[n]+2,x[n]-1:x[n]+2] ) #avg_Sat(IR, x[n],y[n])
        out[n,1] = np.mean( VIS[y[n]-1:y[n]+2,x[n]-1:x[n]+2] ) #avg_Sat(VIS,x[n],y[n])
        out[n,2] = np.mean(  WV[y[n]-1:y[n]+2,x[n]-1:x[n]+2] ) #avg_Sat(WV, x[n],y[n])

    return out

def saveSatData(n,time,data,fileOut):
    out = []
    for t in range(len(time)):
        Sat  = data[t]
        Date = time[t]
        setData = np.zeros([5])
        setData[0] = Date[0]
        setData[1] = Date[1]
        setData[2] = Sat[n,0]
        setData[3] = Sat[n,1]
        setData[4] = Sat[n,2]
        if setData[2]>=0 and \
           setData[3]>=0 and \
           setData[4]>=0 :
            out.append( setData )


    if len(out)!=0 :
        dataOut = pd.DataFrame(np.array(out),columns=['Day','Hour','IR','VIS','WV'])
        dataOut.to_csv(fileOut,index=False)



# MAIN PROGRAM
# 1) Load points
filename = './Points'
pos = pd.read_csv(filename)
xSat = np.array( pos['xSat'].tolist() ).astype(int)
ySat = np.array( pos['ySat'].tolist() ).astype(int)

# 2) Load satellite images
time = []
data = []
for D in DD :
	for H in HH :
	    if isAvaliable(0,D,H) and \
	       isAvaliable(1,D,H) and \
	       isAvaliable(2,D,H) :
	        print('Loading Satellite data:',D,H)
	        time.append( [D,H] )
	        data.append( setSatData(D,H,xSat,ySat) )

# 3) Save data -> one point per file
for nPos in range(len(xSat)):
    pathOut = OUT + str(nPos) + '_' + MM + YY # + '.csv'
    saveSatData(nPos,time,data,pathOut)











