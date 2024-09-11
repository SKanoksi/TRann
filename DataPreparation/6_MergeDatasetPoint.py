"""
Merge dataset of each location
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import os, sys

Suffix = 'TRann-06June19'
IN  = '/home/somrath/Work/Dataset/'  # Path to directory
OUT = '/home/somrath/Work/All_TRann_Datasets_New.csv'

col = ['Year','Month','Day','Hour','IR','VIS','WV','Wind_Strength','Wind_Direction','u10','v10','d2m','t2m','msl','sp','cp','lsp','t2m_Next','tp,wasRain']


# MAIN PROGRAM
filename = './WF-AI_Points'
pos = pd.read_csv(filename)
posLat = pos['lat'].tolist()
posLon = pos['lon'].tolist()
z    = pos['Geopotential']
mask1 = pos['Land1']
mask2 = pos['Land2']
mask3 = pos['Land3']
mask4 = pos['Land4']
mask5 = pos['Land5']
mask6 = pos['Land6']
mask7 = pos['Land7']


data = [None]*len(posLat)
for n in range(len(posLat)):

# 1. Read
    pathIn = IN +str(n)+'_' + Suffix   #+ '.csv'
    if os.path.isfile(pathIn) :
        dataIn = pd.read_csv(pathIn)
    else:
        continue
    dataIn = dataIn.drop(['Unnamed: 0'],axis=1)


# 2. Add lat,lon,land,z
    dataIn.insert(0,'z',      z[n])
    dataIn.insert(0,'land7', mask7)
    dataIn.insert(0,'land6', mask6)
    dataIn.insert(0,'land5', mask5)
    dataIn.insert(0,'land4', mask4)
    dataIn.insert(0,'land3', mask3)
    dataIn.insert(0,'land2', mask2)
    dataIn.insert(0,'land1', mask1)
    dataIn.insert(0,'Lon', posLon[n])
    dataIn.insert(0,'Lat', posLat[n])


    data[n] = dataIn
    print('Get:', n)

# 5. Save
data = pd.concat(data)
data.to_csv(OUT,index=False)

