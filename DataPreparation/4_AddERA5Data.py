"""
Add ERA5 data to each experimental point and TIME (depend on avaliable satellite images)
1) 1 point per ERA5 grid
2) their corresponding points on the satellite images are reliable (acceptable noise)
3) Close to Thailand (roughly the same weather system)
4) Interpolation = Bilinear
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import os, sys
from netCDF4 import Dataset
from scipy import interpolate


BaseHour = 0           #     0     | 24*(31)-1  # as 1 file contains two months
LastHour = 24*31+1     # 24*(31)+1 | 24*(31*2)  # as 1 file contains two months
IN       = '/home/somrath/Work/BackUp/Dataset/07_2018/'  # Path to directory
PathERA5 = '/home/somrath/Work/BackUp/2018_JUL_AUG.nc'
MM   = '07'      # Month 01-12
YY   = '2018'    # Year XXXX
TYPE = ['u10','v10','d2m','t2m','msl','sp','cp','lsp'] # Change this, will affect t2m_Next
col  = ['u10','v10','d2m','t2m','msl','sp','cp','lsp','t2m_Next','cp_Next','lsp_Next']



# MAIN PROGRAM
# 1) Load points and ERA5 ------------------------------------
filename = './Points'
pos = pd.read_csv(filename)
posLat = pos['lat'].tolist()
posLon = pos['lon'].tolist()

# Corp (0-30 deg, 90-110 deg)
# Resolution = 0.25 degree
# Interpolate ###
lat  = np.arange(30,  0,-0.25)
tLat = (90-30)*4
bLat =  (90-0)*4
lon  = np.arange(90,110,0.25)
lLon =      90*4
rLon =     110*4
fin  = Dataset(PathERA5,'r')
ERA5 = [None]*len(TYPE)
for ty in range(len(TYPE)):
    era5_Data = fin.variables[TYPE[ty]][BaseHour:LastHour,tLat:bLat,lLon:rLon]
    ERA5[ty] = era5_Data.data



# 2) Load dataset -> 1 point per file ------------------------------------
for nPos in range(len(posLat)):
    pathIn = IN + str(nPos) + '_' + MM + YY #+ '.csv'
    if not os.path.isfile(pathIn) :
        continue

    csvIn = pd.read_csv(pathIn)
    time  = 24*(csvIn['Day']-1) + csvIn['Hour']     # *** BE CAREFUL ***
    time  = np.array( time.tolist() ).astype(int)


# 3) Save data -> 1 point per file ------------------------------------
    setData = np.zeros([len(col),len(time)])
    for ty in range(len(TYPE)):
        for t in range(len(time)):
            interp = interpolate.interp2d(lon,lat, ERA5[ty][time[t],:,:], kind='linear')
            setData[ty,t] = interp( posLon[nPos], posLat[nPos] )

    # Special t2m_Next ----- ### ERROR ### Last hour of ERA5
    #flag = 0
    for t in range(len(time)):
        #if time[t]+1 < LastHour:
        interp = interpolate.interp2d(lon,lat, ERA5[3][time[t]+1,:,:], kind='linear') # 't2m' = TYPE[3] ***
        setData[8,t] = interp( posLon[nPos], posLat[nPos] )    # 8 = last column ***

        interp = interpolate.interp2d(lon,lat, ERA5[6][time[t]+1,:,:], kind='linear') # 'cp' = TYPE[3] ***
        setData[9,t] = interp( posLon[nPos], posLat[nPos] )    # 9 = last column ***
        interp = interpolate.interp2d(lon,lat, ERA5[7][time[t]+1,:,:], kind='linear') # 'lsp' = TYPE[3] ***
        setData[10,t] = interp( posLon[nPos], posLat[nPos] )    # 10 = last column***
        #else:
        #    flag = 1
        #    setData[8,t] = 0

    # Set dataframe
    dataOut = pd.DataFrame(setData.T,columns=col)
    csvIn   = pd.concat([csvIn,dataOut], axis=1)
    #if flag == 1 :
    #    csvIn = csvIn.drop(len(time)-1)
    #    print('Remove last set: No 1-hour-ahead data')

    # Save dataframe
    csvIn.to_csv(pathIn,index=False)
    print('FINISH:', pathIn)








