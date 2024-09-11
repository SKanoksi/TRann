"""
Merge dataset of each month at each location
1. Merge | every months (Add month)
2. Wind | (u,v) -> (W,theta)
3. Rain | (m) -> (mm)
4. Total Rain -> Rain/No-rain | <0.1 -> No-rain
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import os, sys

Suffix = 'TRann-06June19'
RainThreshold1 = 0.50
RainThreshold2 = 20.0
IN  = '/home/somrath/Work/Dataset/'  # Path to directory
OUT = '/home/somrath/Work/Dataset/'
MM  = ['07','08','09','10','11','12']
YY  = ['2018','2018','2018','2018','2018','2018']


# MAIN PROGRAM
filename = './Points'
pos = pd.read_csv(filename)
posLat = pos['lat'].tolist()
posLon = pos['lon'].tolist()

# 1. Merge
for nPos in range(len(posLat)):
    dataIn = [None]*len(MM)
    for m in range(len(MM)) :
        pathIn = IN  +  MM[m]+'_'+YY[m]  +  '/' +str(nPos)+'_'+MM[m]+YY[m]   #+ '.csv'
        if os.path.isfile(pathIn) :
            dataIn[m] = pd.read_csv(pathIn)
            dataIn[m].insert(0,'Month',int(MM[m]))
            dataIn[m].insert(0,'Year' ,int(YY[m]))
    if all(v is None for v in dataIn):
        continue
    data = pd.concat(dataIn)

# 2. Wind | (u,v) -> (W,theta)
    wind = np.sqrt( data['u10']**2 + data['v10']**2 )
    deg  = np.arctan2( data['v10'], data['u10'] )
    iU = data.columns.get_loc('u10')
    iV = data.columns.get_loc('v10')
    data.insert(iU,'Wind_Strength', wind)
    data.insert(iV,'Wind_Direction', deg)
    data.drop(['u10','v10'],axis=1)

# 3. Rain | (m) -> (mm) -> Total rain
    data['cp']  = 1000*data['cp']
    data['lsp'] = 1000*data['lsp']
    data['tp']  = data['cp'] + data['lsp']

    data['cp_Next']  = 1000*data['cp_Next']
    data['lsp_Next'] = 1000*data['lsp_Next']
    data['tp_Next']  = data['cp_Next'] + data['lsp_Next']

# 4. Total Rain -> Rain/No-rain | <0.1 -> No-rain
    data['NoRain']    = data['tp_Next'].lt(RainThreshold1).astype(int)
    data['HeavyRain'] = data['tp_Next'].gt(RainThreshold2).astype(int)

# 4.5 Recorrect tp
    """
    x = data['tp'].copy()
    for i in range(data.shape[0]) :
        if not data['wasRain'].iloc[i] :
            x.iloc[i] = 0.0

    data['tp'] = x
    """
# 5. Save
    pathOut = OUT +str(nPos)+'_' + Suffix   #+ '.csv'
    data.to_csv(pathOut)

    print('Finish:', nPos)
