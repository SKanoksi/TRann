"""
Select usable satellite images
1) at min = 00
2) average of min = 50 and min = 10 (-> min = 00)
then, save to min = XX.JPG
"""

import numpy as np
import imageio
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import os


HH  = range(0,24,1)
DD  = range(1,28+1,1)
MM  = '02'        # Month 01-12
YY   = '2019'   # Year XXXX
TYPE = ['IR/IR1_','VIS/VIS_','WV/WV_']
DICT = '/home/somrath/Work/Satellite_2018/'  # Path to directory

def preHour(x):
    if x==0:
        y = 23
    else:
        y = x-1
    return y

def toSTR(x):
	if x < 10 :
		return '0'+str(x)
	else:
		return str(x)

def isAvaliable(path,hr,min):
	filePath = path + '_' + toSTR(hr) + toSTR(min) + '.JPG'
	return os.path.isfile(filePath)

def checkTypeImage(path,hr):
	if isAvaliable(path,hr,0):
		flag = 1
	else:
		if isAvaliable(path,hr,10) and isAvaliable(path,preHour(hr),50):
			flag = 2
		else:
			flag = 0
	return flag

def readImage(path,hr,min):
	filePath = path + '_' + toSTR(hr) + toSTR(min) + '.JPG'
	return imageio.imread(filePath)

def saveCleanImage(path,hr,flag):
	filePath = path + '_' + toSTR(hr) + 'XX.JPG'
	if flag == 1 :
		img = readImage(path,hr,0)
		imageio.imwrite(filePath,img)
	if flag == 2 :
		img = readImage(path,hr,10).astype(float)/2 + readImage(path,preHour(hr),50).astype(float)/2
		imageio.imwrite(filePath,img.astype(np.uint8))

# MAIN PROGRAM
# Check
table = np.zeros([len(DD), len(HH)])
for nD in range(len(DD)):
	for nH in range(len(HH)):
		i = 0

		for typE in TYPE:
			path = DICT + typE +'ASIA'+'.'+YY+'_'+MM+ toSTR(DD[nD])
			flag = checkTypeImage(path,HH[nH])
			if flag != 0:
				i = i +1

		if i == 3:
			table[nD,nH] = 1

# Save
for nD in range(len(DD)):
	for nH in range(len(HH)):

		if table[nD,nH] == 1 :
		    for typE in TYPE:
			    path = DICT + typE +'ASIA'+'.'+YY+'_'+MM+ toSTR(DD[nD])
			    flag = checkTypeImage(path,HH[nH])
			    print('Saving: ',path)
			    saveCleanImage(path,HH[nH],flag)

# Finish
print("Finish: %g datasets are avaliable in %s/%s" % (np.sum(table),MM,YY))


