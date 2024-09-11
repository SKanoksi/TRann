import numpy as np
import imageio
from matplotlib import pyplot as plt
from matplotlib import colors
import pandas as pd
import seaborn as sns
import math
from scipy import interpolate
from netCDF4 import Dataset

filename = './2018_JUL_AUG.nc'
fin = Dataset(filename,'r')
Hour = 17
Day  = 5
time = 24*(Day-1) + Hour #    Hour = 0-23, Day = 1-30 or 31

# Corp (0-30 deg, 90-110 deg)
# Resolution = 0.25 degree
tLat = (90-30)*4
bLat =  (90-0)*4
lLon =      90*4
rLon =     110*4
z = fin.variables["cp"][time,tLat:bLat,lLon:rLon]

zlat = np.arange(30,  0,-0.25)
zlon = np.arange(90,110,0.25)
f = interpolate.interp2d(zlon,zlat,z.data,kind='linear')


plt.figure(1)
zlat = np.arange(21,  5,-0.1)
zlon = np.arange(97,107,0.1)
zNew = f(zlon,zlat)
plt.imshow(zNew[::-1,:])
plt.colorbar()

plt.show()



