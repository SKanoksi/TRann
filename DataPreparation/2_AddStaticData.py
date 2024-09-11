"""
Add static data, i.e., height(z) and land cover, to each experimental point
1) 1 point per ERA5 grid
2) their corresponding points on the satellite images are reliable (acceptable noise)
3) Close to Thailand (roughly the same weather system)
"""

import numpy as np
import imageio
from matplotlib import pyplot as plt
from matplotlib import colors
import pandas as pd
import seaborn as sns
import math
from scipy import interpolate
from netCDF4 import Dataset


filename = './Test'
pos = pd.read_csv(filename)
lat = pos['lat'].tolist()
lon = pos['lon'].tolist()



# 1) Import height ------------------------------------
filePath = './BackUp/2018DEC.nc'
fin = Dataset(filePath,'r')
# Corp (0-30 deg, 90-110 deg)
# Resolution = 0.25 degree
tLat = (90-30)*4
bLat =  (90-0)*4
lLon =      90*4
rLon =     110*4
z = fin.variables["z"][-1,tLat:bLat,lLon:rLon]

zlat = np.arange(30,  0,-0.25)
zlon = np.arange(90,110,0.25)
f = interpolate.interp2d(zlon,zlat,z.data,kind='linear')

# Add height
zPoint = np.zeros([len(lat)])
for n in range(len(lat)):
    zPoint[n] = f(lon[n],lat[n])

pos['Geopotential'] = zPoint


# 2) Import land cover (1/4 of the Earth) ------------------------------------
filePath = './BackUp/LandCover.tif'
land = imageio.imread(filePath)
# Corp (5-21 deg, 97-107 deg)
# Resolution = 15 arcseconds
# 1 degree = 3600 arcseconds
tLat = (90-21)*240
bLat = (90- 5)*240
lLon =      97*240
rLon =     107*240
land = land[tLat:bLat,lLon:rLon]
land = land.astype(int)

'''
% Classified by water content
1 | Water, snow, Wetland, Mangrove
2 | Evergreen, Mixed, Deciduous
3 | Paddy field, Cropland, Cropland/Vegetation mosaic
4 | Herbaceous, Sparse vegetation
5 | Urban
6 | Opentree, Shrub
7 | Rock, Sand
'''
landDict = {20:1,19:1,15:1,14:1, 1:2,3:2,5:2,2:2,4:2, 12:3,11:3,13:3, 8:4,9:4,10:4, 18:5, 6:6,7:6, 16:7,17:7}

def latlon_Histogram(Lat,Lon):
    Lat = round( (21-Lat)*240 )
    Lon = round( (Lon-97)*240 )
    his = np.zeros([7])
    for j in range(Lat-20,Lat+20):
        for i in range(Lon-20,Lon+20):
            mask = landDict.get(land[j,i])-1
            his[mask] = his[mask]+1
    return his

# Add land cover
landMask1 = []
landMask2 = []
landMask3 = []
landMask4 = []
landMask5 = []
landMask6 = []
landMask7 = []
for n in range(len(lat)):
    his = latlon_Histogram(lat[n],lon[n])
    landMask1.append( his[0] )
    landMask2.append( his[1] )
    landMask3.append( his[2] )
    landMask4.append( his[3] )
    landMask5.append( his[4] )
    landMask6.append( his[5] )
    landMask7.append( his[6] )

pos['Land1'] = landMask1
pos['Land2'] = landMask2
pos['Land3'] = landMask3
pos['Land4'] = landMask4
pos['Land5'] = landMask5
pos['Land6'] = landMask6
pos['Land7'] = landMask7


# 3) Save ------------------------------------------------------------------------
print(pos)
pos.to_csv(filename,index=False)

# Check
plt.figure(1)
zlat = np.arange(21,  5,-0.1)
zlon = np.arange(97,107,0.1)
zNew = f(zlon,zlat)
plt.imshow(zNew[::-1,:])
plt.colorbar()

plt.figure(2)
plt.imshow(land)
#plt.colorbar()

plt.show()




