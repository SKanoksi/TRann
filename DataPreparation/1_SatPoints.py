"""
Select representative points which satisfy
1) 1 point per ERA5 grid
2) their corresponding points on the satellite images are reliable (acceptable noise, 3x3)
3) Close to Thailand (roughly the same weather system)
"""

import numpy as np
import imageio
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

# 600,1000 -> lat 21,5
# 500,750  -> long 97,107
VIS = imageio.imread('VIS.JPG')
VIS = VIS[600:1000,500:750]
VIS = VIS.astype(int)

IR = imageio.imread('IR.JPG')
IR = IR[600:1000,500:750]
IR = IR.astype(int)

WV = imageio.imread('WV.JPG')
WV = WV[600:1000,500:750]
WV = WV.astype(int)

'''
plt.subplot(321)
plt.imshow(VIS)
plt.subplot(323)
plt.imshow(IR)
plt.subplot(325)
plt.imshow(WV)
'''
h = VIS.shape[0]
w = VIS.shape[1]

def drawGridPoint(x,y,color):
    for j in y:
        for i in x:
            VIS[j,i,0] = color[0]
            VIS[j,i,1] = color[1]
            VIS[j,i,2] = color[2]
            IR[j,i,0]  = color[0]
            IR[j,i,1]  = color[1]
            IR[j,i,2]  = color[2]
            WV[j,i,0]  = color[0]
            WV[j,i,1]  = color[1]
            WV[j,i,2]  = color[2]

def drawPoint(x,y,color):
    for i,j in zip(x,y):
        VIS[j,i,0] = color[0]
        VIS[j,i,1] = color[1]
        VIS[j,i,2] = color[2]
        IR[j,i,0]  = color[0]
        IR[j,i,1]  = color[1]
        IR[j,i,2]  = color[2]
        WV[j,i,0]  = color[0]
        WV[j,i,1]  = color[1]
        WV[j,i,2]  = color[2]

# 1) Clean image
# If colorDiff>20 -> Discard, else -> Average
def createMask(img):
    threshold = 20
    mask = np.ones([h,w])
    for j in range(0,h):
        for i in range(0,w):
            R = int( img[j,i,0] )
            G = int( img[j,i,1] )
            B = int( img[j,i,2] )
            if (abs(R-G) > threshold) or \
               (abs(R-B) > threshold) or \
               (abs(G-B) > threshold) :
                mask[j,i] = 0
    return mask

def cleanImage(imageIn,mask,color):
    for j in range(0,h):
        for i in range(0,w):
            if mask[j,i]==1 :
                avg = np.mean(imageIn[j,i,:])
                imageIn[j,i,0] = avg
                imageIn[j,i,1] = avg
                imageIn[j,i,2] = avg
            else:
                imageIn[j,i,0] = color[0]
                imageIn[j,i,1] = color[1]
                imageIn[j,i,2] = color[2]

# Clean VIS, IR, WV
mask = createMask(VIS)
cleanImage(VIS,mask,[0,255,0])
mask = createMask(IR)
cleanImage(IR,mask,[0,255,0])
mask = createMask(WV)
cleanImage(WV,mask,[0,255,0])



# 2) Draw ERA5 datapoint
# abs(5-21) = 16 deg -> 16/0.25 = 64
# abs(107-97) = 10 deg -> 10/0.25 = 40
xERA5 = np.arange(0,40,1)*w/40
yERA5 = np.arange(0,64,1)*h/64
xERA5 = np.round(xERA5)
yERA5 = np.round(yERA5)
xERA5 = xERA5.astype(int)
yERA5 = yERA5.astype(int)
# Draw after selecting our experimental points only !!!
# (--> interfere with checking nearby == usable)


# 3) Select domain -> all
#drawGridPoint(range(160,330+1),[170,450],[0,0,255])
#drawGridPoint([160,330],range(170,450+1),[0,0,255])



# 4) Select our experimental points
xP = xERA5[1:-1]
yP = yERA5[1:-1]
# 4.1) nearby 3x3 == usable
flag = np.ones([yP.shape[0],xP.shape[0]])
for j in range(yP.shape[0]):
    for i in range(xP.shape[0]):
        for y in range(3):
            for x in range(3):
                if (VIS[ yP[j]+y,xP[i]+x, 0] == 0 and \
                    VIS[ yP[j]+y,xP[i]+x, 1] == 255 and \
                    VIS[ yP[j]+y,xP[i]+x, 2] == 0 ) or  \
                   (IR[ yP[j]+y,xP[i]+x, 0]  == 0 and \
                    IR[ yP[j]+y,xP[i]+x, 1]  == 255 and \
                    IR[ yP[j]+y,xP[i]+x, 2]  == 0 ) or  \
                   (WV[ yP[j]+y,xP[i]+x, 0]  == 0 and \
                    WV[ yP[j]+y,xP[i]+x, 1]  == 255 and \
                    WV[ yP[j]+y,xP[i]+x, 2]  == 0 ) :
                    flag[j,i] = 0
nS = np.sum(flag)
nS = nS.astype(int)
xS = np.zeros([nS])
xS = xS.astype(int)
yS = np.zeros([nS])
yS = yS.astype(int)
num = 0
for j in range(yP.shape[0]):
    for i in range(xP.shape[0]):
        if flag[j,i] == 1:
            xS[num] = xP[i]
            yS[num] = yP[j]
            num = num + 1

print("Total selected points = ",nS)

drawPoint(xS-1,yS-1,[255,0,0])
drawPoint(xS  ,yS-1,[255,0,0])
drawPoint(xS+1,yS-1,[255,0,0])
drawPoint(xS-1,yS  ,[255,0,0])
#drawPoint(xS  ,yS  ,[255,0,0])
drawPoint(xS+1,yS  ,[255,0,0])
drawPoint(xS-1,yS+1,[255,0,0])
drawPoint(xS  ,yS+1,[255,0,0])
drawPoint(xS+1,yS+1,[255,0,0])



# 2) ...above...
drawGridPoint(xERA5,yERA5,[0,0,255])


# Show
'''
plt.subplot(322)
plt.imshow(VIS)
plt.subplot(324)
plt.imshow(IR)
plt.subplot(326)
plt.imshow(WV)
plt.show()
'''

plt.figure(1)
plt.imshow(VIS)
plt.figure(2)
plt.imshow(IR)
plt.figure(3)
plt.imshow(WV)
plt.show()


# Save ERA5 positions (lat,lon,xPoint,yPoint) to CSV
xP,yP = np.meshgrid(xP,yP)
xP = xP.flatten()
yP = yP.flatten()
lon = 97 + (xP*10/w)
lat = 21 - (yP*16/h)
filename = './Test'
pos = np.array([lat,lon,xP,yP])
pos = pd.DataFrame(pos.T,columns=['lat','lon','xSat','ySat'])
pos.to_csv(filename,index=False)
print(pos)




