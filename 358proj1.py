# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Read in original RGB image.
import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt
from skimage.transform import pyramid_reduce

A = io.imread('shed1-small.jpg')
(m,n,o) = A.shape
plt.imshow(A)
plt.title('Image A')
plt.axis('off')
plt.show()
# Extract color channels.
RC = A[:,:,0] # Red channel
GC = A[:,:,1] # Green channel
BC = A[:,:,2] # Blue channel
# Create an all black channel.
allBlack = np.zeros((m, n), dtype=np.uint8)
# Create color versions of the individual color channels.
justRed = np.stack((RC, allBlack, allBlack), axis=2)
justGreen = np.stack((allBlack, GC, allBlack),axis=2)
justBlue = np.stack((allBlack, allBlack, BC),axis=2)
#Plot and save Red
plt.imshow(justRed)
plt.title('Red Channel A')
plt.axis('off')
plt.show()
io.imsave('RC_image.jpg', justRed)
#Plot and save Green
plt.imshow(justGreen)
plt.title('Green Channel A')
plt.axis('off')
plt.show()
io.imsave('GC_image.jpg', justGreen)
#Plot and save Blue
plt.imshow(justBlue)
plt.title('Blue Channel A')
plt.axis('off')
plt.show()
io.imsave('BC_image.jpg', justBlue)

#Compute Grey Level AG
AG = (RC + GC + BC) / 3
plt.imshow(AG, cmap='gray')
plt.title('Gray Level AG')
plt.axis('off')
plt.show()

#Get the histograms for RC GC BC AG Ready
hist_RC, bins_RC = np.histogram(RC.flatten(), bins=256, range=(0, 255))
hist_GC, bins_GC = np.histogram(GC.flatten(), bins=256, range=(0, 255))
hist_BC, bins_BC = np.histogram(BC.flatten(), bins=256, range=(0, 255))
hist_AG, bins_AG = np.histogram(AG.flatten(), bins=256, range=(0, 255))

plt.plot(hist_RC, color='red')
plt.title('RC Histogram')
plt.show()

plt.plot(hist_GC, color='green')
plt.title('GC Histogram')
plt.show()


plt.plot(hist_BC, color='blue')
plt.title('BC Histogram')
plt.show()

plt.plot(hist_AG, color='gray')
plt.title('AG Histogram')
plt.show()

#Binarazing the image

TB = int(input("Enter a TB value: "))
AB = np.where(AG<TB, 0, 255).astype(np.uint8)

# Display image AB
plt.imshow(AB, cmap='gray')
plt.title('Binarized Image AB')
plt.axis('off')  
plt.show()

#Simple Edge Detection
TE = int(input("Enter a TE value: "))
#Gx along rows
Gx = np.zeros((m, n), dtype=np.int32)
for i in range(m):
    for j in range(n-1):
        Gx[i,j] = AG[i, j+1] - AG[i,j]
  
Gx[i, n-1] = 0
        
#Gy along columns        
Gy = np.zeros((m, n), dtype=np.int32)        
for i in range(m-1):
    for j in range(n):
        Gy[i,j] = AG[i+1, j] - AG[i,j]
            
Gy[m-1, :] = 0
    
#Gm gradient magnitude
Gm = np.sqrt(Gx**2 + Gy**2)

#Edge image of AE
AE = np.where(Gm > TE, 255, 0).astype(np.uint8)
plt.imshow(AE, cmap='gray')
plt.title('Edge Image AE ')
plt.axis('off')  # Optional: Hide the axes
plt.show()

#Downscale image AG using pyramid_reduce from image processing library
AG2 = pyramid_reduce(AG, downscale = 2)
AG4 = pyramid_reduce(AG2, downscale = 2)
AG8 = pyramid_reduce(AG4, downscale = 2)

#Show AG AG2 AG4 AG8
plt.imshow(AG)
plt.title('AG image')
plt.axis('off')
plt.show()

plt.imshow(AG2)
plt.title('AG2 image')
plt.axis('off')
plt.show()

plt.imshow(AG4)
plt.title('AG4 image')
plt.axis('off')
plt.show()

plt.imshow(AG8)
plt.title('AG8 image')
plt.axis('off')
plt.show()


