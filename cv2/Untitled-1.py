"""
Task [I] - Demonstrating how to compute the histogram of an image using 4 methods.
(1). numpy based
(2). matplotlib based
(3). opencv based
(4). do it myself (DIY)
check the precision, the time-consuming of these four methods and print the result.
"""


import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2

###
#please coding here for solving Task [I].


img=cv2.imread('E:/pysource/test1/iread.png')        #读取文件
cv2.imshow('image is',img)                           #展示文件
b, g, r=cv2.split(img)                               #通道分离
cv2.imshow('red is',r)                               #展示红色通道
cv2.waitKey(0)
plt.hist(img[:,:,2].ravel(),bins=256,color='r')
plt.xlabel('bins=256 red levels')
plt.ylabel('Counted pixel numbers in each level')
plt.title('red Histogram')
plt.show()
hist = cv2.calcHist([img], [2], None, [256], [0.0, 256.0])
plt.plot(hist, color='r')
plt.show()





###





"""
Task [II]Refer to the link below to do the gaussian filtering on the input image.
Observe the effect of different @sigma on filtering the same image.
Try to figure out the gaussian kernel which the ndimage has used [Solution to this trial wins bonus].
https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
"""

###
#please coding here for solving Task[II]
'''
from scipy import ndimage
img2 = cv2.imread('E:/pysource/test1/iread.png')

index = 141
plt.subplot(index)
plt.imshow(img2)

for sigma in (2, 5, 10):
    im_blur = np.zeros(img2.shape, dtype=np.uint8)
    for i in range(3):
        im_blur[:, :, i] =ndimage.gaussian_filter(img2[:, :, i], sigma) #高斯滤波

    index +=1
    plt.subplot(index)
    plt.imshow(im_blur)

plt.show()
'''



"""
Task [III] Check the following link to accomplish the generating of random images.
Measure the histogram of the generated image and compare it to the according gaussian curve
in the same figure.
"""

###
#please coding here for solving Task[III]

'''
mean = (1, 2)
cov = [[1, 0], [0, 1]]
x = np.random.multivariate_normal(mean, cov, (256, 256), 'raise')

plt.hist(x.ravel(), bins=128, color='b')
plt.show()
'''
