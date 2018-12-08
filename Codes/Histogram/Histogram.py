import matplotlib.pyplot as plt
import numpy as np
import cv2

def Count_XPixels():
    # img = cv2.imread('CROP1.jpg')
    # gray = cv2.GaussianBlur(img, (7, 7), 0)
    # edged = cv2.Canny(gray, 50, 100)
    # edged = cv2.dilate(edged, None, iterations=1)
    # edged = cv2.erode(edged, None, iterations=1)
    # np.set_printoptions(threshold=np.nan)

    edged = cv2.imread('BoneAngle.jpg',0)

    result = []
    pos = []
    #image size (96 x 209)
    pixels = 0
    for i in range(edged.shape[0]):
        for j in range(edged.shape[1]):
            if edged[i, j] == 255:
                pixels+=1
        result.append(pixels)
        pos.append(i)
        pixels=0

    return result,pos,edged.shape[0]


def Graph_Xaxis(result,pos,xsize):
    print(result)
    print(pos)
    axes = plt.gca()
    axes.set_xlim([0,xsize])
    plt.plot(pos,result)
    plt.xlabel('X-axis Histogram')

def Count_YPixels():
    edged = cv2.imread('BoneAngle.jpg',0)

    result = []
    pos = []

    pixels = 0
    for i in range(edged.shape[1]):
        for j in range(edged.shape[0]):
            if edged[j, i] == 255:
                pixels+=1
        result.append(pixels)
        pos.append(i)
        pixels=0
    return result,pos,edged.shape[1]



def Graph_Yaxis(result,pos,ysize):
    print(result)
    print(pos)
    axes = plt.gca()
    axes.set_ylim([0, ysize])
    plt.plot(result, pos)
    plt.ylabel('Y-axis Histogram')


def BothOnX(count,pos):
    print(count)
    print(pos)
    axes = plt.gca()
    axes.set_xlim([0, 96])
    plt.plot(pos,count)
    plt.ylabel('Pixels')

# Draw on X-axis and Y-axis

pixel_count , pixel_pos , maxsize = Count_XPixels()
Graph_Xaxis(pixel_count,pixel_pos,maxsize)
pixel_count , pixel_pos , maxsize = Count_YPixels()
Graph_Yaxis(pixel_count,pixel_pos,maxsize)

#X-axis only

# pixel_count , pixel_pos , maxsize = Count_XPixels()
# BothOnX(pixel_count,pixel_pos)
# pixel_count , pixel_pos , maxsize = Count_YPixels()
# BothOnX(pixel_count,pixel_pos)

plt.show()
