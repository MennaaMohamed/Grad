import matplotlib.pyplot as plt
import numpy as np
import cv2

def Count_YPixels():
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

    return result,pos

pixel_count , pixel_pos = Count_YPixels()
print(pixel_count)
print(pixel_pos)
axes = plt.gca()
axes.set_xlim([0,96])
plt.plot(pixel_count)
plt.ylabel('Pixels')
plt.show()
