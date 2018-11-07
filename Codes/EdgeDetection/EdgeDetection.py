import cv2
import imutils
import numpy as np
from PIL import Image
from imutils import contours, perspective
from matplotlib import pyplot as plt

def manual_canny(img):
    edges = cv2.Canny(img,170,200)
    print(edges)
    plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')

    # plt.imsave('ManualCanny.jpg', edges,cmap = 'gray')
    plt.show()

def auto_canny(image, sigma=0.33):

    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    wide = cv2.Canny(blurred, 10, 200)
    tight = cv2.Canny(blurred, 225, 250)
    # compute the median of the single channel pixel intensities
    v = np.median(blurred)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    cv2.imshow("Original", image)
    cv2.imshow("Edges", np.hstack([wide, tight, edged]))
    # cv2.imwrite("WideCanny.jpg",wide)
    # cv2.imwrite("TightCanny.jpg", tight)
    # cv2.imwrite("AutoCanny.jpg", edged)
    cv2.waitKey(0)



def Dilated_Canny(image):
    gray = cv2.GaussianBlur(image, (7, 7), 0)
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    cv2.imwrite("DilatedCanny.jpg",edged)


img = cv2.imread("CROP1.JPG",0)
#manual_canny(img)
#auto_canny(img)
Dilated_Canny(img)


