import cv2
import numpy as np

filename = 'image4.jpg'
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
# cv2.cornerHarris(image, blockSize, ksize, k)
dst = cv2.cornerHarris(gray,2,3,0.04)
cv2.imshow('Harris Corner detector',dst)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)
cv2.imshow('Harris Corner dilated',dst)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,255,0]

cv2.imshow('dst',img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()