import cv2
import matlab.engine
import scipy.io
from skimage import exposure, transform
from skimage import img_as_ubyte
from PIL import Image
from skimage.feature import hog

img_rows = 224
img_cols = 224

img = cv2.imread('dis_radio_(31).jpg')
img = cv2.resize(img, (img_rows, img_cols))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = exposure.equalize_adapthist(img, clip_limit=0.03)

eng = matlab.engine.start_matlab()
img = Image.fromarray(img , 'RGB')
image_mat = matlab.uint8(list(img.getdata()))
image_mat.reshape((img.size[0], img.size[1], 3))
ret = eng.anna_phog_demo(image_mat)
print(ret)