import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)


# matplotlib.rcParams['font.size'] = 9
#
# img = Image.open("images/HasilFiltering/HomomorphicFilter-1-meter.jpg")
# img = np.array(img)
# image = img
# binary_global = image > threshold_otsu(image)
#
# window_size = 25
# thresh_niblack = threshold_niblack(image, window_size=window_size, k=0.5)
# thresh_sauvola = threshold_sauvola(image, window_size=window_size)
#
# binary_niblack = image > thresh_niblack
# binary_sauvola = image > thresh_sauvola
#
# plt.figure(figsize=(8, 7))
# plt.subplot(2, 2, 1)
# plt.imshow(image, cmap=plt.cm.gray)
# plt.title('Original')
# plt.axis('off')
#
# plt.subplot(2, 2, 2)
# plt.title('Global Threshold')
# plt.imshow(binary_global, cmap=plt.cm.gray)
# plt.axis('off')
#
# plt.subplot(2, 2, 3)
# plt.imshow(binary_niblack, cmap=plt.cm.gray)
# plt.title('Niblack Threshold')
# plt.axis('off')
#
# plt.subplot(2, 2, 4)
# plt.imshow(binary_sauvola, cmap=plt.cm.gray)
# plt.title('Sauvola Threshold')
# plt.axis('off')
#
# plt.show()

# from PIL import Image
# import cv2
#
# # Image.open() can also open other image types
# img60 = cv2.imread("60 meter/0011_60_n.JPG")
# img100 = cv2.imread("100 meter/0003_100_n.JPG")
# img150 = cv2.imread("150 meter/0009_150_n.JPG")
#
#
# """Shape of the image (3456, 5184, 3) """
# # print("Shape of the image", img60.shape)
# # print("Shape of the image", img100.shape)
# # print("Shape of the image", img150.shape)
# rows, cols, _ = img60.shape
# print(rows)
#
# crop= img150[1320:2552, 1640:3504]
#
# cv2.namedWindow('asli', cv2.WINDOW_KEEPRATIO)
# cv2.imshow("asli", img150)
# #
# cv2.namedWindow('zoom', cv2.WINDOW_KEEPRATIO)
# cv2.imshow("zoom", crop)
# cv2.waitKey(0)

# import the necessary packages
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("images/HasilFiltering/FaceDetection-1-meter.jpg", 0)

binr = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

kernel = np.ones((5, 5), np.uint8)

# invert the image
invert = cv2.bitwise_not(binr)

# erode the image
erosion = cv2.erode(invert, kernel,iterations=1)

plt.figure(figsize=(8, 7))
plt.subplot(2, 2, 1)
plt.imshow(erosion, cmap='gray')
plt.title('erosion')
plt.axis('off')

# define the kernel
kernel = np.ones((3, 3), np.uint8)

# invert the image
invert = cv2.bitwise_not(binr)

# dilate the image
dilation = cv2.dilate(invert, kernel, iterations=1)

# print the output
plt.subplot(2, 2, 2)
plt.imshow(dilation, cmap='gray')
plt.title('dilation')
plt.axis('off')

# define the kernel
kernel = np.ones((3, 3), np.uint8)

# opening the image
opening = cv2.morphologyEx(binr, cv2.MORPH_OPEN,
                           kernel, iterations=1)
# print the output
plt.subplot(2, 2, 3)
plt.imshow(opening, cmap='gray')
plt.title('opening')
plt.axis('off')

kernel = np.ones((3, 3), np.uint8)

# opening the image
closing = cv2.morphologyEx(binr, cv2.MORPH_CLOSE, kernel, iterations=1)

# print the output
plt.subplot(2, 2, 4)
plt.imshow(closing, cmap='gray')
plt.title('closing')
plt.axis('off')
plt.show()