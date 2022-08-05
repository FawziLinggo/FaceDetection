import cv2
import dlib
import numpy as np
import os,random
# lendmark_path = "model/shape_predictor_68_face_landmarks.dat"
#
# detector = dlib.get_frontal_face_detector()
# predictor =dlib.shape_predictor(lendmark_path)

img2 = cv2.imread("images/HasilFiltering/FaceDetection-1-meter.jpg")


imgGray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

binr = cv2.threshold(imgGray2, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
invert=cv2.bitwise_not(binr)
kernel = np.ones((25, 25), np.uint16)
dilation = cv2.dilate(binr, kernel, iterations=7)


kernel1 = np.ones((25, 25), np.uint16)
closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel1, iterations=10)

real = cv2.imread("real.png")
cv2.namedWindow('Threshold', cv2.WINDOW_KEEPRATIO)
cv2.imshow("Threshold", binr)

cv2.namedWindow('dilasi', cv2.WINDOW_KEEPRATIO)
cv2.imshow("dilasi", dilation)

cv2.namedWindow('closing', cv2.WINDOW_KEEPRATIO)
cv2.imshow("closing", closing)

cv2.imwrite("closing.png", closing)

closing = cv2.imread("closing.png")
real = cv2.resize(real, closing.shape[1::-1])
src = cv2.bitwise_and(closing,real)

cv2.namedWindow('masking', cv2.WINDOW_KEEPRATIO)
cv2.imshow("masking", src)

cv2.waitKey(0)