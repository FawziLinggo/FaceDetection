import sys
import matplotlib.image as mpimg
import os
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from math import log10, sqrt

try:
    os.remove("images/Masking/masking2.png")
except:
    pass
try:
    os.remove("images/Masking/masking3.png")
except:
    pass
try:
    os.remove("images/Masking/masking.png")
except:
    pass

detector = dlib.get_frontal_face_detector()
lendmark_path = "model/shape_predictor_68_face_landmarks.dat"

# path = input("Enter Photo Path : \n ")
path="citra-matlab/1-meter/WhatsApp Image 2022-11-10 at 19.43.56.jpeg"

img = cv2.imread(path)
faces = detector(img)
predictor = dlib.shape_predictor(lendmark_path)



def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    print(mse)
    if (mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr, mse


def box(img, titik, scale=5):
    masking = np.zeros_like(img)
    masking = cv2.fillPoly(masking, [titik], (255, 255, 255))
    img = cv2.bitwise_and(img, masking)

    bbox = cv2.boundingRect(titik)
    x, y, w, h = bbox
    imgimg = img[y:y + h, x:x + w]
    imgimg = cv2.resize(imgimg, (0, 0), None, scale, scale)
    return imgimg

try:
    for face in faces:
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        landmarks = predictor(img, face)
        titik = []
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            titik.append([x, y])

        titik = np.array(titik)
        titik = cv2.convexHull(titik)
        kotak = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 4)
        cv2.imwrite("images/Masking/kotaksaja.png", kotak)
        faceBox = box(cv2.imread(path), titik)
        cv2.imwrite("images/Masking/masking2.png", faceBox)


    img = cv2.imread("images/Masking/masking2.png", 0)
    binr = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
    img_masking_2 = img
    plt.figure(figsize=(8, 7))
    plt.subplot(2, 2, 1)
    plt.imshow(binr, cmap='gray')
    plt.title('Threshold')
    plt.axis('off')
    print("Menampilkan Threshold Image")

    kernel = np.ones((25, 25), np.uint16)
    dilation = cv2.dilate(binr, kernel, iterations=2)
    print("Berhasil melakukan Morfologi dilatasi dengan 1 kali perulangan")

    plt.subplot(2, 2, 2)
    plt.imshow(dilation, cmap='gray')
    plt.title('dilation')
    plt.axis('off')
    print("Menampilkan Morfologi dilation")

    kernel1 = np.ones((25, 25), np.uint16)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel1, iterations=10)

    # print the output
    plt.subplot(2, 2, 3)
    plt.imshow(closing, cmap='gray')
    plt.title('closing')
    plt.axis('off')

    print("Menampilkan Morfologi dilasi")
    cv2.imwrite("images/Masking/Masking.png", closing)
    img_masking = cv2.imread("images/Masking/Masking.png",0)
    img_masking_2 = cv2.resize(img_masking_2, img_masking.shape[1::-1])
    src = cv2.bitwise_and(img_masking, img_masking_2)

    # show image real for subplot (2,2,4)
    cv2.imwrite("images/Masking/Masking3.png", src)
    show_ = mpimg.imread("images/Masking/Masking3.png")

    value = PSNR(img_masking_2, src)
    if value == 100:
        # print the output
        plt.subplot(2, 2, 4)
        plt.imshow(show_ , cmap='gray')
        plt.title('PSNR : '+str(value)+', MSE : 0 ')
        plt.axis('off')
        print("Menampilkan Morfologi Masking")
    else:
        plt.subplot(2, 2, 4)
        plt.imshow(show_, cmap='gray')
        plt.title('PSNR : %.2f, MSE : %s ' % (value[0], value[1]))
        plt.axis('off')
        print("Menampilkan Morfologi Masking")

    plt.show()
    print("Clossing Program")


except Exception as e:
    print(e)
    img = cv2.imread(path,0)
    binr = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
    kernel = np.ones((7, 7), np.uint16)
    dilation = cv2.dilate(binr, kernel, iterations=7)

    plt.subplot(2, 2, 1)
    plt.imshow(binr, cmap='gray')
    plt.title('Threshold')
    plt.axis('off')
    print("Menampilkan Threshold Image")

    plt.subplot(2, 2, 2)
    plt.imshow(dilation, cmap='gray')
    plt.title('dilation')
    plt.axis('off')
    print("Menampilkan Morfologi dilation")

    kernel1 = np.ones((7, 7), np.uint16)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel1, iterations=1)

    # print the output
    plt.subplot(2, 2, 3)
    plt.imshow(closing, cmap='gray')
    plt.title('closing')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title("gagal menentukan countur wajah", color='red')
    plt.axis('off')

    plt.show()
    print("Clossing Program")
