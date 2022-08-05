import cv2
import dlib
import numpy as np
import os,random


### Path ###
meter_1="1-meter/"
meter_1 = meter_1 + random.choice(os.listdir(meter_1))
meter_60="60 meter/"
meter_60 = meter_60 + random.choice(os.listdir(meter_60))
meter_100="100 meter/"
meter_100 = meter_100 + random.choice(os.listdir(meter_100))
meter_150="150 meter/"
meter_150 = meter_100 + random.choice(os.listdir(meter_150))
lendmark_path = "model/shape_predictor_68_face_landmarks.dat"
### Path ###

def box(img,titik, scale=5, masked=False, cropped=True):
    if masked:
        masking = np.zeros_like(img)
        masking = cv2.fillPoly(masking, [titik], (255, 255, 255))
        img = cv2.bitwise_and(img, masking)

        # Masking
        cv2.namedWindow('Masking', cv2.WINDOW_KEEPRATIO)
        cv2.imshow("Masking", masking)

        imgMasking = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, imgMasking_threshold = cv2.threshold(imgMasking, 120, 255, cv2.THRESH_BINARY)
        ret, imgMasking_otsu = cv2.threshold(imgMasking, 120, 255, cv2.THRESH_OTSU)
        ret, imgMasking_otsu_threshold = cv2.threshold(imgMasking_otsu, 120, 255, cv2.THRESH_BINARY)

        cv2.namedWindow('Threshold', cv2.WINDOW_KEEPRATIO)
        cv2.imshow("Threshold", imgMasking_threshold)
        cv2.namedWindow('Otsu', cv2.WINDOW_KEEPRATIO)
        cv2.imshow("Otsu", imgMasking_otsu)
        cv2.namedWindow('Threshold_Otsu', cv2.WINDOW_KEEPRATIO)
        cv2.imshow("Threshold_Otsu", imgMasking_otsu_threshold)

    if cropped:
        bbox = cv2.boundingRect(titik)
        x, y, w, h = bbox
        imgimg = img[y:y + h, x:x + w]
        imgimg = cv2.resize(imgimg, (0, 0), None, scale, scale)
        return imgimg
    else:
        return masking

detector = dlib.get_frontal_face_detector()
predictor =dlib.shape_predictor(lendmark_path)

## Ganti Path sesuai yang diinginkan ##
img = cv2.imread(meter_1)

##khusus 150 meter uncomment perintah berikut
#img = img[600:5184, 1100:3456]  ##Shape image (3456, 5184, 3)
#img = cv2.resize(img,(0,0), None,0.5,0.5)
imgOriginal = img.copy()

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = detector(imgGray)

for face in faces:
    x1,y1= face.left(),face.top()
    x2,y2=face.right(),face.bottom()

    #Perintah ini untuk menampilkan Bounding Box
    #imgOriginal=cv2.rectangle(img,(x1,y1), (x2,y2), (0,255,255),4)
    landmarks = predictor(imgGray,face)
    titik=[]
    for n in range(68):
        x=landmarks.part(n).x
        y=landmarks.part(n).y
        titik.append([x,y])

        #kode ini untuk mengaktifkan landmarks
        # cv2.circle(imgOriginal,(x,y),5,(50,50,255),cv2.FILLED)

        #untuk mengetahui Point
        # cv2.putText(imgOriginal, str(n),(x,y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 4, (0,0,255),5)

    titik = np.array(titik)
    titik = cv2.convexHull(titik)
    faceBox = box(imgOriginal, titik, masked=True)
    cv2.namedWindow('Face Box', cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Face Box",faceBox)

    # For Masking Color
    # faceBox_ = box(imgOriginal, titik, cropped=False, masked=True)
    # # faceBoxOnly = np.zeros_like(img)
    # # faceBoxOnly[:] = 153,0,157
    # # faceBoxOnly = cv2.bitwise_and(faceBox_,faceBoxOnly)
    # cv2.namedWindow('Face Box Only', cv2.WINDOW_KEEPRATIO)
    # cv2.imshow("Face Box Only",faceBox_)

#cv2.namedWindow('Gambar Keseluruhan', cv2.WINDOW_KEEPRATIO)
#cv2.imshow("Gambar Keseluruhan", imgOriginal)
cv2.waitKey(0)