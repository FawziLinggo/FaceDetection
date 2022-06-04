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
lendmark = "model/shape_predictor_68_face_landmarks.dat"
### Path ###

def box(crop,titik, scale=5):
    masking= np.zeros_like(crop)
    masking= cv2.fillPoly(masking,[titik], (255,255,255))
    crop = cv2.bitwise_and(crop, masking)
    cv2.namedWindow('Masking', cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Masking", crop)


    bbox = cv2.boundingRect(titik)
    x,y,w,h= bbox
    imgCrop = crop[y:y+h, x:x+w]
    imgCrop = cv2.resize(imgCrop,(0,0),None,scale,scale)
    return imgCrop

detector = dlib.get_frontal_face_detector()
predictor =dlib.shape_predictor(lendmark)

## Ganti Path sesuai yang diinginkan ##
img = cv2.imread(meter_100) #ganti path di sini
crop = img

##khusus 150 meter uncomment perintah berikut
#crop = img[600:5184, 1100:3456]  ##Shape image (3456, 5184, 3)
crop = cv2.resize(crop,(0,0), None,0.5,0.5)
imgOriginal = crop.copy()

imgGray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
faces = detector(imgGray)

for face in faces:
    x1,y1= face.left(),face.top()
    x2,y2=face.right(),face.bottom()

    #Perintah ini untuk menampilkan Bounding Box
    #imgOriginal=cv2.rectangle(crop,(x1,y1), (x2,y2), (0,255,255),4)
    landmarks = predictor(imgGray,face)
    titik=[]
    for n in range(68):
        x=landmarks.part(n).x
        y=landmarks.part(n).y
        titik.append([x,y])

        #kode ini untuk mengaktifkan landmarks
        #cv2.circle(imgOriginal,(x,y),5,(50,50,255),cv2.FILLED)

        #untuk mengetahui Point
        #cv2.putText(imgOriginal, str(n),(x,y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (0,0,255),5)

    titik = np.array(titik)
    faceBox = box(imgOriginal, titik)
    cv2.namedWindow('Hanya Wajah', cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Hanya Wajah",faceBox)
cv2.namedWindow('Gambar Keseluruhan', cv2.WINDOW_KEEPRATIO)
cv2.imshow("Gambar Keseluruhan", imgOriginal)
cv2.waitKey(0)