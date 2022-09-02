""" untuk Kebutuhan logging """
import logging
import sys

logging.basicConfig(level=logging.INFO, filename='app.log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')

import os, random
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt


""" import Class HomomorphicFilter """
from filtering import HomomorphicFilter

""" Path Penyimpanan Homomorpich"""
path_save_homomorpic = "images/HasilFiltering/HomomorphicFilter-"
path_save_face_detection = "images/HasilFiltering/FaceDetection-"

""" Untuk Path Foto """
path =""

""" Path Detector """
detector = dlib.get_frontal_face_detector()
lendmark_path = "model/shape_predictor_68_face_landmarks.dat"

def run_program(path):
        try:
            os.remove("images/Masking/masking2.png")
        except:
            logging.info("tidak ada file masking2.png")

        print("\n ++++++ Jenis-Jenis Foto ++++++ \n"
              "1. Citra Nir\n"
              "2. Citra Vis \n"
              "3. Exit Program \n"
              " +++++++++++++++++++++++++++++++"
              )
        Hello = int(input("Masukkan Angka : "))
        if (Hello == 1):
            a = "citra-nir/"

        elif (Hello == 2):
            a = "citra-vis/"

        elif (Hello == 2):
            logging.info("Clossing Program")
            sys.exit()
        else:
            print(" ERROR : must number :)")

        """ Path """
        """ 1 Meter """
        meter_1 = a+"1-meter/"
        meter_1 = meter_1 + random.choice(os.listdir(meter_1))

        """ 60 Meter """
        meter_60 = a+"60 meter/"
        meter_60 = meter_60 + random.choice(os.listdir(meter_60))

        """ 100 Meter """
        meter_100 = a + "100 meter/"
        meter_100 = meter_100 + random.choice(os.listdir(meter_100))

        """ 150 Meter """
        meter_150 = a+"150 meter/"
        meter_150 = meter_150 + random.choice(os.listdir(meter_150))

        print("\n ++++++ Jenis-Jenis Foto ++++++ \n"
              "1. jarak 1 Meter \n"
              "2. jarak 60 Meter \n"
              "3. jarak 100 Meter \n"
              "4. jarak 150 Meter \n"
              "5. Exit Program \n"
              " +++++++++++++++++++++++++++++++"
              )
        Hello = int(input("Masukkan Angka : "))
        if (Hello == 1):
            subject = "1-meter.jpg"
            path += meter_1
            logging.info("Menjalankan Program dengan jarak 1 meter")
            homomorphic(path, subject, Hello)

        elif (Hello == 2):
            subject = "60-meter.jpg"
            path += meter_60
            logging.info("Menjalankan Program dengan jarak 60 meter")
            homomorphic(path, subject, Hello)

        elif (Hello == 3):
            subject = "100-meter.jpg"
            path += meter_100
            logging.info("Menjalankan Program dengan jarak 100 meter")
            homomorphic(path, subject, Hello)

        elif (Hello == 4):
            subject = "150-meter.jpg"
            path += meter_150
            logging.info("Menjalankan Program dengan jarak 150 meter")
            homomorphic(path, subject, Hello)

        elif (Hello == 5):
            logging.info("Clossing Program")
            sys.exit()

        else:
            print(" ERROR : must number :)")


def homomorphic(path, subject,Hello, path_save_homomorpic=path_save_homomorpic):
    try:
        if(Hello==1):
            masking(path)
            img = cv2.imread(path)[:, :, 0]
            homo_filter = HomomorphicFilter(a=0.75, b=1.25)
            img_filtered = homo_filter.filter(I=img, filter_params=[25], filter='gaussian')
            img_filtered = cv2.equalizeHist(img_filtered)
            logging.info("Berhasil Melakukan filtering HomomorphicFilter")
            path_save_homomorpic += subject
            cv2.imwrite(path_save_homomorpic, img_filtered)
            logging.info("Berhasil Menyimpan gambar filtering HomomorphicFilter pada : %s ", path_save_homomorpic)
            face_detector(path_save_homomorpic,subject)
            sys.exit()

        if(Hello==4):
            masking(path)
            img = cv2.imread(path)[:, :, 0]
            homo_filter = HomomorphicFilter(a=0.75, b=1.25)
            img_filtered = homo_filter.filter(I=img, filter_params=[25], filter='gaussian')
            img_filtered = cv2.equalizeHist(img_filtered)
            img_filtered = img_filtered[1320:2552, 1640:3504]
            logging.info("Berhasil Melakukan filtering HomomorphicFilter")
            path_save_homomorpic += subject
            cv2.imwrite(path_save_homomorpic, img_filtered)
            logging.info("Berhasil Menyimpan gambar filtering HomomorphicFilter pada : %s ", path_save_homomorpic)
            morfologi(path_save_homomorpic)

        else:
            masking(path)
            img = cv2.imread(path)[:, :, 0]
            homo_filter = HomomorphicFilter(a=0.75, b=1.25)
            img_filtered = homo_filter.filter(I=img, filter_params=[25], filter='gaussian')
            img_filtered = cv2.equalizeHist(img_filtered)
            img_filtered_ = img_filtered[1320:2552, 1640:3504]
            logging.info("Berhasil Melakukan filtering HomomorphicFilter")
            path_save_homomorpic += subject
            cv2.imwrite(path_save_homomorpic, img)
            logging.info("Berhasil Menyimpan gambar filtering HomomorphicFilter pada : %s ", path_save_homomorpic)

            logging.info("Memanggil fungsi Face Detector")
            face_detector(path_save_homomorpic,subject)
            #cv2.imwrite(path_save_homomorpic, img_filtered)

    except:
        logging.error("Error saat melakukan Filtering : %s", OSError)

def face_detector(path_save_homomorpic,subject, path_save_face_detection=path_save_face_detection):
    crop = cv2.imread(path_save_homomorpic)
    imgGray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    faces = detector(imgGray)

    predictor = dlib.shape_predictor(lendmark_path)

    for face in faces:
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()

        landmarks = predictor(imgGray, face)
        titik = []
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            titik.append([x, y])

        titik = np.array(titik)
        titik = cv2.convexHull(titik)
        faceBox = box(crop, titik)


        kotak = cv2.rectangle(crop, (x1, y1), (x2, y2), (0, 255, 255), 4)
        kotak_saja = kotak[y1:y2, x1:x2]

        path_save_face_detection_box = path_save_face_detection+ "box-" + subject
        path_save_face_detection_kotak = path_save_face_detection+ "box-full-" + subject
        path_save_face_detection += subject
        cv2.imwrite(path_save_face_detection,faceBox)
        cv2.imwrite(path_save_face_detection_box,kotak_saja)
        cv2.imwrite(path_save_face_detection_kotak, kotak)
        logging.info("Menyimpan hasil bounding box")
        morfologi(path_save_face_detection)

def morfologi(path_save_face_detection):
    img = cv2.imread(path_save_face_detection, 0)
    binr = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]

    try:
        img_masking_2 = cv2.imread("images/Masking/masking2.png")
        plt.figure(figsize=(8, 7))
        plt.subplot(2, 2, 1)
        plt.imshow(binr, cmap='gray')
        plt.title('Threshold')
        plt.axis('off')
        logging.info("Menampilkan Threshold Image")

        kernel = np.ones((25, 25), np.uint16)
        dilation = cv2.dilate(binr, kernel, iterations=7)
        logging.info("Berhasil melakukan Morfologi dilatasi dengan 7 kali perulangan")

        plt.subplot(2, 2, 2)
        plt.imshow(dilation, cmap='gray')
        plt.title('dilation')
        plt.axis('off')
        logging.info("Menampilkan Morfologi dilation")

        kernel1 = np.ones((25, 25), np.uint16)
        closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel1, iterations=10)

        # print the output
        plt.subplot(2, 2, 3)
        plt.imshow(closing, cmap='gray')
        plt.title('closing')
        plt.axis('off')

        logging.info("Menampilkan Morfologi dilasi")
        cv2.imwrite("images/Masking/Masking.png", closing)
        img_masking = cv2.imread("images/Masking/Masking.png")
        img_masking_2 = cv2.resize(img_masking_2, img_masking.shape[1::-1])
        src = cv2.bitwise_and(img_masking, img_masking_2)
        # print the output
        plt.subplot(2, 2, 4)
        plt.imshow(src, cmap='gray')
        plt.title('Masking')
        plt.axis('off')
        logging.info("Menampilkan Morfologi Masking")
        plt.show()
        logging.info("Clossing Program")
        sys.exit()
    except:
        try:
            os.remove("images/Masking/masking2.png")
        except:
            logging.info("tidak ada file masking2.png")
            plt.close()
            kernel = np.ones((7, 7), np.uint16)
            dilation = cv2.dilate(binr, kernel, iterations=1)

            plt.subplot(2, 2, 1)
            plt.imshow(binr, cmap='gray')
            plt.title('Threshold')
            plt.axis('off')
            logging.info("Menampilkan Threshold Image")

            plt.subplot(2, 2, 2)
            plt.imshow(dilation, cmap='gray')
            plt.title('dilation')
            plt.axis('off')
            logging.info("Menampilkan Morfologi dilation")

            kernel1 = np.ones((7, 7), np.uint16)
            closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel1, iterations=2)

            # print the output
            plt.subplot(2, 2, 3)
            plt.imshow(closing, cmap='gray')
            plt.title('closing')
            plt.axis('off')

            plt.subplot(2, 2, 4)
            plt.title("gagal menentukan countur wajah", color='red')
            plt.axis('off')

            plt.show()
            logging.info("Clossing Program")
            sys.exit()


def box(img,titik, scale=5):
    masking = np.zeros_like(img)
    masking = cv2.fillPoly(masking, [titik], (255, 255, 255))
    img = cv2.bitwise_and(img, masking)

    bbox = cv2.boundingRect(titik)
    x, y, w, h = bbox
    imgimg = img[y:y + h, x:x + w]
    imgimg = cv2.resize(imgimg, (0, 0), None, scale, scale)
    return imgimg

def masking(path):
    try:
        imgGray = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        faces = detector(imgGray)
        predictor = dlib.shape_predictor(lendmark_path)

        for face in faces:
            # x1, y1 = face.left(), face.top()
            # x2, y2 = face.right(), face.bottom()

            landmarks = predictor(imgGray, face)
            titik = []
            for n in range(68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                titik.append([x, y])

            titik = np.array(titik)
            titik = cv2.convexHull(titik)
            faceBox = box(cv2.imread(path), titik)
            cv2.imwrite("images/Masking/masking2.png",faceBox)
    except:
        print("gagal prediksi wajah")



