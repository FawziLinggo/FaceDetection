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

""" Path """
""" 1 Meter """
meter_1 = "1-meter/"
meter_1 = meter_1 + random.choice(os.listdir(meter_1))

""" 60 Meter """
meter_60 = "60 meter/"
meter_60 = meter_60 + random.choice(os.listdir(meter_60))

""" 100 Meter """
meter_100 = "100 meter/"
meter_100 = meter_100 + random.choice(os.listdir(meter_100))

""" 150 Meter """
meter_150 = "150 meter/"
meter_150 = meter_150 + random.choice(os.listdir(meter_150))

""" Path Penyimpanan Homomorpich"""
path_save_homomorpic = "images/HasilFiltering/HomomorphicFilter-"
path_save_face_detection = "images/HasilFiltering/FaceDetection-"

""" Untuk Path Foto """
path =""

""" Path Detector """
detector = dlib.get_frontal_face_detector()

def run_program(path):
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
            img = cv2.imread(path)[:, :, 0]
            homo_filter = HomomorphicFilter(a=0.75, b=1.25)
            img_filtered = homo_filter.filter(I=img, filter_params=[25, 4])
            img_filtered = cv2.equalizeHist(img_filtered)
            logging.info("Berhasil Melakukan filtering HomomorphicFilter")
            path_save_homomorpic += subject
            cv2.imwrite(path_save_homomorpic, img_filtered)
            logging.info("Berhasil Menyimpan gambar filtering HomomorphicFilter pada : %s ", path_save_homomorpic)

            face_detector(path_save_homomorpic,subject)
        if(Hello==4):
            img = cv2.imread(path)[:, :, 0]
            homo_filter = HomomorphicFilter(a=0.75, b=1.25)
            img_filtered = homo_filter.filter(I=img, filter_params=[25, 4])
            img_filtered = cv2.equalizeHist(img_filtered)
            img_filtered = img[1320:2552, 1640:3504]
            logging.info("Berhasil Melakukan filtering HomomorphicFilter")
            path_save_homomorpic += subject
            cv2.imwrite(path_save_homomorpic, img_filtered)
            logging.info("Berhasil Menyimpan gambar filtering HomomorphicFilter pada : %s ", path_save_homomorpic)
            morfologi(path_save_homomorpic)


        else:
            img = cv2.imread(path)[:, :, 0]
            homo_filter = HomomorphicFilter(a=0.75, b=1.25)
            img_filtered = homo_filter.filter(I=img, filter_params=[25, 4])
            img_filtered = cv2.equalizeHist(img_filtered)
            img_filtered = img[1320:2552, 1640:3504]
            logging.info("Berhasil Melakukan filtering HomomorphicFilter")
            path_save_homomorpic += subject
            cv2.imwrite(path_save_homomorpic, img_filtered)
            logging.info("Berhasil Menyimpan gambar filtering HomomorphicFilter pada : %s ", path_save_homomorpic)

            logging.info("Memanggil fungsi Face Detector")
            face_detector(path_save_homomorpic,subject)



    except:
        logging.error("Error saat melakukan Filtering : %s", OSError)

def face_detector(path_save_homomorpic,subject, path_save_face_detection=path_save_face_detection):
    img = cv2.imread(path_save_homomorpic)
    crop = img
    imgGray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    faces = detector(imgGray)
    # face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')
    # faces = face_cascade.detectMultiScale(imgGray, 1.1, 4)
    # for (x, y, w, h) in faces:
    #     kotak = cv2.rectangle(crop, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #     kotak_saja = kotak[y:h, x:w]
    #
    #     logging.info("Sukses Membuat bounding box")
    #
    #     path_save_face_detection_box = path_save_face_detection + "box-" + subject
    #     path_save_face_detection += subject
    #     cv2.imwrite(path_save_face_detection, kotak_saja)
    #     cv2.imwrite(path_save_face_detection_box, kotak)
    #     logging.info("Menyimpan hasil bounding box")
    #     morfologi(path_save_face_detection)
    #
    for face in faces:
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()

        """Perintah ini untuk menampilkan Bounding Box"""
        kotak = cv2.rectangle(crop, (x1, y1), (x2, y2), (0, 255, 255), 4)
        kotak_saja = kotak[y1:y2, x1:x2]
        logging.info("Sukses Membuat bounding box")

        path_save_face_detection_box = path_save_face_detection+ "box-" + subject
        path_save_face_detection += subject
        cv2.imwrite(path_save_face_detection,kotak_saja)
        cv2.imwrite(path_save_face_detection_box,kotak)
        logging.info("Menyimpan hasil bounding box")
        morfologi(path_save_face_detection)

def morfologi(path_save_face_detection):
    img = cv2.imread(path_save_face_detection, 0)
    binr = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernel = np.ones((5, 5), np.uint8)

    # invert the image
    invert = cv2.bitwise_not(binr)

    # erode the image
    erosion = cv2.erode(invert, kernel, iterations=1)
    logging.info("Berhasil melakukan Morfologi erosi")

    plt.figure(figsize=(8, 7))
    plt.subplot(2, 2, 1)
    plt.imshow(erosion, cmap='gray')
    plt.title('erosion')
    plt.axis('off')
    logging.info("Menampilkan Morfologi erosi")

    # define the kernel
    kernel = np.ones((3, 3), np.uint8)
    invert = cv2.bitwise_not(binr)
    dilation = cv2.dilate(invert, kernel, iterations=1)
    logging.info("Berhasil melakukan Morfologi dilasi")

    # print the output
    plt.subplot(2, 2, 2)
    plt.imshow(dilation, cmap='gray')
    plt.title('dilation')
    plt.axis('off')
    logging.info("Menampilkan Morfologi dilasi")


    # define the kernel
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binr, cv2.MORPH_OPEN,
                               kernel, iterations=1)
    logging.info("Berhasil melakukan Morfologi opening")

    # print the output
    plt.subplot(2, 2, 3)
    plt.imshow(opening, cmap='gray')
    plt.title('opening')
    plt.axis('off')
    logging.info("Menampilkan Morfologi opening")


    kernel = np.ones((3, 3), np.uint8)

    # opening the image
    closing = cv2.morphologyEx(binr, cv2.MORPH_CLOSE, kernel, iterations=1)
    logging.info("Berhasil melakukan Morfologi closing")


    # print the output
    plt.subplot(2, 2, 4)
    plt.imshow(closing, cmap='gray')
    plt.title('closing')
    plt.axis('off')
    logging.info("Menampilkan Morfologi opening")
    plt.show()

