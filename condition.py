""" untuk Kebutuhan logging """
import logging
import sys
import matplotlib.image as mpimg
import os, random
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from math import log10, sqrt

""" import Class HomomorphicFilter """
from filtering import HomomorphicFilter

# ---------------------------------- PENJELASAN ----------------------------------
#
#   Ini adalah perintah basic logging dengan format :
#   %(asctime)s - %(levelname)s - %(message)s
#
#   Dokumentasi bisa dibaca disini :
#   https://docs.python.org/3/library/logging.html
#
# --------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, filename='app.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------------- PENJELASAN ----------------------------------
#
#   :path:  Untuk Path Foto yang akan mengembalikan setiap fungsi (path)
#   :path_save_homomorpic: Tempat menyimpan hasil filtering hormomorphic
#   :path_save_face_detection: Tempat menyimpan hasil face detection
#   :detector: Tempat mengambil file dlib
#   :lendmark_path: Tempat mengambil file landmarks.dat
#
# --------------------------------------------------------------------------------
path = ""
path_save_homomorpic = "images/HasilFiltering/grayscale-"
path_save_face_detection = "images/HasilFiltering/FaceDetection-"
detector = dlib.get_frontal_face_detector()
lendmark_path = "model/shape_predictor_68_face_landmarks.dat"


# ---------------------------------- PENJELASAN ----------------------------------
#
#   fungsi run_program() akan mengembalikan nilai dari variabel path yang
#   akan digunakan untuk menginisialisasi path kosong. selanjutnya dilakukan
#   try except, dimana saat menjalankan program ini dia akan mengeceh apakan ada
#   file masking2.png, jika ada maka akan menghapus terlebih dahulu. jika tidak
#   maka akan melanjutkan program dengan memberikan info pada log.
#
#   selanjutnya kita program akan meminta input yang akan diinisialisasi oleh
#   variabel "Hello_". untuk jika kita memasukkan (meng-input) nilai 1 maka akan
#   digunakan citra foto Nir sedangkan nilai 2 akan menggunakan citra asli. dan 3
#   untuk exit.
#
#   Kemudian akan diminta untuk menginput data foto dari jarak berapa, mulai dari
#   1, 60 hingga 100 meter. yang akan diinisialisasi dengan variabel "Hello"
#
# --------------------------------------------------------------------------------
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
    Hello_ = int(input("Masukkan Angka : "))
    if (Hello_ == 1):
        a = "citra-nir/"

    elif (Hello_ == 2):
        a = "citra-vis/"

    elif (Hello_ == 2):
        logging.info("Clossing Program")
        sys.exit()
    else:
        print(" ERROR : must number :)")

    # ---------------------------------- PENJELASAN ----------------------------------
    #
    #   Ini adalah inisialisasi jarak, masing masing jarak akan menentukan directorinya
    #   dilanjutkan dengan memilih secara acak file yang ada dalam directory tersebut
    #
    # --------------------------------------------------------------------------------
    meter_1 = a + "1-meter/"
    meter_1 = meter_1 + random.choice(os.listdir(meter_1))
    meter_60 = a + "60 meter/"
    meter_60 = meter_60 + random.choice(os.listdir(meter_60))
    meter_100 = a + "100 meter/"
    meter_100 = meter_100 + random.choice(os.listdir(meter_100))
    meter_150 = a + "150 meter/"
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

    # ---------------------------------- PENJELASAN ----------------------------------
    #
    #   Setelah menentukan jarak berapa maka nilai Hello akan dilakukan statement if-
    #   else, jika 1 maka... jika 2 maka... fan seterusnya. di dalam kondisi tersebut
    #   sudah termasuk didalamnya untuk menjalankan fungsi homomorphic() yang ada di
    #   dalam program ini, ayang akan mengembalikan dilai dari path, subject, Hello. dimana
    #   subject yang akan dighunakan sebagau nama foto (name file) yang mewakili jarak foto
    #   tersebut
    #
    # --------------------------------------------------------------------------------

    if (Hello == 1):
        subject = "1-meter.jpg"
        path += meter_1
        logging.info("Menjalankan Program dengan jarak 1 meter")
        homomorphic(path, subject, Hello, Hello_)

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


# ---------------------------------- PENJELASAN ----------------------------------
#
#   Ini adalah fungi homomorphic() untuk melakukan proses homomorphic filter, dimulai
#   dengan membaca program menggunakan cv2.imread selanjutnya di ubah menjadi
#   grayscale[:,:,0] dan selanjutnya adalah pengembalian ke dalam class HomomorphicFilter()
#   yang akan menginisiaisasi nilai a dan b. dimana nilai a dan b adalah float yang digunakan
#   untuk emphasis filter dengan rumus :
#    H = a + b*H
#
#   Langkah selanjutnya adalah pemanggilan kembali fungsi filter() pada kelas HomomorphicFilter()
#   ini merupakan Metode untuk menerapkan filter homomorfik pada gambar (image).
#
#   def filter(self, I, filter_params, filter='butterworth', H=None):
#
#   dengan Atribut:
#   I: Gambar saluran tunggal
#   filter_params: Parameter yang akan digunakan pada filter:
#
#       1. Butterworth:
#           filter_params[0]: Frekuensi batas
#           filter_params[1]: Urutan filter
#       2. gaussian:
#           filter_params[0]: Frekuensi batas
#           filter: Pilih filter, opsi:
#                   butterworth
#                   gaussian
#                   external
#       H: Digunakan untuk melewati filter eksternal
#
#   fungsi homomorphic() akan mengembalikan nilai variabel berikut :
#     :param path:
#     :param subject:
#     :param Hello:
#     :param Hello_:
#     :param path_save_homomorpic:
#     :return:
#
#   Hasil dari path_save_homomorpic akan menjadi nilai return untuk fungsi
#   morfologi() pada kodingan ini
# --------------------------------------------------------------------------------

def homomorphic(path, subject, Hello, Hello_=False, path_save_homomorpic=path_save_homomorpic):
    try:
        if (Hello == 1 & Hello_ == 2):
            masking(path)
            img = cv2.imread(path)
            # logging.info("Berhasil Melakukan filtering HomomorphicFilter")
            path_save_homomorpic += subject
            cv2.imwrite(path_save_homomorpic, img)
            logging.info("Berhasil Menyimpan gambar filtering HomomorphicFilter pada : %s ", path_save_homomorpic)
            morfologi(path_save_homomorpic)

        else:
            if (Hello == 1):
                masking(path)
                img = cv2.imread(path)
                # logging.info("Berhasil Melakukan filtering HomomorphicFilter")
                path_save_homomorpic += subject
                cv2.imwrite(path_save_homomorpic, img)
                logging.info("Berhasil Menyimpan gambar filtering HomomorphicFilter pada : %s ", path_save_homomorpic)
                face_detector(path_save_homomorpic, subject)
                sys.exit()

            if (Hello == 4):
                masking(path)
                img = cv2.imread(path)
                # logging.info("Berhasil Melakukan filtering HomomorphicFilter")
                path_save_homomorpic += subject
                cv2.imwrite(path_save_homomorpic, img)
                logging.info("Berhasil Menyimpan gambar filtering HomomorphicFilter pada : %s ", path_save_homomorpic)
                morfologi(path_save_homomorpic)

            else:
                masking(path)
                img = cv2.imread(path)
                # logging.info("Berhasil Melakukan filtering HomomorphicFilter")
                path_save_homomorpic += subject
                cv2.imwrite(path_save_homomorpic, img)
                logging.info("Berhasil Menyimpan gambar filtering HomomorphicFilter pada : %s ", path_save_homomorpic)

                logging.info("Memanggil fungsi Face Detector")
                face_detector(path_save_homomorpic, subject)
                # cv2.imwrite(path_save_homomorpic, img_filtered)

    except:
        logging.error("Error saat melakukan Filtering : %s", OSError)


# ---------------------------------- PENJELASAN ----------------------------------
#
#   Fungsi face_detector adalah fungsi untuk mendeteksi wajah dengan mengambil path
#   dari path_save_homomorpic yang selanjutnya diubah menjadi Gray, kemudial akan
#   dilakukan prediksi wajah dengan fungsi detector dari dlib selanjutnya dilakukan
#   perulangan yang akan menginisialisasi face dengan menciptakan bounding box
#   x1,y1, x2,y2 seperti pada kodingan. serta memanggil fungsi landmark untuk
#   menciptakan titik croping pada wajah yang sudah terdeteksi menggunakan bounding
#   box sebelumnya. selanjutnya fungsi cv2.imwrite hanya untuk melakukan penyimpanan
#   gambar hasil deteksi wajah.
#
# --------------------------------------------------------------------------------
def face_detector(path_save_homomorpic, subject, path_save_face_detection=path_save_face_detection):
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

        path_save_face_detection_box = path_save_face_detection + "box-" + subject
        path_save_face_detection_kotak = path_save_face_detection + "box-full-" + subject
        path_save_face_detection += subject
        cv2.imwrite(path_save_face_detection, faceBox)
        cv2.imwrite(path_save_face_detection_box, kotak_saja)
        cv2.imwrite(path_save_face_detection_kotak, kotak)
        logging.info("Menyimpan hasil bounding box")
        morfologi(path_save_face_detection)


# ---------------------------------- PENJELASAN ----------------------------------
#
#   Fungsi morfologi() akan memanggil path path_save_face_detection, selanjutnya akan
#   dilakukan beberapa fungsi morfologi citra di antaranya threshold yang menggunakan
#   cv2.THRESH_OTSU + cv2.THRESH_BINARY, kemudian  cv2.dilate atau dilatasi yang menggunakan
#   kernel :
#       np.ones((25, 25), np.uint16)
#
#   kemudian dilakukan perulangan sebanyak 7 kali, np.ones() adalaha function mengembalikan
#   array baru dengan bentuk dan tipe data tertentu, di mana nilai elemen diatur ke 1.
#   Fungsi ini sangat mirip dengan fungsi numpy zeros().
#
#   selanjutnya dilakukan morfologi dengan closing, dimana kernel yang digunakan adalah
#   kernel :
#       np.ones((25, 25), np.uint16)
#   kemudian dilakukan perulangan 10 kali dengan memanggil nilai cv2.MORPH_CLOSE
#
#   terakhir hanya menampilkan gambar dengan subplot (2,2,(1-4))
#
#       --------------------+--------------------
#           Gambar 1        |   Gambar 2
#       --------------------+--------------------
#           Gambar 3        |   Gambar 4
#       --------------------+--------------------
#
# --------------------------------------------------------------------------------
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

        # show image real for subplot (2,2,4)
        cv2.imwrite("images/Masking/Masking3.png", src)
        show_ = mpimg.imread("images/Masking/Masking3.png")

        value = PSNR(img_masking_2, src)
        if value != 100:
            plt.subplot(2, 2, 4)
            plt.imshow(show_)
            plt.title('PSNR : %.2f, MSE : %s ' % (value[0], value[1]))
            plt.axis('off')
            logging.info("Menampilkan Morfologi Masking")
        else:
            # print the output
            plt.subplot(2, 2, 4)
            plt.imshow(show_)
            plt.title('PSNR : %.2f, MSE : 0 ' % value)
            plt.axis('off')
            logging.info("Menampilkan Morfologi Masking")

        plt.show()
        logging.info("Clossing Program")
        sys.exit()
    except:
        try:
            # print("tes")
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


def box(img, titik, scale=5):
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
            cv2.imwrite("images/Masking/masking2.png", faceBox)
    except:
        print("gagal prediksi wajah")


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    print(mse)
    if (mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr, mse
