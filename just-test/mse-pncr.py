from math import log10, sqrt
import cv2
import numpy as np

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr, mse


def main():
    original = cv2.imread("/home/adi/PycharmProjects/FaceDetection-main/images/Masking/masking2.png")

    img_masking = cv2.imread("/home/adi/PycharmProjects/FaceDetection-main/images/Masking/Masking.png")
    img_masking_2 = cv2.resize(original, img_masking.shape[1::-1])

    src = cv2.bitwise_and(img_masking, img_masking_2)
    src = cv2.resize(src, img_masking_2.shape[1::-1])
    cv2.imwrite("/home/adi/PycharmProjects/FaceDetection-main/images/Masking/Masking3.png",src)


    # compressed = cv2.imread(original, 1)
    value= PSNR(img_masking_2, src)
    if value == 100:
        print("mse = 0")
        print(f"PSNR value is {value} dB")
    else:
        print("PSNR :", value[0])
        print("MSE  :", value[1])
        # print(f"PSNR value is {value} dB")
    # print(f"MSE value is {mse} ")
    # cv2.namedWindow("window_name", cv2.WINDOW_KEEPRATIO)
    # cv2.imshow("window_name", src)
    # cv2.waitKey(0)

if __name__ == "__main__":
    main()