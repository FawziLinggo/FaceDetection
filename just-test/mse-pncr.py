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

    # compressed = cv2.imread(original, 1)
    value,mse = PSNR(img_masking_2, src)
    print(f"PSNR value is {value} dB")
    print(f"MSE value is {mse} ")


if __name__ == "__main__":
    main()