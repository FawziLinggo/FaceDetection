import logging
import numpy as np


# Homomorphic filter class
class HomomorphicFilter:
    """Homomorphic filter implemented with diferents filters and an option to an external filter.

    High-frequency filters implemented:
        butterworth
        gaussian
    Attributes:
        a, b: Floats used on emphasis filter:
            H = a + b*H

        .
    """

    def __init__(self, a=0.5, b=1.5):
        self.a = float(a)
        self.b = float(b)

    # Filters
    def __butterworth_filter(self, I_shape, filter_params):
        P = I_shape[0] / 2
        Q = I_shape[1] / 2
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U - P) ** 2 + (V - Q) ** 2)).astype(float)
        H = 1 / (1 + (Duv / filter_params[0] ** 2) ** filter_params[1])
        return (1 - H)

    def __gaussian_filter(self, I_shape, filter_params):
        P = I_shape[0] / 2
        Q = I_shape[1] / 2
        H = np.zeros(I_shape)
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U - P) ** 2 + (V - Q) ** 2)).astype(float)
        H = np.exp((-Duv / (2 * (filter_params[0]) ** 2)))
        return (1 - H)

    # Methods
    def __apply_filter(self, I, H):
        H = np.fft.fftshift(H)
        I_filtered = (self.a + self.b * H) * I
        return I_filtered

    def filter(self, I, filter_params, filter='butterworth', H=None):
        """
        Method to apply homormophic filter on an image
        Attributes:
            I: Single channel image
            filter_params: Parameters to be used on filters:
                butterworth:
                    filter_params[0]: Cutoff frequency
                    filter_params[1]: Order of filter
                gaussian:
                    filter_params[0]: Cutoff frequency
            filter: Choose of the filter, options:
                butterworth
                gaussian
                external
            H: Used to pass external filter
        """

        #  Validating image
        if len(I.shape) != 2:
            raise Exception('Improper image')

        # Take the image to log domain and then to frequency domain
        I_log = np.log1p(np.array(I, dtype="float"))
        I_fft = np.fft.fft2(I_log)

        # Filters
        if filter == 'butterworth':
            H = self.__butterworth_filter(I_shape=I_fft.shape, filter_params=filter_params)
        elif filter == 'gaussian':
            H = self.__gaussian_filter(I_shape=I_fft.shape, filter_params=filter_params)
        elif filter == 'external':
            print('external')
            if len(H.shape) != 2:
                raise Exception('Invalid external filter')
        else:
            raise Exception('Selected filter not implemented')

        # Apply filter on frequency domain then take the image back to spatial domain
        I_fft_filt = self.__apply_filter(I=I_fft, H=H)
        I_filt = np.fft.ifft2(I_fft_filt)
        I = np.exp(np.real(I_filt)) - 1
        return np.uint8(I)


# End of class HomomorphicFilter

if __name__ == "__main__":
    import cv2

    # Code parameters
    path_in = '/home/fawzi/PycharmProjects/pythonProject/images/in/'
    path_out = '/home/fawzi/PycharmProjects/pythonProject/images/out/'
    img_path = '1.JPG'

    # Derived code parameters
    img_path_in = path_in + img_path
    #img_path_out = path_out + 'filtered_60.png'

    # Main code
    img = cv2.imread(img_path_in)[:, :, 0]
    homo_filter = HomomorphicFilter(a=0.75, b=1.25)
    img_filtered = homo_filter.filter(I=img, filter_params=[50, 4])
    # cv2.imwrite(img_path_out, img_filtered)

    cv2.namedWindow('Gambar Filtering', cv2.WINDOW_KEEPRATIO)
    cv2.imwrite("images/HasilFiltering/Filtering.jpg", img)
    cv2.imshow("Gambar Filtering", img)
    cv2.waitKey(0)

    # # erosion
    # for i in range(0, 3):
    #     eroded = cv2.erode(img_filtered.copy(), None)
    #     cv2.namedWindow(f"Erosi {i+1} kali", cv2.WINDOW_KEEPRATIO)
    #     cv2.imshow(f"Erosi {i+1} kali", eroded)
    #     cv2.waitKey(0)
    # # dilation
    # for i in range(0, 3):
    #     dilated = cv2.dilate(eroded, None, iterations = i + 1)
    #     cv2.namedWindow(f"Dilasi {i+1} kali", cv2.WINDOW_KEEPRATIO)
    #     cv2.imshow(f"Dilasi {i+1} kali", dilated)
    #     cv2.waitKey(0)
    #
    # kernelSizes = [(3, 3), (5, 5), (7, 7)]
    # # Opening
    # for kernelSize in kernelSizes:
    #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
    #     opening = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, kernel)
    #     cv2.namedWindow(f"Opening : ({kernelSize[0]}, {kernelSize[1]}", cv2.WINDOW_KEEPRATIO)
    #     cv2.imshow(f"Opening : ({kernelSize[0]}, {kernelSize[1]}", opening)
    #     cv2.waitKey(0)