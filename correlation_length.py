import sys
import glob
import cv2
import numpy as np
import scipy.fftpack as fft
from matplotlib import pyplot as plt

if __name__ == '__main__':

    # -- Turn wildcard into array. ex: *.png
    img_list = glob.glob(sys.argv[1])
    print(img_list)

    for i in range(len(img_list)):

        img = plt.imread(img_list[i])[:,:,0]
        # -- Show image
        # plt.imshow()
        # plt.show()

        # -- Do 2D Fourier transformation with zero in the center
        psd = np.abs(fft.fftshift(fft.fft2(fft.ifftshift(img)))) ** 2
        # -- Auto-correlation
        auto_cor = np.real(fft.fftshift(fft.fft2(fft.ifftshift(psd))))
        auto_cor /= np.max(auto_cor)

        print("Show auto-correlation")
        plt.imshow(auto_cor, aspect='auto')
        plt.xticks(fontsize=10, fontweight='bold')
        plt.yticks(fontsize=10, fontweight='bold')
        plt.colorbar()
        plt.show()

        # -- Assume lengths of X,Y are the same
        N = len(img)
        X = np.arange(int(-N/2), int(N/2), 1)
        plt.plot(X, auto_cor[int(N/2),:])
        plt.plot(X, auto_cor[:,int(N/2)])
        plt.show()

