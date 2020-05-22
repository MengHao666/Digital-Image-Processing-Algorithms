import cv2
import numpy as np
from numpy.fft import *
import matplotlib.pyplot as plt


def DFT(sig):
    N = sig.size
    V = np.array([[np.exp(-1j * 2 * np.pi * v * y / N) for v in range(N)] for y in range(N)])
    return sig.dot(V)


def FFT(x):
    N = x.shape[1]  # 只需考虑第二个维度，然后在第一个维度循环
    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 8:  # this cutoff should be optimized
        return np.array([DFT(x[i, :]) for i in range(x.shape[0])])
    else:
        X_even = FFT(x[:, ::2])
        X_odd = FFT(x[:, 1::2])
        factor = np.array([np.exp(-2j * np.pi * np.arange(N) / N) for i in range(x.shape[0])])
        return np.hstack([X_even + np.multiply(factor[:, :int(N / 2)], X_odd),
                          X_even + np.multiply(factor[:, int(N / 2):], X_odd)])


def FFT2D(img):
    return FFT(FFT(img).T).T


def FFT_SHIFT(img):
    M, N = img.shape
    M = int(M / 2)
    N = int(N / 2)
    return np.vstack((np.hstack((img[M:, N:], img[M:, :N])), np.hstack((img[:M, N:], img[:M, :N]))))


def paint_my_square(img, size=4, bias=None):
    if bias is None:
        bias = [0, 0]
    img2 = np.zeros(img.shape, img.dtype)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i][j] = 0
    d = int(img.shape[0] / 2 - size / 2)
    for i in range(size):
        for j in range(size):
            img[d + i + bias[0]][d + j + bias[1]] = 255
            # print("img[{}][{}] = 255".format(d + i, d + j))

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img2[i][j] = img[i][j]

    return img2


# img = cv2.imread("Lenna.png", cv2.IMREAD_COLOR)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


gray = np.zeros((128, 128), "uint8")

# gray = paint_my_square(gray)
# gray = paint_my_square(gray, bias=[-40, -40])
# gray = paint_my_square(gray,32)
gray = paint_my_square(gray, 2)

plt.imshow(gray, cmap='gray')
plt.show()

# gray = cv2.imread("./image/lena256.bmp", cv2.IMREAD_GRAYSCALE)

my_fft = abs(FFT_SHIFT(FFT2D(gray)))
target = abs(fftshift(fft2(gray)))
print('distance from numpy.fft: ', np.linalg.norm(my_fft - target))

print(type(my_fft))
print(my_fft.shape)
# print("my_fft=", my_fft)

plt.subplot(2, 2, 1)
plt.title('original')
plt.imshow(gray, cmap='gray')
plt.subplot(2, 2, 2)
plt.title('my FFT2D')
my_fft = np.log(1 + my_fft)
# print("my_fft", my_fft)
print("my_fft", my_fft.dtype)

plt.imshow(my_fft, cmap='gray')
plt.subplot(2, 2, 3)
plt.title('numpy.fft2')
plt.imshow(np.log(1 + target), cmap='gray')
plt.show()
