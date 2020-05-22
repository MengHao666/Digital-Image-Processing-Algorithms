import cv2
import numpy as np
from matplotlib import pyplot as plt


# robert 算子[[-1,-1],[1,1]]
def robert_suanzi(img):
    r, c = img.shape
    r_sunnzi = [[-1, -1], [1, 1]]
    for x in range(r):
        for y in range(c):
            if (y + 2 <= c) and (x + 2 <= r):
                imgChild = img[x:x + 2, y:y + 2]
                list_robert = r_sunnzi * imgChild
                img[x, y] = abs(list_robert.sum())  # 求和加绝对值
    return img


# # sobel算子的实现
def sobel_suanzi(img):
    r, c = img.shape
    new_image = np.zeros((r, c))
    new_imageX = np.zeros(img.shape)
    new_imageY = np.zeros(img.shape)
    s_suanziX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # X方向
    s_suanziY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    for i in range(r - 2):
        for j in range(c - 2):
            new_imageX[i + 1, j + 1] = abs(np.sum(img[i:i + 3, j:j + 3] * s_suanziX))
            new_imageY[i + 1, j + 1] = abs(np.sum(img[i:i + 3, j:j + 3] * s_suanziY))
            new_image[i + 1, j + 1] = (new_imageX[i + 1, j + 1] * new_imageX[i + 1, j + 1] + new_imageY[i + 1, j + 1] *
                                       new_imageY[i + 1, j + 1]) ** 0.5
    # return np.uint8(new_imageX)
    # return np.uint8(new_imageY)
    return np.uint8(new_image)  # 无方向算子处理的图像


# Laplace算子
# 常用的Laplace算子模板  [[0,1,0],[1,-4,1],[0,1,0]]   [[1,1,1],[1,-8,1],[1,1,1]]
def Laplace_suanzi(img):
    r, c = img.shape
    new_image = np.zeros((r, c))
    L_sunnzi = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    # L_sunnzi = np.array([[1,1,1],[1,-8,1],[1,1,1]])
    for i in range(r - 2):
        for j in range(c - 2):
            new_image[i + 1, j + 1] = abs(np.sum(img[i:i + 3, j:j + 3] * L_sunnzi))
    return np.uint8(new_image)


def convolve_np(X, F):
    X_height = X.shape[0]
    X_width = X.shape[1]

    F_height = F.shape[0]
    F_width = F.shape[1]

    H = int((F_height - 1) / 2)
    W = int((F_width - 1) / 2)

    out = np.zeros((X_height, X_width))

    for i in np.arange(H, X_height - H):
        for j in np.arange(W, X_width - W):
            sum = 0
            for k in np.arange(-H, H + 1):
                for l in np.arange(-W, W + 1):
                    a = X[i + k, j + l]
                    w = F[H + k, W + l]
                    sum += (w * a)
            out[i, j] = sum
    return out


def Prewitt_suanzi(img):
    # img = cv2.imread('image/2.jpg', 0)
    # print(img.shape)
    # print(img.ndim)

    height = img.shape[0]
    width = img.shape[1]

    Hx = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]])

    Hy = np.array([[-1, -1, -1],
                   [0, 0, 0],
                   [1, 1, 1]])

    img_x = convolve_np(img, Hx) / 6.0
    img_y = convolve_np(img, Hy) / 6.0

    img_out = np.sqrt(np.power(img_x, 2) + np.power(img_y, 2))

    img_out = (img_out / np.max(img_out)) * 255

    return img_out


img = cv2.imread('./image/camera.bmp', 0)
# img = cv2.imread('./image/lena256.bmp', 0)
# img = cv2.imread('./image/1.jpg', 0)
# img = cv2.imread('./image/2.jpg', 0)
cv2.imshow('image', img)

# # robert算子
out_robert = robert_suanzi(img)
cv2.imshow('out_robert_image', out_robert)

# sobel 算子
out_sobel = sobel_suanzi(img)
cv2.imshow('out_sobel_image', out_sobel)

# Laplace算子
out_laplace = Laplace_suanzi(img)
cv2.imshow('out_laplace_image', out_laplace)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Prewitt算子
# img = cv2.imread('image/2.jpg', 0)
out_prewitt = Prewitt_suanzi(img)
plt.imshow(out_prewitt, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([])
plt.show()
