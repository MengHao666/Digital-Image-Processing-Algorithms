# -*- coding: utf-8 -*-
"""
Image Enhancement Algorithms.
Written by HaoMeng
2020/05/18
"""
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import cv2


# 1>>>>>>>>>Exponential transformation>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def ExpPlot(c, v):
    x = np.arange(0, 1, 0.01)
    y = (c + x) ** v
    plt.plot(x, y, 'r', linewidth=1)
    plt.title('FUNCTION PLOT')
    plt.xlim([0, 1]), plt.ylim([0, 1])
    plt.show()


def ExpTran(img, esp=0, gama=1):
    # ExpPlot(esp, gama)
    imarray = np.array(img)
    height, width = imarray.shape

    for i in range(height):
        for j in range(width):
            tmp = imarray[i, j] / 255
            tmp = int(pow(tmp + esp, gama) * 255)
            if tmp >= 0 and tmp <= 255:
                imarray[i, j] = tmp
            elif tmp > 255:
                imarray[i, j] = 255
            else:
                imarray[i, j] = 0
    return imarray


# 2>>>>>>histogram_equalization>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def histogram_equalization(img_arr, level=256, **args):
    ### equalize the distribution of histogram to enhance contrast
    ### @params img_arr : numpy.array uint8 type, 2-dim
    ### @params level : the level of gray scale
    ### @return arr : the equalized image array

    # calculate hists
    hists = calc_histogram_(img_arr, level)

    # equalization
    (m, n) = img_arr.shape
    hists_cdf = calc_histogram_cdf_(hists, m, n, level)  # calculate CDF

    arr = np.zeros_like(img_arr)
    arr = hists_cdf[img_arr]  # mapping

    return arr


def calc_histogram_(gray_arr, level=256):
    ### calculate the histogram of a gray scale image
    ### @params gray_arr : numpy.array uint8 type, 2-dim
    ### @params level : the level of gray scale
    ### @return hists : list type
    hists = [0 for _ in range(level)]
    for row in gray_arr:
        for p in row:
            hists[p] += 1
    return hists


def calc_histogram_cdf_(hists, block_m, block_n, level=256):
    ### calculate the CDF of the hists
    ### @params hists : list type
    ### @params block_m : the histogram block's height
    ### @params block_n : the histogram block's width
    ### @params level : the level of gray scale
    ### @return hists_cdf : numpy.array type
    hists_cumsum = np.cumsum(np.array(hists))
    const_a = (level - 1) / (block_m * block_n)
    hists_cdf = (const_a * hists_cumsum).astype("uint8")
    return hists_cdf


# 3>>>>>>median filtering>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 定义添加椒盐噪声的函数
def saltpepper(src, n):
    # 无论图片
    img = src.copy()
    # print(src.shape)
    # print("src.ndim=", src.ndim)
    m = int((img.shape[0] * img.shape[1]) * n)
    for a in range(m):
        i = int(np.random.random() * img.shape[1])
        j = int(np.random.random() * img.shape[0])

        if img.ndim == 2:
            img[j, i] = 255
        elif img.ndim == 3:
            img[j, i, 0] = 255
            img[j, i, 1] = 255
            img[j, i, 2] = 255
    for b in range(m):
        i = int(np.random.random() * img.shape[1])
        j = int(np.random.random() * img.shape[0])
        if img.ndim == 2:
            img[j, i] = 0
        elif img.ndim == 3:
            img[j, i, 0] = 0
            img[j, i, 1] = 0
            img[j, i, 2] = 0
    return img


def rgbd_median_filter(src, K_size):
    img = src.copy()
    # print(img.ndim)
    if img.ndim == 3:
        H, W, C = img.shape
        # print(H, W, C)
        ## Zero padding
        pad = K_size // 2
        out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float)
        out[pad:pad + H, pad:pad + W] = img.copy().astype(np.float)

        tmp = out.copy()

        for y in range(H):
            for x in range(W):
                for c in range(C):
                    out[pad + y, pad + x, c] = np.median(tmp[y:y + K_size, x:x + K_size, c])

        out = out[pad:pad + H, pad:pad + W].astype(np.uint8)
        return out

    elif img.ndim == 2:
        high, wide = img.shape
        mid = (K_size - 1) // 2
        med_arry = []
        for i in range(high - K_size):
            for j in range(wide - K_size):
                for m1 in range(K_size):
                    for m2 in range(K_size):
                        med_arry.append(int(img[i + m1, j + m2]))

                # for n in range(len(med_arry)-1,-1,-1):
                med_arry.sort()  # 对窗口像素点排序
                # print(med_arry)
                img[i + mid, j + mid] = med_arry[(len(med_arry) + 1) // 2]  # 将滤波窗口的中值赋给滤波窗口中间的像素点
                del med_arry[:]
        return img


# 4>>>>>>Laplace image sharpening>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def Laplace(Image, a):  # 输入图像Image，模板a
    im = array(Image)
    img = array(Image)
    # print(type(a))
    # print(len(a))
    dim = math.sqrt(int(len(a)))  # 模板的维度dim
    w = im.shape[0]  # 计算输入图像的宽高
    h = im.shape[1]
    b = []  # 待处理的与模板等大小的图像块,分BGR通道
    g = []
    r = []
    if sum(a) == 0:  # 判断模板a的和是否为0
        A = 1
    else:
        A = sum(a)
    if size(im.shape) == 3:  # 判断图像为灰度图像还是RGB图像
        for i in range(int(dim / 2), w - int(dim / 2)):
            for j in range(int(dim / 2), h - int(dim / 2)):
                for m in range(-int(dim / 2), -int(dim / 2) + int(dim)):
                    for n in range(-int(dim / 2), -int(dim / 2) + int(dim)):
                        b.append(im[i + m, j + n, 0])
                        g.append(im[i + m, j + n, 1])
                        r.append(im[i + m, j + n, 2])
                img[i, j, 0] = sum(np.multiply(np.array(a), np.array(b))) / A
                img[i, j, 1] = sum(np.multiply(np.array(a), np.array(g))) / A
                img[i, j, 2] = sum(np.multiply(np.array(a), np.array(r))) / A
                b = []
                g = []
                r = []
    else:
        for i in range(int(dim / 2), w - int(dim / 2)):
            for j in range(int(dim / 2), h - int(dim / 2)):
                for m in range(-int(dim / 2), -int(dim / 2) + int(dim)):
                    for n in range(-int(dim / 2), -int(dim / 2) + int(dim)):
                        b.append(im[i + m, j + n])
                img[i, j] = sum(np.multiply(np.array(a), np.array(b))) / A
                b = []
    return img


# # 1.实现图像灰度的指数变换增强《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《
# original image

# img = io.imread(r'./image/peppers.bmp')
# plt.figure()
# plt.subplots_adjust(wspace=0.5, hspace=0.5)
#
# img1 = ExpTran(img, 0, 0.1)
# plt.subplot(521,)
# plt.imshow(img1, cmap="gray")
# plt.subplot(522)
# plt.hist(img1.flatten(), bins=256)
# plt.xlabel('(x+0)**0.1')
#
# img2 = ExpTran(img, 0, 0.5)
# plt.subplot(523)
# plt.imshow(img2, cmap="gray")
# plt.subplot(524)
# plt.hist(img2.flatten(), bins=256)
# plt.xlabel('(x+0)**0.5')
#
# plt.subplot(525)
# plt.imshow(img, cmap="gray")
# plt.subplot(526)
# plt.hist(img.flatten(), bins=256)
# plt.xlabel('original image')
#
# img3 = ExpTran(img, 0, 1.5)
# plt.subplot(527)
# plt.imshow(img3, cmap="gray")
# plt.subplot(528)
# plt.hist(img3.flatten(), bins=256)
# plt.xlabel('(x+0)**1.5')
#
# img4 = ExpTran(img, 0, 3)
# plt.subplot(529)
# plt.imshow(img4, cmap="gray")
# plt.subplot(5, 2, 10)
# plt.hist(img4.flatten(), bins=256)
# plt.xlabel('(x+0)**3')
# plt.show()

# 2.完成图像的直方图均衡化处理《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《
# original image
# img = io.imread(r'./image/car.jpg')
# plt.figure()
# plt.subplot(221)
# plt.imshow(img, cmap="gray")
# plt.subplot(222)
# plt.hist(img.flatten(), bins=256)
# plt.xlabel('Original Image')
# # HE
# he_eq_img = histogram_equalization(img)
# plt.subplot(223)
# plt.imshow(he_eq_img, cmap="gray")
# plt.subplot(224)
# plt.hist(he_eq_img.flatten(), bins=256)
# plt.xlabel('After Histogram Equalization')
# plt.show()

# 3.实现图像的中值滤波平滑处理----二维中值滤波《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《
# original_img = io.imread(r'./image/rgb.jpg')
# original_img = io.imread(r'./image/1.jpg')
# # original_img = io.imread(r'./image/baboo256.bmp')
# plt.figure()
# plt.subplot(221)
# # plt.imshow(original_img)
# plt.imshow(original_img, cmap='gray')
# plt.title('Original Image')
#
# pepper_img = saltpepper(original_img, 0.1)
# # io.imsave('./image/rgb_pepper.jpg', pepper_img)
# plt.subplot(222)
# plt.imshow(pepper_img, cmap='gray')
# plt.title('Pepper Image')
#
# median_filtering_img1 = rgbd_median_filter(pepper_img, 3)
# plt.subplot(223)
# plt.imshow(median_filtering_img1, cmap='gray')
# plt.title('median filter Image (size=3)')
#
# median_filtering_img2 = rgbd_median_filter(pepper_img, 5)
# plt.subplot(224)
# plt.imshow(median_filtering_img2, cmap='gray')
# plt.title('median filter Image (size=5)')
#


# plt.show()

# 4.实现图像的Laplace锐化处理《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《《
img = cv2.imread('./image/rgb.jpg')
# img = cv2.imread('./image/lena256.bmp')
a = [0, -1, 0, -1, 5, -1, 0, -1, 0]
im = Laplace(img, a)
cv2.imshow('Origin', img)
cv2.imshow('Laplace', im)
cv2.waitKey()
cv2.destroyAllWindows()
