import cv2
import numpy as np
from matplotlib import pyplot as plt

# read and display BMP image
# img = cv2.imread("model.bmp")
# cv2.namedWindow("Image", 0)
# cv2.imshow("Image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# read and display JPG image
# jpgImg = cv2.imread("1.jpg")
# cv2.namedWindow("jpgImg", 0)
# cv2.imshow("jpgImg", jpgImg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# # jpg2bmp
# cv2.imwrite('jpg2bmp.bmp', jpgImg)


# read and display RAW image
# rawImg = cv2.imread("2.3FR")
# cv2.namedWindow("rawImg", 0)
# cv2.imshow("rawImg", rawImg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# # raw2bmp
# cv2.imwrite('raw2bmp.bmp', rawImg)

# image addition
# img1 = cv2.imread("./imagetest/model.bmp")
# img2 = cv2.imread("./imagetest/blur.bmp")
# added = cv2.add(img1, img2)
# cv2.imwrite('added.bmp', added)
# cv2.namedWindow("addedImg", 0)
# cv2.imshow("addedImg", added)
# cv2.waitKey(0)

# Image inversion
# img3 = cv2.imread("./imagetest/model.bmp")
# dst = cv2.bitwise_not(img3)
# cv2.namedWindow("Image inversion", 0)
# cv2.imshow("Image inversion", dst)
# cv2.waitKey()
# cv2.imwrite('inversion.bmp', dst)

# 图像翻转
# image = cv2.imread("1.jpg")
# # Flipped Horizontally 水平翻转
# h_flip = cv2.flip(image, 1)
# cv2.namedWindow("Image inversion", 0)
# cv2.imshow("Image inversion", h_flip)
# cv2.waitKey()
# # Flipped Vertically 垂直翻转
# v_flip = cv2.flip(image, 0)
# # Flipped Horizontally & Vertically 水平垂直翻转
# hv_flip = cv2.flip(image, -1)

# 图像缩放
# img = cv2.imread('1.jpg')
# res = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
# cv2.namedWindow("Image inversion", 1)
# cv2.imshow("Image inversion", res)
# cv2.waitKey()

# 图像平移
img = cv2.imread('./imagetest/1.jpg', 0)
rows, cols = img.shape

M = np.float32([[1, 0, 100], [0, 1, 50]])
dst = cv2.warpAffine(img, M, (cols, rows))

cv2.namedWindow("img", 0)
cv2.namedWindow("dst", 0)
cv2.imshow('img', img)
cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 图像旋转
# img = cv2.imread('1.jpg', 0)
# rows, cols = img.shape
#
# # cols-1 and rows-1 are the coordinate limits.
# M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 90, 1)
# dst = cv2.warpAffine(img, M, (cols, rows))
#
# cv2.namedWindow("img", 0)
# cv2.namedWindow("dst", 0)
# cv2.imshow('img', img)
# cv2.imshow('dst', dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 灰度图像直方图均衡化
# img = cv2.imread("imagetest/1.jpg", 1)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print(gray.shape)
# cv2.namedWindow("src", 0)
# cv2.namedWindow("dst", 0)
# cv2.imshow("src", gray)
#
# dst = cv2.equalizeHist(gray)
# cv2.imshow("dst", dst)
#
# cv2.waitKey(0)


# 彩色图像直方图均衡化
# img = cv2.imread("1.jpg", 1)
# print(img.shape)
# cv2.namedWindow("src", 0)
# cv2.namedWindow("dst", 0)
#
# cv2.imshow("src", img)
#
# # 彩色图像均衡化,需要分解通道 对每一个通道均衡化
# (b, g, r) = cv2.split(img)
# bH = cv2.equalizeHist(b)
# gH = cv2.equalizeHist(g)
# rH = cv2.equalizeHist(r)
# # 合并每一个通道
# result = cv2.merge((bH, gH, rH))
# cv2.imshow("dst", result)
#
# cv2.waitKey(0)
