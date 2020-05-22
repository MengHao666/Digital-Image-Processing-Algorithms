# -- coding: utf-8 --
import cv2,math
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from PIL import ImageFilter,Image

class filter_self:
    #高通滤波
    def high_pass_filter(path):
        #这个是滤波器使用的模板矩阵
        kernel_3x3 = np.array([[-1, -1, -1],
                            [-1, 8, -1],
                            [-1, -1, -1]])

        kernel_5x5 = np.array([[-1, -1, -1, -1, -1],
                            [-1, 1, 2, 1, -1],
                            [-1, 2, 4, 2, -1],
                            [-1, 1, 2, 1, -1],
                            [-1, -1, -1, -1, -1]])
        #显示原始图像
        srcImage = cv2.imread(path)
        b,g,r=cv2.split(srcImage)#先将bgr格式拆分
        srcimg=cv2.merge([r,g,b])
        plt.subplot(231),plt.imshow(srcimg),plt.title('origin_img')

        #以灰度的方式加载图片
        img = cv2.imread(path, 0)
        plt.subplot(232),plt.imshow(img),plt.title('gray_img')

        #通过使用模板矩阵进行高通滤波
        k3 = ndimage.convolve(img, kernel_3x3)
        k5 = ndimage.convolve(img, kernel_5x5)

        #使用OpenCV的高通滤波
        blurred = cv2.GaussianBlur(img, (11, 11), 0)
        g_hpf = img - blurred

        plt.subplot(233),plt.imshow(k3),plt.title('3x3')
        plt.subplot(234),plt.imshow(k5),plt.title('5x5')
        plt.subplot(235),plt.imshow(blurred),plt.title('blurred')
        plt.subplot(236),plt.imshow(g_hpf),plt.title('g_hpf')
        plt.show()
    #低通滤波
    def low_pass_filter(path):
        #显示原始图像
        srcImage = cv2.imread(path)
        b,g,r=cv2.split(srcImage)#先将bgr格式拆分
        srcimg=cv2.merge([r,g,b])
        plt.subplot(131),plt.imshow(srcimg),plt.title('origin_img')
        img = cv2.imread(path, 0) 
        result = cv2.blur(img,(5,5))
        plt.subplot(132),plt.imshow(result),plt.title('blur_img')
        l_hpf = img - result
        plt.subplot(133),plt.imshow(l_hpf),plt.title('l_hpf_img')
        plt.show()

    #线性平滑滤波（盒式滤波、均值滤波、高斯滤波）
    def Linear_Smooth_Filter(path):
         #显示原始图像
        srcImage = cv2.imread(path)
        b,g,r=cv2.split(srcImage)#先将bgr格式拆分
        srcimg=cv2.merge([r,g,b])
        plt.subplot(221),plt.imshow(srcimg),plt.title('origin_img')
        #img = cv2.imread(path)
        # 均值滤波
        img_mean = cv2.blur(srcimg,(5,5))
        plt.subplot(222),plt.imshow(img_mean),plt.title('mean_img')
        #盒式滤波
        box_img=cv2.boxFilter(srcimg,-1,(5, 5))
        plt.subplot(223),plt.imshow(box_img),plt.title('box_img')
        # 高斯滤波
        img_Guassian = cv2.GaussianBlur(srcimg,(5,5),0)
        plt.subplot(224),plt.imshow(img_Guassian),plt.title('Guassian_img')
        plt.show()
    #导向滤波
    def guideFilter(I, p, winSize, eps, s):
        #输入图像的高、宽
        h, w = I.shape[:2] 
        #缩小图像
        size = (int(round(w*s)), int(round(h*s)))
        small_I = cv2.resize(I, size, interpolation=cv2.INTER_CUBIC)
        small_p = cv2.resize(I, size, interpolation=cv2.INTER_CUBIC)
        #缩小滑动窗口
        X = winSize[0]
        small_winSize = (int(round(X*s)), int(round(X*s)))
        #I的均值平滑
        mean_small_I = cv2.blur(small_I, small_winSize)
        #p的均值平滑
        mean_small_p = cv2.blur(small_p, small_winSize)
        #I*I和I*p的均值平滑
        mean_small_II = cv2.blur(small_I*small_I, small_winSize)
        mean_small_Ip = cv2.blur(small_I*small_p, small_winSize)
        #方差
        var_small_I = mean_small_II - mean_small_I * mean_small_I #方差公式
        #协方差
        cov_small_Ip = mean_small_Ip - mean_small_I * mean_small_p
        small_a = cov_small_Ip / (var_small_I + eps)
        small_b = mean_small_p - small_a*mean_small_I
        #对a、b进行均值平滑
        mean_small_a = cv2.blur(small_a, small_winSize)
        mean_small_b = cv2.blur(small_b, small_winSize)
        #放大
        size1 = (w, h)
        mean_a = cv2.resize(mean_small_a, size1, interpolation=cv2.INTER_LINEAR)
        mean_b = cv2.resize(mean_small_b, size1, interpolation=cv2.INTER_LINEAR)
        q = mean_a*I + mean_b
        return q
    #非线性平滑滤波（中值滤波、双边滤波、导向滤波）
    def NonLinear_Smooth_Filter(self,path):
         #显示原始图像
        srcImage = cv2.imread(path,cv2.IMREAD_ANYCOLOR)
        b,g,r=cv2.split(srcImage)#先将bgr格式拆分
        srcimg=cv2.merge([r,g,b])
        plt.subplot(131),plt.imshow(srcimg),plt.title('origin_img')
        # 中值滤波
        img_median = cv2.medianBlur(srcimg, 5)
        plt.subplot(132),plt.imshow(img_median),plt.title('median_img')
        # 双边滤波
        img_bilater = cv2.bilateralFilter(srcimg,9,75,75)
        plt.subplot(133),plt.imshow(img_bilater),plt.title('bilater_img')
        # 导向滤波
        #image_0_1 = srcImage/255.0#将图像归一化
        #b, g, r = cv2.split(image_0_1)
        #gf1 = self.guidefilter(image_0_1, b, (16,16), math.pow(0.1,2))
        #gf2 = self.guideFilter(g, g, (16,16), math.pow(0.1,2))
        #gf3 = self.guideFilter(r, r, (16,16), math.pow(0.1,2))
        #gf = cv2.merge([gf1, gf2, gf3])
        #plt.subplot(224),plt.imshow(gf1),plt.title('guide_img')
        plt.show()

    #锐化滤波
    def Sharpen_Filter(self,path):
        #显示原始图像
        srcImage = cv2.imread(path,cv2.IMREAD_ANYCOLOR)
        b,g,r=cv2.split(srcImage)#先将bgr格式拆分
        srcimg=cv2.merge([r,g,b])
        plt.subplot(121),plt.imshow(srcimg),plt.title('origin_img')
        # 锐化滤波
        newimage=Image.open(path)
        imsharpen =newimage.filter(ImageFilter.SHARPEN)
        plt.subplot(122),plt.imshow(imsharpen),plt.title('sharpen_img')
        plt.show()

