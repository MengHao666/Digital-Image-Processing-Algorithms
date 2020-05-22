# -*- coding: utf-8 -*-
 
from numpy import *
from scipy import * 
import numpy as np 
import cv2,skimage
import matplotlib.pyplot as plt

class noise_self:
    #定义添加高斯噪声的函数 
    def addGaussianNoise(path,percetage): 
        srcImage = cv2.imread(path)
        b,g,r=cv2.split(srcImage)#先将bgr格式拆分
        srcimg=cv2.merge([r,g,b])
        gaussian_noise_img = skimage.util.random_noise(srcimg, mode='gaussian', seed=None, clip=True)
        plt.subplot(121),plt.imshow(srcimg),plt.title('origin_img')
        plt.subplot(122),plt.imshow(gaussian_noise_img),plt.title('G_Noiseimg')
        plt.show()

    #定义添加椒盐噪声的函数
    def saltpepper(path,n):
        image=cv2.imread(path)
        b,g,r=cv2.split(image)#先将bgr格式拆分
        img=cv2.merge([r,g,b])
        srcimg=cv2.merge([r,g,b])
        m=int((img.shape[0]*img.shape[1])*n)
        for a in range(m):
            i=int(np.random.random()*img.shape[1])
            j=int(np.random.random()*img.shape[0])
            if img.ndim==2:
                img[j,i]=255
            elif img.ndim==3:
                img[j,i,0]=255
                img[j,i,1]=255
                img[j,i,2]=255
        for b in range(m):
            i=int(np.random.random()*img.shape[1])
            j=int(np.random.random()*img.shape[0])
            if img.ndim==2:
                img[j,i]=0
            elif img.ndim==3:
                img[j,i,0]=0
                img[j,i,1]=0
                img[j,i,2]=0
        plt.subplot(121),plt.imshow(srcimg),plt.title('origin_img')
        plt.subplot(122),plt.imshow(img),plt.title('saltpepper_img')
        plt.show()
    #斑点噪声
    def speckle_img(path):
        image=cv2.imread(path)
        b,g,r=cv2.split(image)#先将bgr格式拆分
        img=cv2.merge([r,g,b])
        speckle_noise_img = skimage.util.random_noise(img, mode='speckle', seed=None, clip=True)
        plt.subplot(121),plt.imshow(img),plt.title('origin_img')
        plt.subplot(122),plt.imshow(speckle_noise_img),plt.title('speckle_img')
        plt.show()
    #泊松噪声
    def poisson_img(path):
        image=cv2.imread(path)
        b,g,r=cv2.split(image)#先将bgr格式拆分
        img=cv2.merge([r,g,b])
        poisson_noise_img = skimage.util.random_noise(img, mode='poisson', seed=None, clip=True)
        plt.subplot(121),plt.imshow(img),plt.title('origin_img')
        plt.subplot(122),plt.imshow(poisson_noise_img),plt.title('poisson_img')
        plt.show()