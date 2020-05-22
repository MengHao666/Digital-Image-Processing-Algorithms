# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt
 
class threshold_self:
    #阈值分割
    def threshold_image(path):
        img=cv2.imread(path,0) #0是第二个参数，将其转为灰度图
        
        #利用cv2.threshhold()函数进行简单阈值分割，第一个参数是待分割图像，第二个参数是阈值大小
        #第三个参数是赋值的像素值，第四个参数是阈值分割方法
        ret,thresh1=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        ret,thresh2=cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
        ret,thresh3=cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
        ret,thresh4=cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
        ret,thresh5=cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
        
        titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
        images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
        for i in range(6):
            plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
            plt.title(titles[i])
            plt.xticks(),plt.yticks([]) #显示坐标轴，如为空，则无坐标轴
        plt.show()
 
