import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np 
class image_enhance_self:
    #伪彩色增强
    def Color(self,path):
        im_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_JET)
        plt.subplot(121),plt.imshow(im_gray,'gray'),plt.title('origin_img')
        plt.subplot(122),plt.imshow(im_color),plt.title('icolor_img')
        plt.show()
    # 彩色图像均衡化
    def colorhistogram(path):
        img = cv2.imread(path, 1)
        # 彩色图像均衡化,需要分解通道 对每一个通道均衡化
        (b, g, r) = cv2.split(img)
        bH = cv2.equalizeHist(b)
        gH = cv2.equalizeHist(g)
        rH = cv2.equalizeHist(r)
        srcimg=cv2.merge([r,g,b])
        # 合并每一个通道
        result = cv2.merge(( rH,gH,bH))
        plt.subplot(121),plt.imshow(srcimg),plt.title('origin_img')
        plt.subplot(122),plt.imshow(result),plt.title('equalizeHist_img')
        plt.show()
    #YCrCb颜色模型    
    def ycrcbimage(path):
        img = cv2.imread(path, 1)
        YCrcbimage = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        (b, g, r) = cv2.split(img)
        srcimg=cv2.merge([r,g,b])
        plt.subplot(121),plt.imshow(srcimg),plt.title('origin_img')
        plt.subplot(122),plt.imshow(YCrcbimage),plt.title('ycrcb_img')
        plt.show()
    #HSV颜色模型    
    def hsvimage(path):
        img = cv2.imread(path, 1)
        Hsvimage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        #Hsvimage=cv2.applyColorMap(img, cv2.COLORMAP_NTSC)
        (b, g, r) = cv2.split(img)
        srcimg=cv2.merge([r,g,b])
        plt.subplot(121),plt.imshow(srcimg),plt.title('origin_img')
        plt.subplot(122),plt.imshow(Hsvimage),plt.title('hsv_img')
        plt.show()
    


    
