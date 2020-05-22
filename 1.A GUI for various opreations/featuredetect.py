# -*- coding=utf-8 -*-
# Summary: 使用OpenCV3.x-Python检测AKAZE特征点
# Author:  Amusi
# Reference: https://docs.opencv.org/master/d8/d30/classcv_1_1AKAZE.html
 
import cv2
import numpy
from matplotlib import pyplot as plt 
class feature_self:
    def feature_detection(path):
        img = cv2.imread(path)
        plt.subplot(231),plt.imshow(img),plt.title('origin_img')    
        # AKAZE检测
        akaze = cv2.AKAZE_create()
        keypoints = akaze.detect(img, None)
        # 显示
        # 必须要先初始化img2
        img2 = img.copy()
        akaze_img = cv2.drawKeypoints(img, keypoints, img2, color=(0,255,0))
        plt.subplot(232),plt.imshow(akaze_img),plt.title('akaze_img')
        #brisk检测
        brisk = cv2.BRISK_create()
        keypoints = brisk.detect(img, None)
        
        # 必须要先初始化img2
        img3 = img.copy()
        brisk_img = cv2.drawKeypoints(img, keypoints, img3, color=(0,255,0))
        plt.subplot(233),plt.imshow(brisk_img),plt.title('brisk_img')
        #fast检测
        fast = cv2.FastFeatureDetector_create()
        keypoints = fast.detect(img, None)
        
        # 必须要先初始化img2
        img4 = img.copy()
        fast_img = cv2.drawKeypoints(img, keypoints, img4, color=(0,255,0))
        plt.subplot(234),plt.imshow(fast_img),plt.title('fast_img')

        # kaze检测
        kaze = cv2.KAZE_create()
        keypoints = kaze.detect(img, None)
        
        # 显示
        # 必须要先初始化img2
        img5 = img.copy()
        kaze_img = cv2.drawKeypoints(img, keypoints, img5, color=(0,255,0))
        plt.subplot(235),plt.imshow(kaze_img),plt.title('kaze_img')

        # orb检测
        orb = cv2.ORB_create()
        keypoints = orb.detect(img, None)
        
        # 显示
        # 必须要先初始化img2
        img6 = img.copy()
        orb_img = cv2.drawKeypoints(img, keypoints, img6, color=(0,255,0))
        plt.subplot(236),plt.imshow(orb_img),plt.title('orb_img')

        plt.show()





