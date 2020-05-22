#coding=utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt
 
class morphology_self:
    def morphology(path):
        img = cv2.imread(path,0)
        #OpenCV定义的结构元素
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
        
        #腐蚀图像
        eroded = cv2.erode(img,kernel)
        #显示腐蚀后的图像
        #cv2.imshow("Eroded Image",eroded);
        plt.subplot(221), plt.imshow(eroded, 'gray'),plt.title('Eroded Image')

        #膨胀图像
        dilated = cv2.dilate(img,kernel)
        #显示膨胀后的图像
        #cv2.imshow("Dilated Image",dilated);
        plt.subplot(222), plt.imshow(dilated, 'gray'),plt.title('Dilated Image')
        
        #闭运算
        closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        #显示腐蚀后的图像
        #cv2.imshow("Close",closed);
        plt.subplot(224), plt.imshow(closed, 'gray'),plt.title('Close Image')

        #开运算
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        #显示腐蚀后的图像
        #cv2.imshow("Open", opened);
        plt.subplot(223), plt.imshow(opened, 'gray'),plt.title('Open Image')

        plt.show()
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()