from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

class button_self: 
    def add_num(self,img):
        draw = ImageDraw.Draw(img)
        myfont = ImageFont.truetype('simhei.ttf', size=40)
        fillcolor = "#ff0000"
        width,height = img.size
        draw.text((width-400,8),'数字图像处理演示', font=myfont, fill=fillcolor)
        # img.save('result.jpg','jpeg')
        return img
    
    def scan_pic(self,path):#图片浏览
        im_before = Image.open(path)
        plt.imshow(im_before)
        plt.title('scan_img')
        plt.show()

    def mark_pic(self,path):#添加水印
        im_before = Image.open(path)
        plt.subplot(121),plt.imshow(im_before),plt.title('origin_img')
        im_after = self.add_num(im_before)
        plt.subplot(122),plt.imshow(im_after),plt.title('mark_img')
        plt.show()

    def zoomout_pic(self,path):#图片放大
        im=Image.open(path)
        (x, y) = im.size
        newimg=im.resize((x+200,y+200), Image.ANTIALIAS)
        plt.subplot(121),plt.imshow(im),plt.title('origin_img')
        plt.subplot(122),plt.imshow(newimg),plt.title('resize_img')
        plt.show()

    def cw_pic(self,path):#顺时针旋转90°
        img=Image.open(path)
        region = img.transpose(Image.ROTATE_270) #翻转
        plt.subplot(121),plt.imshow(img),plt.title('origin_img')
        plt.subplot(122),plt.imshow(region),plt.title('cw90_img')
        plt.show()

    def rcw_pic(self,path):#逆时针旋转90°
        img=Image.open(path)
        region = img.transpose(Image.ROTATE_90) #翻转
        plt.subplot(121),plt.imshow(img),plt.title('origin_img')
        plt.subplot(122),plt.imshow(region),plt.title('ccw90_img')
        plt.show()
    
    def file_save(self,path,savepath):#图片保存
        img = Image.open(path)
        img.save(savepath)

    def pre_file(self,file_path):#上一张
        file_dir=os.path.dirname(file_path)
        L=[]
        for root, dirs, files in os.walk(file_dir):
            for file in files:
                L.append(file)
        index=L.index(file_path.split("/")[-1])#查找本文件的索引
        if index==0:
            index=len(L)-1
        else:
            index=index-1
        pre_file_path=file_dir+'/'+L[index]
        return pre_file_path

    def next_file(self,file_path):#下一张
        file_dir=os.path.dirname(file_path)
        L=[]
        for root, dirs, files in os.walk(file_dir):
            for file in files:
                L.append(file)
        index=L.index(file_path.split("/")[-1])#查找本文件的索引
        if index==len(L)-1:
            index=0
        else:
            index=index+1
        pre_file_path=file_dir+'/'+L[index]
        return pre_file_path


    
        
    