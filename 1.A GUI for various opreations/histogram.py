# -- coding: utf-8 --
from PIL import Image, ImageFilter, ImageDraw
import random, sys
import numpy as np
import matplotlib.pyplot as plt


class histogram_self:
    # R直方图
    def R_histogram(path):
        im1 = Image.open(path)
        # 将RGB三个通道分开
        r, g, b = im1.split()

        plt.subplot(121), plt.imshow(im1), plt.title('origin_img')
        ar = np.array(r).flatten()
        # plt.subplot(122),plt.hist(ar,bins = 256, normed = 1, facecolor = 'red', edgecolor = 'red'),plt.title('red_histogram_img')
        plt.subplot(122), plt.hist(ar, bins=256, density=1, facecolor='red', edgecolor='red'), plt.title(
            'red_histogram_img')
        plt.show()

    # G直方图
    def G_histogram(path):
        im1 = Image.open(path)
        # 将RGB三个通道分开
        r, g, b = im1.split()

        plt.subplot(121), plt.imshow(im1), plt.title('origin_img')
        ag = np.array(g).flatten()
        plt.subplot(122), plt.hist(ag, bins=256, density=1, facecolor='green', edgecolor='green'), plt.title(
            'green_histogram_img')
        plt.show()

    # B直方图
    def B_histogram(path):
        im1 = Image.open(path)
        # 将RGB三个通道分开
        r, g, b = im1.split()

        plt.subplot(121), plt.imshow(im1), plt.title('origin_img')
        ab = np.array(b).flatten()
        plt.subplot(122), plt.hist(ab, bins=256, density=1, facecolor='blue', edgecolor='blue'), plt.title(
            'blue_histogram_img')
        plt.show()
