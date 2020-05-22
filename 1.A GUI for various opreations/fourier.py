import cv2
import numpy as np
import matplotlib.pyplot as plt

class fourier_self:
    def fourier_transform(path):
        img = cv2.imread(path,0) #直接读为灰度图像
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        #取绝对值：将复数变化成实数
        #取对数的目的为了将数据变化到0-255
        s1 = np.log(np.abs(fshift))
        plt.subplot(131),plt.imshow(img,'gray'),plt.title('original')
        plt.subplot(132),plt.imshow(s1,'gray'),plt.title('center')
        # 逆变换
        f1shift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f1shift)
        #出来的是复数，无法显示
        img_back = np.abs(img_back)
        plt.subplot(133),plt.imshow(img_back,'gray'),plt.title('img back')
        plt.show()
    
    def Dct_image(path):
        y = cv2.imread(path, 0)
        y1 = y.astype(np.float32)
        Y = cv2.dct(y1)#离散余弦变换
        for i in range(0,240):
            for j in range(0,320):
                if i > 100 or j > 100:
                    Y[i,j] = 0
        y2 = cv2.idct(Y)#反变换
        plt.subplot(131),plt.imshow(y,'gray'),plt.title('origin_img')
        plt.subplot(132),plt.imshow(Y,'gray'),plt.title('Dct_img')
        plt.subplot(133),plt.imshow(y2.astype(np.uint8),'gray'),plt.title('iDCT_img')
        plt.show()

    def radon_transform(self,image):#Radon变换
        '''
        Perform the radon transform on an image, returning the sinogram
        '''
        rows, cols = image.shape
        angles = range(0, 180, 1)
        height = len(angles)
        width = cols
        sinogram = np.zeros((height, width))
        for index, alpha in enumerate(angles):
            M = cv2.getRotationMatrix2D((cols/2, rows/2), alpha, 1)
            rotated = cv2.warpAffine(image, M, (cols, rows))
            sinogram[index] = rotated.sum(axis=0)
        return sinogram
    def image_radon(self,path):#Radon变换
        image = cv2.imread(path, 1)
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        img_canny = cv2.Canny(gray, 100 , 150)
        radon_img=self.radon_transform(img_canny)
        plt.subplot(131),plt.imshow(image,'gray'),plt.title('origin_img')
        plt.subplot(132),plt.imshow(img_canny,'gray'),plt.title('img_canny')
        plt.subplot(133),plt.imshow(radon_img,'gray'),plt.title('radon_img')
        plt.show()