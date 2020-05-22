from PyQt5 import QtWidgets
from imageprocess import Ui_MainWindow
# from PyQt5.QtWidgets import QFileDialog
from imageedit import imageedit_self
from fourier import fourier_self
from addnoise import noise_self
from allfilter import filter_self
from histogram import histogram_self
from imageenhance import image_enhance_self
from threshold import threshold_self
from morphology import morphology_self
from featuredetect import feature_self
from bottombutton import button_self
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class mywindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(mywindow, self).__init__()
        self.setupUi(self)
        self.open.triggered.connect(self.read_file)  # 打开
        self.save.triggered.connect(self.save_file)  # 保存
        # 编辑
        self.zoomin.triggered.connect(self.zoomin_file)  # 放大
        self.zoomout.triggered.connect(self.zoomout_file)  # 缩小
        self.gray.triggered.connect(self.gray_file)  # 灰度
        self.light.triggered.connect(self.light_file)  # 亮度
        self.rotate.triggered.connect(self.rotate_file)  # 旋转
        self.screenshots.triggered.connect(self.screenshots_file)  # 截图
        # 变换
        self.FFT.triggered.connect(self.fft_file)  # 傅里叶变换
        self.cos.triggered.connect(self.cos_file)  # 离散余弦变换
        self.Radon.triggered.connect(self.radon_file)  # Radon变换
        # 噪声
        self.gauss.triggered.connect(self.gauss_file)  # 高斯噪声
        self.sault.triggered.connect(self.sault_file)  # 椒盐噪声
        self.spot.triggered.connect(self.spot_file)  # 斑点噪声
        self.poisson.triggered.connect(self.poisson_file)  # 泊松噪声
        # 滤波
        self.highpass.triggered.connect(self.highpass_file)  # 高通滤波
        self.lowpass.triggered.connect(self.lowpass_file)  # 低通滤波
        self.linearsmooth.triggered.connect(self.linearsmooth_file)  # 平滑滤波（线性）
        self.nonlinear.triggered.connect(self.nonlinear_file)  # 平滑滤波（非线性）
        self.linearsharpen.triggered.connect(self.linearsharpen_file)  # 锐化滤波（线性）
        self.nonlinearsharp.triggered.connect(self.nonlinearsharp_file)  # 锐化滤波（非线性）
        # 直方图统计
        self.Rhistogram.triggered.connect(self.Rhistogram_file)  # R直方图
        self.Ghistogram.triggered.connect(self.Ghistogram_file)  # G直方图
        self.Bhistogram.triggered.connect(self.Bhistogram_file)  # B直方图
        # 图像增强
        self.pseenhance.triggered.connect(self.pseenhance_file)  # 伪彩色增强
        self.realenhance.triggered.connect(self.realenhance_file)  # 真彩色增强
        self.histogramequal.triggered.connect(self.histogramequal_file)  # 直方图均衡
        self.NTSC.triggered.connect(self.NTSC_file)  # NTSC颜色模型
        self.YCbCr.triggered.connect(self.YCbCr_file)  # YCbCr颜色模型
        self.HSV.triggered.connect(self.HSV_file)  # HSV颜色模型
        # 阈值分割
        self.divide.triggered.connect(self.divide_file)  # 阈值分割
        # 形态学处理
        self.morphology.triggered.connect(self.morphology_file)  # 形态学处理
        # 特征提取
        self.feature.triggered.connect(self.feature_file)  # 特征提取
        # 图像分类与识别
        # self.imageclassify.triggered.connect(self.imageclassify_file)#图像分类与识别
        # 按钮功能
        # 浏览
        self.Scan.clicked.connect(self.scan_file)
        # 上一张
        self.Back.clicked.connect(self.pre_file)
        # 下一张
        self.Next.clicked.connect(self.next_file)
        # 添加水印
        self.Mark.clicked.connect(self.mark_file)  # 添加水印
        # 放大
        self.Magnify.clicked.connect(self.manify_file)
        # 顺时针旋转90°
        self.R90CW.clicked.connect(self.r90cw_file)
        # 逆时针旋转90°
        self.R90CCW.clicked.connect(self.r90ccw_file)

    def read_file(self):
        # 选取文件
        filename, filetype = QFileDialog.getOpenFileName(self, "打开文件", "imagetest", "All Files(*);;Text Files(*.png)")
        print(filename, filetype)
        self.lineEdit.setText(filename)
        self.label_pic.setPixmap(QPixmap(filename))

    def save_file(self):
        # 获取文件路径
        file_path = self.lineEdit.text()
        if file_path == '':
            self.showMessageBox()
        else:
            # 用全局变量保存所有需要保存的变量在内存中的值。
            file_name = QFileDialog.getSaveFileName(self, "文件保存", "imagetest/save", "All Files (*);;Text Files (*.png)")
            print(file_name[0])
            btn = button_self()
            btn.file_save(file_path, file_name[0])

    def zoomin_file(self):  # 放大
        file_path = self.lineEdit.text()
        if file_path == '':
            self.showMessageBox()
        else:
            imageedit_self.imagemagnification(file_path)

    def zoomout_file(self):  # 缩小
        file_path = self.lineEdit.text()
        if file_path == '':
            self.showMessageBox()
        else:
            imageedit_self.imagereduction(file_path)

    def gray_file(self):  # 灰度
        file_path = self.lineEdit.text()
        if file_path == '':
            self.showMessageBox()
        else:
            imageedit_self.imagegray(file_path)

    def light_file(self):  # 亮度
        file_path = self.lineEdit.text()
        if file_path == '':
            self.showMessageBox()
        else:
            imageedit_self.imagebrightness(file_path, 1.3, 3)

    def rotate_file(self):  # 旋转
        file_path = self.lineEdit.text()
        if file_path == '':
            self.showMessageBox()
        else:
            imageedit_self.imagerotate(file_path)

    def screenshots_file(self):  # 截图
        file_path = self.lineEdit.text()
        if file_path == '':
            self.showMessageBox()
        else:
            imageedit_self.imagegrab(file_path)

    # 变换
    def fft_file(self):  # 傅里叶变换
        file_path = self.lineEdit.text()
        if file_path == '':
            self.showMessageBox()
        else:
            fourier_self.fourier_transform(file_path)

    def cos_file(self):  # 离散余弦变换
        file_path = self.lineEdit.text()
        if file_path == '':
            self.showMessageBox()
        else:
            fourier_self.Dct_image(file_path)

    def radon_file(self):  # Radon变换
        file_path = self.lineEdit.text()
        if file_path == '':
            self.showMessageBox()
        else:
            fourier = fourier_self()
            fourier.image_radon(file_path)

    # 噪声
    def gauss_file(self):  # 高斯噪声
        file_path = self.lineEdit.text()
        if file_path == '':
            self.showMessageBox()
        else:
            noise_self.addGaussianNoise(file_path, 0.01)  # 添加10%的高斯噪声

    def sault_file(self):  # 椒盐噪声
        file_path = self.lineEdit.text()
        if file_path == '':
            self.showMessageBox()
        else:
            noise_self.saltpepper(file_path, 0.01)  # 添加10%的椒盐噪声

    def spot_file(self):  # 斑点噪声
        file_path = self.lineEdit.text()
        if file_path == '':
            self.showMessageBox()
        else:
            noise_self.speckle_img(file_path)  # 添加斑点噪声

    def poisson_file(self):  # 泊松噪声
        file_path = self.lineEdit.text()
        if file_path == '':
            self.showMessageBox()
        else:
            noise_self.poisson_img(file_path)  # 添加泊松噪声

    # 滤波
    def highpass_file(self):  # 高通滤波
        file_path = self.lineEdit.text()
        if file_path == '':
            self.showMessageBox()
        else:
            filter_self.high_pass_filter(file_path)

    def lowpass_file(self):  # 低通滤波
        file_path = self.lineEdit.text()
        if file_path == '':
            self.showMessageBox()
        else:
            filter_self.low_pass_filter(file_path)

    def linearsmooth_file(self):  # 平滑滤波（线性）
        file_path = self.lineEdit.text()
        if file_path == '':
            self.showMessageBox()
        else:
            filter_self.Linear_Smooth_Filter(file_path)

    def nonlinear_file(self):  # 平滑滤波（非线性）
        file_path = self.lineEdit.text()
        if file_path == '':
            self.showMessageBox()
        else:
            filter1 = filter_self()
            filter1.NonLinear_Smooth_Filter(file_path)  #

    def linearsharpen_file(self):  # 锐化滤波（线性）
        file_path = self.lineEdit.text()
        if file_path == '':
            self.showMessageBox()
        else:
            filter1 = filter_self()
            filter1.Sharpen_Filter(file_path)

    def nonlinearsharp_file(self):  # 锐化滤波（非线性）
        file_path = self.lineEdit.text()
        if file_path == '':
            self.showMessageBox()
        else:
            filter1 = filter_self()
            filter1.NonLinear_Smooth_Filter(file_path)  #

    # 直方图统计
    def Rhistogram_file(self):  # R直方图
        file_path = self.lineEdit.text()
        if file_path == '':
            self.showMessageBox()
        else:
            histogram_self.R_histogram(file_path)  #

    def Ghistogram_file(self):  # G直方图
        file_path = self.lineEdit.text()
        if file_path == '':
            self.showMessageBox()
        else:
            histogram_self.G_histogram(file_path)  #

    def Bhistogram_file(self):  # B直方图
        file_path = self.lineEdit.text()
        if file_path == '':
            self.showMessageBox()
        else:
            histogram_self.B_histogram(file_path)  #

    # 图像增强
    def pseenhance_file(self):  # 伪彩色增强
        file_path = self.lineEdit.text()
        if file_path == '':
            self.showMessageBox()
        else:
            image_enhance = image_enhance_self()
            image_enhance.Color(file_path)  #

    def realenhance_file(self):  # 真彩色增强
        file_path = self.lineEdit.text()
        if file_path == '':
            self.showMessageBox()
        else:
            image_enhance = image_enhance_self()
            image_enhance.Color(file_path)  #

    def histogramequal_file(self):  # 直方图均衡
        file_path = self.lineEdit.text()
        if file_path == '':
            self.showMessageBox()
        else:
            image_enhance_self.colorhistogram(file_path)  #

    def NTSC_file(self):  # NTSC颜色模型
        file_path = self.lineEdit.text()
        if file_path == '':
            self.showMessageBox()
        else:
            image_enhance_self.colorhistogram(file_path)  #

    def YCbCr_file(self):  # YCbCr颜色模型
        file_path = self.lineEdit.text()
        if file_path == '':
            self.showMessageBox()
        else:
            image_enhance_self.ycrcbimage(file_path)  #

    def HSV_file(self):  # HSV颜色模型
        file_path = self.lineEdit.text()
        if file_path == '':
            self.showMessageBox()
        else:
            image_enhance_self.hsvimage(file_path)  #

    # 阈值分割
    def divide_file(self):  # 阈值分割
        file_path = self.lineEdit.text()
        if file_path == '':
            self.showMessageBox()
        else:
            threshold_self.threshold_image(file_path)  #

    # 形态学处理
    def morphology_file(self):  # 形态学处理
        file_path = self.lineEdit.text()
        if file_path == '':
            self.showMessageBox()
        else:
            morphology_self.morphology(file_path)

    # 特征提取
    def feature_file(self):  # 特征提取
        file_path = self.lineEdit.text()
        if file_path == '':
            self.showMessageBox()
        else:
            feature_self.feature_detection(file_path)  #

    # 按钮功能
    # 浏览
    def scan_file(self):
        file_path = self.lineEdit.text()
        if file_path == '':
            self.showMessageBox()
        else:
            btn = button_self()
            btn.scan_pic(file_path)  #

    # 上一张
    def pre_file(self):
        file_path = self.lineEdit.text()
        if file_path == '':
            self.showMessageBox()
        else:
            btn = button_self()
            pre_path = btn.pre_file(file_path)  #
            self.lineEdit.setText('')
            self.lineEdit.setText(pre_path)
            self.label_pic.setPixmap(QPixmap(pre_path))

    # 下一张
    def next_file(self):
        file_path = self.lineEdit.text()
        if file_path == '':
            self.showMessageBox()
        else:
            btn = button_self()
            next_path = btn.next_file(file_path)  #
            self.lineEdit.setText('')
            self.lineEdit.setText(next_path)
            self.label_pic.setPixmap(QPixmap(next_path))

    # 添加水印
    def mark_file(self):  #
        file_path = self.lineEdit.text()
        if file_path == '':
            self.showMessageBox()
        else:
            btn = button_self()
            btn.mark_pic(file_path)  #

    # 图片放大
    def manify_file(self):
        file_path = self.lineEdit.text()
        if file_path == '':
            self.showMessageBox()
        else:
            btn = button_self()
            btn.zoomout_pic(file_path)

    # 顺时针旋转90°
    def r90cw_file(self):
        file_path = self.lineEdit.text()
        if file_path == '':
            self.showMessageBox()
        else:
            btn = button_self()
            btn.cw_pic(file_path)

    # 顺时针旋转90°
    def r90ccw_file(self):
        file_path = self.lineEdit.text()
        if file_path == '':
            self.showMessageBox()
        else:
            btn = button_self()
            btn.rcw_pic(file_path)

    def showMessageBox(self):
        res_3 = QMessageBox.warning(self, "警告", "请选择文件，再执行该操作！", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    ui = mywindow()
    ui.show()
    sys.exit(app.exec_())
