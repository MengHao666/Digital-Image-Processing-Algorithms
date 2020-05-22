import math
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


# 复数类
class complex:
    def __init__(self):
        self.real = 0.0  # 实部
        self.imag = 0.0j  # 虚部 必须为int或者float类


# 复数乘法
def mul_ee(complex0, complex1):
    complex_ret = complex()
    complex_ret.real = complex0.real * complex1.real - complex0.imag * complex1.imag  # 实*实-虚*虚
    complex_ret.imag = complex0.real * complex1.imag + complex0.imag * complex1.real  # 实*虚+实*虚
    return complex_ret


# 复数加法
def add_ee(complex0, complex1):
    complex_ret = complex()
    complex_ret.real = complex0.real + complex1.real
    complex_ret.imag = complex0.imag + complex1.imag
    return complex_ret


# 复数减法
def sub_ee(complex0, complex1):
    complex_ret = complex()
    complex_ret.real = complex0.real - complex1.real
    complex_ret.imag = complex0.imag - complex1.imag
    return complex_ret


# 对输入数据进行按位倒叙排列(雷德算法)
def forward_input_data(input_data, num):
    j = num // 2
    for i in range(1, num - 2):
        if (i < j):
            complex_tmp = input_data[i]
            input_data[i] = input_data[j]
            input_data[j] = complex_tmp
            # print ("forward x[%d] <==> x[%d]" % (i, j))
        k = num // 2
        while (j >= k):
            j = j - k
            k = k // 2
        j = j + k


# 实现一维FFT
def fft_1d(in_data, num):
    PI = 3.1415926
    forward_input_data(in_data, num)  # 实现将数据按位倒序
    # 计算蝶形级数，也就是迭代次数
    M = 1  # M用于记录num的二分次数
    tmp = num / 2;  # 满足num = 2^m，基2算法
    while (tmp != 1):
        M = M + 1
        tmp = tmp / 2
    #  print("FFT level：%d" % M)
    complex_ret = complex()
    for L in range(1, M + 1):  # 二分次数
        B = int(math.pow(2, L - 1))  # B为指数函数返回值，为float，需要转换integer
        for J in range(0, B):  # 分组数
            P = math.pow(2, M - L) * J
            for K in range(J, num, int(math.pow(2, L))):  # 控制偏移值
                # print("L:%d B:%d, J:%d, K:%d, P:%f" % (L, B, J, K, P))
                complex_ret.real = math.cos((2 * PI / num) * P)
                complex_ret.imag = -math.sin((2 * PI / num) * P)
                complex_mul = mul_ee(complex_ret, in_data[K + B])
                complex_add = add_ee(in_data[K], complex_mul)
                complex_sub = sub_ee(in_data[K], complex_mul)
                in_data[K] = complex_add
                in_data[K + B] = complex_sub
            #  print("A[%d] real: %f, image: %f" % (K, in_data[K].real, in_data[K].imag))
            #  print("A[%d] real: %f, image: %f" % (K + B, in_data[K + B].real, in_data[K + B].imag))\


# 二维FFT的实现方法是先对行做FFT将结果放回该行，然后再对列做FFT结果放在该列，结果就是二维FFT
def fft_2d(in_data):  # in_data已经是属于复数的集合
    axis = in_data.shape  # axis的结果刚好是行和列
    for i in range(axis[0]):  # 先对行做处理
        fft_1d(in_data[i, :], axis[1])
    for j in range(axis[1]):  # 再对列做处理
        fft_1d(in_data[:, j], axis[0])


def ifft_1d(in_data, num):
    # 先求共轭qq
    for i in range(num):
        in_data[i].imag = - in_data[i].imag
    # 调用傅里叶变换
    fft_1d(in_data, num)
    # 求时域点共轭
    for i in range(num):
        in_data[i].real = in_data[i].real / num;
        in_data[i].imag = -in_data[i].imag / num;


def ifft_2d(in_data):
    axis = in_data.shape  # axis的结果刚好是行和列
    for i in range(axis[0]):  # 先对行做处理
        ifft_1d(in_data[i, :], axis[1])
    for j in range(axis[1]):  # 再对列做处理
        ifft_1d(in_data[:, j], axis[0])


def fft_shift(in_data):
    m, n = in_data.shape
    k = m // 2
    t = n // 2
    for i in range(k):
        for j in range(t):
            temp = in_data[i][j]
            in_data[i][j] = in_data[i + k][j + t]
            in_data[i + k][j + t] = temp
        for j in range(t, n):
            temp = in_data[i][j]
            in_data[i][j] = in_data[i + k][j - t]
            in_data[i + k][j - t] = temp


def ifft_shift(in_data):
    fft_shift(in_data)


def complex_to_realcomplex(input_data):
    data = []
    m, n = input_data.shape
    for i in range(m):
        for j in range(n):
            data.append(input_data[i][j].real + input_data[i][j].imag)
    data = np.array(data).reshape((m, n))
    return data


def test_fft_1d():
    in_data = [2, 3, 4, 4, 5, 8, 6, 4]  # 待测试的8点元素
    # 变量data为长度为8、元素为complex类实例的list，用于存储输入数据
    data = [(complex()) for i in range(len(in_data))]
    # 将8个测试点转换为complex类的形式，存储在变量data中
    for i in range(len(in_data)):
        data[i].real = in_data[i]
        data[i].imag = 0.0j
    # 输出FFT需要处理的数据
    print("The input data:")
    for i in range(len(in_data)):
        print("x[%d] real: %f, imag: %f" % (i, data[i].real, data[i].imag))
    fft_1d(data, 8)  # 进行傅里叶变换处理
    # 输出经过FFT处理后的结果
    print("The output data:")
    for i in range(len(in_data)):
        print("X[%d] real: %f, imag: %f" % (i, data[i].real, data[i].imag))


# test_fft_1d()
def test_fft_2d():
    # in_data = [[2, 3, 4,4],[5,8,6,4],[7,4,3,5],[9,6,4,2]]  # 待测试的8点元素
    # 变量data用于存储输入数据
    in_data = cv.imread('7.jpg', 0)
    m, n = in_data.shape
    data = [[(complex()) for i in range(n)] for j in range(m)]
    # print('data\n',np.array(data).shape)
    # 将测试点转换为complex类的形式，存储在变量data中
    for i in range(m):
        for j in range(n):
            data[i][j].real = in_data[i][j]
            data[i][j].imag = 0.0j
    # 输出FFT需要处理的数据
    # print("The input data:")
    # for i in range(m):
    #     for j in range(n):
    #         print("x[%d][%d] real: %f, imag: %f" % (i,j,data[i][j].real, data[i][j].imag))
    data = np.array(data)
    fft_2d(data)  # 进行傅里叶变换处理
    # 输出经过FFT处理后的结果
    # print("The output data:")
    # for i in range(4):
    #     for j in range(5):
    #         print("x[%d][%d] real: %f, imag: %f" % (i,j, data[i][j].real, data[i][j].imag))
    # 将频谱低频从左上角移动至中心
    fft_shift(data)
    # k = np.max(np.log(1 + np.abs(data)))
    # res = (255 / k) * (np.log(1 + np.abs(data)))
    datac = complex_to_realcomplex(data)
    res = np.log(np.abs(datac))
    # 展示结果
    plt.subplot(121), plt.imshow(in_data, 'gray'), plt.title('Original Fourier')
    plt.axis('off')
    plt.subplot(122), plt.imshow(res, 'gray'), plt.title('Fourier Fourier')
    plt.axis('off')
    plt.show()


# test_fft_2d()

def paint_my_square(img, size=2, bias=None):
    if bias is None:
        bias = [0, 0]
    img2 = np.zeros(img.shape, img.dtype)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i][j] = 0
    d = int(img.shape[0] / 2 - size / 2)
    for i in range(size):
        for j in range(size):
            img[d + i + bias[0]][d + j + bias[1]] = 255
            # print("img[{}][{}] = 255".format(d + i, d + j))

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img2[i][j] = img[i][j]

    return img2


def test_ifft_2d():
    # in_data = [[2, 3, 4,4],[5,8,6,4],[7,4,3,5],[9,6,4,2]]
    # 变量data用于存储输入数据
    # img = cv2.imread("Lenna.png", cv2.IMREAD_COLOR)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = np.zeros((128, 128), "uint8")
    gray = paint_my_square(gray)
    # gray = paint_my_square(gray,32)
    # gray = paint_my_square(gray, 32, bias=[-40, -40])
    plt.imshow(gray, cmap='gray')
    plt.show()

    # in_data = cv.imread('7.jpg', 0)
    in_data = gray
    m, n = in_data.shape
    data = [[(complex()) for i in range(n)] for j in range(m)]
    # print('data\n',np.array(data).shape)
    # 将测试点转换为complex类的形式，存储在变量data中
    for i in range(m):
        for j in range(n):
            data[i][j].real = in_data[i][j]
            data[i][j].imag = 0.0j
    data = np.array(data)
    fft_2d(data)  # 进行傅里叶变换处理
    # 将频谱低频从左上角移动至中心
    fft_shift(data)
    data_shift = data.copy()
    # k = np.max(np.log(1 + np.abs(data)))
    # res = (255 / k) * (np.log(1 + np.abs(data)))
    datac = complex_to_realcomplex(data)
    res = np.log(np.abs(datac))
    # 傅里叶逆变换
    # 先将低频移回原位置
    ifft_shift(data_shift)
    ifft_2d(data_shift)
    datac2 = complex_to_realcomplex(data_shift)
    res2 = np.abs(datac2)

    # 展示结果
    # plt.subplot(121), plt.imshow(res, 'gray'), plt.title('Fourier Fourier')
    plt.subplot(121), plt.imshow(np.log(1 + abs(datac)), 'gray'), plt.title('Fourier Fourier')
    plt.axis('off')
    plt.subplot(122), plt.imshow(res2, 'gray'), plt.title('Fourier-to-image')
    plt.axis('off')
    plt.show()


test_ifft_2d()
