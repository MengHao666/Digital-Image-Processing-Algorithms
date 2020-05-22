import numpy as np
import matplotlib.pyplot as plt

my_contour1 = np.asarray(
    [[41, 41], [43, 41, ], [45, 41, ], [47, 41], [49, 41], [51, 41], [53, 41], [55, 41],
     [57, 41], [59, 41], [61, 41], [63, 41], [65, 41], [67, 41], [69, 41], [71, 41],  # 41列

     [73, 41], [73, 43], [73, 45], [73, 47], [73, 49], [73, 51], [73, 53], [73, 55], [73, 57],  # 73行
     [73, 59], [73, 61], [73, 63], [73, 65], [73, 67], [73, 69], [73, 71], [73, 73],

     [71, 73], [69, 73], [67, 73], [65, 73], [63, 73], [61, 73], [59, 73], [57, 73], [55, 73],  # 73行
     [53, 73], [51, 73], [49, 73], [47, 73], [45, 73], [43, 73], [41, 73],

     [41, 71], [41, 69], [41, 67], [41, 65], [41, 63], [41, 61], [41, 59], [41, 57], [41, 55],  # 41行
     [41, 53], [41, 51], [41, 49], [41, 47], [41, 45], [41, 43]
     ])

my_contour2 = np.asarray([

    [73, 41], [73, 43], [73, 45], [73, 47], [73, 49], [73, 51], [73, 53], [73, 55], [73, 57],  # 73行
    [73, 59], [73, 61], [73, 63], [73, 65], [73, 67], [73, 69], [73, 71], [73, 73],

    [71, 73], [69, 73], [67, 73], [65, 73], [63, 73], [61, 73], [59, 73], [57, 73], [55, 73],  # 73行
    [53, 73], [51, 73], [49, 73], [47, 73], [45, 73], [43, 73], [41, 73],

    [41, 71], [41, 69], [41, 67], [41, 65], [41, 63], [41, 61], [41, 59], [41, 57], [41, 55],  # 41行
    [41, 53], [41, 51], [41, 49], [41, 47], [41, 45], [41, 43],

    [41, 41], [43, 41, ], [45, 41, ], [47, 41], [49, 41], [51, 41], [53, 41], [55, 41],
    [57, 41], [59, 41], [61, 41], [63, 41], [65, 41], [67, 41], [69, 41], [71, 41],  # 41列

])


def DFT(sig):
    N = sig.shape[0]
    contours_complex = np.empty(N, dtype=complex)
    contours_complex = sig[:, 0] + sig[:, 1] * 1j  # 横坐标作为实数部分
    V = np.array([[np.exp(-1j * 2 * np.pi * v * y / N) for v in range(N)] for y in range(N)])
    return contours_complex.dot(V)


def reconstruct(au, M):
    N = au.shape[0]
    # print("N=", N)
    res = np.zeros(N, dtype=complex)
    for k in range(N):
        for u in range(M):
            res[k] += au[u] * np.exp(1j * 2 * np.pi * u * k / M)
        res[k] /= N
    # print("res=", res)
    # print("res_=", np.stack((res.real, res.imag), -1))
    return np.stack((res.real, res.imag), -1)


def plotarray(q, length=100):
    img = np.zeros((length, length))
    for i in range(length):
        for j in range(length):
            img[i][j] = 255

    for m in range(q.shape[0]):
        # print(q[m][0])
        # print(q[m][1])
        img[int(q[m][0])][int(q[m][1])] = 0
    return img


# print("my_contour.shape=", my_contour2.shape)

# 进行离散傅立叶变换
fd = DFT(my_contour1)
# print("fd2=", fd2)
plot_contour1 = plotarray(my_contour1)
res64 = plotarray(reconstruct(fd, 64))
res63 = plotarray(reconstruct(fd, 63))
res50 = plotarray(reconstruct(fd, 50))
res32 = plotarray(reconstruct(fd, 32))
res16 = plotarray(reconstruct(fd, 16))
res8 = plotarray(reconstruct(fd, 8))
res4 = plotarray(reconstruct(fd, 4))

plt.figure()
plt.subplots_adjust(wspace=0, hspace=1)  # 调整子图间距

plt.subplot(421)
plt.imshow(plot_contour1, cmap='gray')
plt.title('Original Image')

plt.subplot(422)
plt.imshow(res64, cmap='gray')
plt.title('M=64')

plt.subplot(423)
plt.imshow(res63, cmap='gray')
plt.title('M=63')

plt.subplot(424)
plt.imshow(res50, cmap='gray')
plt.title('M=50')

plt.subplot(425)
plt.imshow(res32, cmap='gray')
plt.title('M=32')

plt.subplot(426)
plt.imshow(res16, cmap='gray')
plt.title('M=16')

plt.subplot(427)
plt.imshow(res8, cmap='gray')
plt.title('M=8')

plt.subplot(428)
plt.imshow(res4, cmap='gray')
plt.title('M=4')

plt.show()
