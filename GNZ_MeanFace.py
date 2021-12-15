# -*- coding: utf-8 -*- 
# @Time : 2021/12/3 12:16 
# @Author : Tianyi  
# @File : GNZ_MeanFace.py

from numpy import *
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]


def readImage(path):
    #循环遍历lfw数据集下的所有子文件
    for img_file in os.listdir(path): # 直接遍历lfwtest（25张）下文件
        img = Image.open(path+ '/' + img_file)
        img = img.resize((63, 63))  # 归一化 缩放
        imglist = np.array(img, dtype=np.float64)  # 转为float64类型的Numpy数组
    return imglist

def convertToGrayscale(image):
    ycbcr_image = image.convert('YCbCr')
    (y, cb, cr) = ycbcr_image.split()
    return y

def computePCA(X):
    # 输入：矩阵X，其中该矩阵中存储训练数据，每一行为一条训练数据
    # 返回：投影矩阵（按照维度的重要性排序）、方差和均值 """
    num_data, dim = X.shape # x-data, y-dim

    # 去中心
    mean_X= X.mean(axis=0)
    X = X - mean_X

    if dim > num_data:
        # 紧致
        M = dot(X, X.T) # cov
        e, EV = linalg.eigh(M)
        tmp = dot(X.T,EV).T
        V = tmp[::-1]  # 由于最后的特征向量是我们所需要的，所以需要将其逆转
        S = sqrt(e)[::-1]  # 由于特征值是按照递增顺序排列的，所以需要将其逆转
        for i in range(V.shape[1]):
            V[:, i] /= S
    else:
        # PCA- 使用 SVD 方法
        U, S, V = linalg.svd(X)
        U, S, V = linalg.svd(X)
        V = V[:num_data]  # 仅仅返回前 nun_data 维的数据才合理

    # 降维后的数据：
    lowdata = X * V
    print("lowdata:",lowdata.shape) #test
    # return projection matrix, square_error, mean value, lowdata
    return V,S,mean_X, lowdata
if __name__ == '__main__':
    imglist = get_imlist("lfwtest")
    img = Image.open(imglist[0]).convert('L')
    img = array(img)
    print(len(imglist)) # test
    m, n = img.shape[0:2]  # image size height and width
    print("m:", m, "n", n)
    plt.figure()
    plt.imshow(img,cmap='gray')
    plt.show()

    vectorized_imgs = []
    for i in range(0,len(imglist)):
        img = array(Image.open(imglist[i]).convert('L'))
        vectorized_imgs.append(img.ravel()) # to 1d
        print(vectorized_imgs[i], "len:", len(vectorized_imgs[i]))
        # if (len(vectorized_imgs[i]) == 120000):
        #     plt.figure()
        #     plt.imshow(img, cmap='gray')
        #     plt.show()
    # print(len(vectorized_imgs)) # test

    # average face test
    # gamma = np.array(vectorized_imgs)
    # aver_face = np.mean(gamma, axis=0)
    # plt.figure()
    # plt.imshow(aver_face.reshape(m,n), cmap='gray')
    # plt.show()

    # imgMat = array([array(Image.open(img).convert('L')).flatten()
    #                       for img in imglist], 'f')

    imgMat = np.array(vectorized_imgs)
    # print(imgMat) # test
    V, S, imgMean,lowdata = computePCA(imgMat)
    print(len(V))
    plt.figure()
    plt.gray()
    plt.subplot(2, 4, 1), plt.title("MeanFace")
    plt.axis('off')
    plt.imshow(imgMean.reshape(m, n), cmap='gray') # meanFace
    # print(V[2].reshape(m, n)) # test
    for i in range(7): # 用不用len(v)
        plt.subplot(2, 4, i+2), plt.title("EigenFace")
        plt.axis('off')
        plt.imshow(V[i].reshape(m, n))
    plt.show()
