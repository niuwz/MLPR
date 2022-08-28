# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from scipy.io import arff
from skimage import feature as sf

from Dataset import dataset


def get_hog_array(pixels):
    '''
    提取原图像的HOG特征
    pixels: 手写数字图像数据的像素，1*784维
    hog_array: 提取到的HOG特征
    '''
    imgs = pixels.reshape(28, 28)
    hog_array = sf.hog(imgs, orientations=8, pixels_per_cell=(
        7, 7), cells_per_block=(4, 4))
    return hog_array


if __name__ == "__main__":
    # 读取数据
    data, _ = arff.loadarff("data/origin/mnist_784.arff")
    data = pd.DataFrame(data).to_numpy()
    temp = []
    for i in range(len(data)):
        temp.append(int(data[i, -1]))
    data = np.hstack([data[:, :-1], np.array(temp).reshape(-1, 1)])
    data = pd.DataFrame(data)
    data = dataset(data.iloc[:, :])
    # 逐条提取HOG特征
    hog_vectors = []
    for i in data.x:
        hg = get_hog_array(i)
        hog_vectors.append(hg)
    hog_vectors = np.array(hog_vectors)
    # 特征向量与标签合并
    ret_data = np.hstack([hog_vectors, data.y.reshape(-1, 1)])
    print(ret_data.shape)
    ret_data = pd.DataFrame(ret_data)
    # 储存HOG特征向量 作为未来训练与测试的基础
    ret_data.to_csv("hog_mnist.csv")
