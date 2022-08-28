# -*- coding:utf-8 -*-
from Dataset import dataset
from classifications import *
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

warnings.filterwarnings('ignore')


def KNN_predict(k, train_data, validation_data, test_data):
    '''KNN测试过程 KNN原理详见classification中KNN类'''
    knn = KNN(train_data, k)
    cnt = 0
    # 验证集数据预测以及正确率计算
    for i in range(len(validation_data)):
        x, y = validation_data[i]
        if knn.predict(x) == y:
            cnt += 1
    vali_acc = cnt/len(validation_data)*100
    cnt = 0
    # 测试集数据预测以及正确率计算
    for i in range(len(test_data)):
        x, y = test_data[i]
        if knn.predict(x) == y:
            cnt += 1
    test_acc = cnt/len(test_data)*100
    print("KNN: k={} 验证集分类准确率:{:.4f}% 测试集分类准确率:{:.4f}%".format(k, vali_acc, test_acc))
    return vali_acc, test_acc


if __name__ == "__main__":
    data = pd.read_csv("pca_mnist.csv", index_col=0)
    # data = pd.read_csv("hog_mnist.csv", index_col=0)
    # 特征数量
    VECTOR_LENGTH = data[0].shape[1]-1
    # 留出测试集
    test_data = dataset(data.iloc[60000:, :])
    datas = []
    vali_losses = []
    test_losses = []
    # 五折交叉验证
    # 分为五个数据集
    for i in range(0, 60000, 12000):
        datas.append(data.iloc[i:i+12000, :])
    for k in range(1, 7):
        vali_losses = []
        test_losses = []
        for i in range(5):
            train_data = dataset(pd.DataFrame(
                np.array(datas[:i]+datas[i+1:]).reshape(-1, VECTOR_LENGTH+1)))
            validation_data = dataset(datas[i])
            '''KNN训练与正确率计算'''
            vali_loss, test_loss = KNN_predict(
                k, train_data, validation_data, test_data)
            vali_losses.append(vali_loss)
            test_losses.append(test_loss)
        plt.figure("测试数据集")
        plt.plot(range(5), test_losses, label=f'k={k}', linestyle='-.', linewidth=2,
                 marker='o', markersize=8, markeredgecolor='k', markerfacecolor='red')
        plt.grid()
        plt.legend()
        plt.xlabel("测试数据集")
        plt.ylabel("准确率")
        plt.xticks(np.arange(0, 5, 1))
        plt.yticks(np.arange(80, 101, 5))

        plt.figure("验证数据集")
        plt.plot(range(5), vali_losses, label=f'k={k}',  linestyle='-.', linewidth=2,
                 marker='o', markersize=8, markeredgecolor='k', markerfacecolor='red')
        plt.grid()
        plt.legend()
        plt.xlabel("验证数据集")
        plt.ylabel("准确率")
        plt.xticks(np.arange(0, 5, 1))
        plt.yticks(np.arange(80, 101, 5))
    plt.show()
