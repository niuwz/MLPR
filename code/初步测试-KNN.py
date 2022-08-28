# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import time
from classifications import *
from Dataset import dataset


if __name__ == "__main__":
    # 读取前2000条数据进行初步训练，第2001~3000条数据进行初步测试
    data = pd.read_csv("pca_mnist.csv", index_col=0)
    # data = pd.read_csv("hog_mnist.csv", index_col=0)
    train_data = dataset(data.iloc[:2000, :])
    test_data = dataset(data.iloc[2000:3000, :])
    # 对比K的大小对分类准确率的影响
    accs = []
    for k in range(1, 31):
        knn = KNN(train_data, k)
        cnt = 0
        for i in range(len(test_data)):
            x, y = test_data[i]
            if knn.predict(x) == y:
                cnt += 1
        acc = cnt/len(test_data)*100
        accs.append(acc)
        print("k={} 分类准确率:{}%".format(k, acc))
    # 对比训练数据数量对运行时间和准确率的影响
    # 这里采用的测试数据数据量为5000
    times = []
    accs2 = []
    data_size = []
    # 固定k=5
    k = 5
    for n in range(1, 10):
        data_size.append(n*5000)
        train_data_i = dataset(data.iloc[:5000*n, :])
        test_data_i = dataset(data.iloc[60000:65000, :])
        knn = KNN(train_data_i, k)
        cnt = 0
        # 记录起始时间
        t_start = time.time()
        for i in range(len(test_data_i)):
            x, y = test_data_i[i]
            if knn.predict(x) == y:
                cnt += 1
        # 记录结束时间
        t_end = time.time()
        # 计算运行时间
        t = t_end-t_start
        times.append(t)
        # 计算准确率(百分数)
        acc = cnt/len(test_data_i)*100
        accs2.append(acc)
        print("数据数量:{} 所用时间:{:.4f}s 分类准确率:{}%".format(n*5000, t, acc))

    # 绘制曲线
    plt.figure("数据数量对运行时间影响")
    plt.plot(data_size, times, color='k', linestyle='-.', linewidth=2,
             marker='o', markersize=8, markeredgecolor='k', markerfacecolor='red')
    plt.xlabel('数据数量')
    plt.grid()
    plt.ylabel("时间/s")

    plt.figure("数据数量对准确率影响")
    plt.plot(data_size, accs2, color='k', linestyle='-.', linewidth=2,
             marker='o', markersize=8, markeredgecolor='k', markerfacecolor='red')
    plt.xlabel('数据数量')
    plt.grid()
    plt.ylabel("准确率%")

    plt.figure("K大小对准确率影响")
    plt.plot(accs, color='g', linestyle='--', linewidth=2,
             marker='^', markersize=4, markeredgecolor='k', markerfacecolor='red')
    plt.xlabel('K')
    plt.grid()
    plt.ylabel("准确率%")
    plt.show()
