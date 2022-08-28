# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from classifications import *
from Dataset import dataset

if __name__ == "__main__":
    # 读取前1000条数据进行初步训练
    data = pd.read_csv("pca_mnist.csv", index_col=0)
    # data = pd.read_csv("hog_mnist.csv", index_col=0)
    data = dataset(data.iloc[:1000, :])
    accs = []
    for i in range(10, 301, 50):
        osvm = OVRSVM(5, 0.5, i, 0.001)
        osvm.svm_train(data)
        acc = osvm.score(data)*100
        accs.append(acc)
        print("Train:{:.4f}%".format(acc))
    plt.figure()
    plt.plot(range(50, 301, 50), accs)
    plt.xlabel("迭代次数")
    plt.ylabel("正确率")
    plt.figure()
    plt.show()
