# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from classifications import *
from Dataset import dataset


if __name__ == "__main__":
    data = pd.read_csv("pca_mnist.csv", index_col=0)
    # data = pd.read_csv("hog_mnist.csv", index_col=0)
    train_data = dataset(data.iloc[:2000, :])
    test_data = dataset(data.iloc[2000:3000, :])

    bp = MLP(350, 10, lr=0.001)
    # bp = MLP(128, 10, lr=0.001)
    batch_size = 80
    losses = []
    # 梯度下降迭代过程
    for epoch in range(3000):
        loss = 0
        for i in range(0, len(train_data)-batch_size, batch_size):
            x, y = train_data[i:i+batch_size]
            x = x.reshape(batch_size, -1)
            y = onehot(y)
            loss += bp.train(x, y)
        losses.append(loss.mean()/i*batch_size)
        if (epoch % 100) == 0:
            print(loss.mean()/i*batch_size)

    cnt = 0
    # 计算测试集准确率
    n = len(test_data)
    for i in range(len(test_data)):
        x, y = test_data[i]
        x = x.reshape(1, -1)
        if bp.test(x, y):
            cnt += 1
    print(cnt/n)

    plt.plot(losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()
