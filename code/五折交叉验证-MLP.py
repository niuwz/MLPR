from Dataset import dataset
from classifications import *
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

warnings.filterwarnings('ignore')


def MLP_train(train_data, validation_data, test_data, hidden_cells, lr, batch_size, epochs):
    '''MLP测试过程 MLP原理详见classification中MLP类'''
    mlp = MLP(350, 10, lr=lr, hidden_cells=hidden_cells)
    losses = []
    for epoch in range(epochs):
        loss = 0
        for i in range(0, len(train_data)-batch_size, batch_size):
            x, y = train_data[i:i+batch_size]
            x = x.reshape(batch_size, -1)
            y = onehot(y)
            loss += mlp.train(x, y)
        losses.append(loss.mean()/i*batch_size)
        if ((epoch+1) % 1000) == 0:
            print("epoch:{}:loss:{}".format(epoch+1, loss.mean()/i*batch_size))
            # 学习率衰减
            mlp.lr *= 0.95
    plt.plot(losses)
    plt.xlabel('Epoches')
    plt.ylabel("Loss")
    # 验证集数据预测以及正确率计算
    cnt = 0
    n = len(validation_data)
    for i in range(len(validation_data)):
        x, y = validation_data[i]
        x = x.reshape(1, -1)
        if mlp.test(x, y):
            cnt += 1
    vali_acc = cnt/n*100
    # 测试集数据预测以及正确率计算
    cnt = 0
    n = len(test_data)
    for i in range(len(test_data)):
        x, y = test_data[i]
        x = x.reshape(1, -1)
        if mlp.test(x, y):
            cnt += 1
    test_acc = cnt/n*100

    print("MLP验证集分类准确率:{:.4f}% 测试集分类准确率:{:.4f}%".format(vali_acc, test_acc))
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
    for i in range(5):
        train_data = dataset(pd.DataFrame(
            np.array(datas[:i]+datas[i+1:]).reshape(-1, VECTOR_LENGTH+1)))
        validation_data = dataset(datas[i])
        '''MLP训练与正确率计算'''
        vali_loss, test_loss = MLP_train(train_data, validation_data, test_data, hidden_cells=(64, ),
                                         lr=0.001, batch_size=100, epochs=3000)
        vali_losses.append(vali_loss)
        test_losses.append(test_loss)

    plt.figure("验证数据集")
    plt.plot(vali_losses, color='k', linestyle='-.', linewidth=2,
             marker='o', markersize=8, markeredgecolor='k', markerfacecolor='red')
    plt.xlabel('分类器')
    plt.ylabel("验证集准确率%")
    plt.yticks(np.arange(94, 101, 2))
    plt.xticks(np.arange(0, 5, 1))
    plt.grid()

    plt.figure("测试数据集")
    plt.plot(test_losses, color='k', linestyle='-.', linewidth=2,
             marker='o', markersize=8, markeredgecolor='k', markerfacecolor='red')
    plt.xlabel('分类器')
    plt.ylabel("测试集准确率%")
    plt.yticks(np.arange(90, 101, 2))
    plt.xticks(np.arange(0, 5, 1))
    plt.grid()

    plt.show()
