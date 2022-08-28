# -*- coding:utf-8 -*-
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from Dataset import dataset
from scipy.io import arff

if __name__ == "__main__":
    data, _ = arff.loadarff("mnist_784.arff")
    data = pd.DataFrame(data).to_numpy()
    temp = []
    for i in range(len(data)):
        temp.append(int(data[i, -1]))
    data = np.hstack([data[:, :-1], np.array(temp).reshape(-1, 1)])
    data = pd.DataFrame(data)
    data = dataset(data.iloc[:, :])
    '''********PCA保留特征数量与方差和占比关系********'''
    # r = []
    # for n in range(1, 501, 2):
    #     pca = PCA(n)
    #     pca.fit(data.x)
    #     r.append(np.sum(pca.explained_variance_ratio_))

    # plt.plot(range(1, 501, 2), r)
    # plt.grid()
    # plt.yticks(np.arange(0, 1.01, 0.05))
    # plt.xticks(np.arange(0, 501, 25))
    # plt.xlabel("保留特征数量")
    # plt.ylabel("方差和占比")
    # plt.savefig("PCA降维.png")
    # plt.show()
    '''********根据图像选择保留特征数量n=350'''
    pca = PCA(350)
    pca.fit(data.x)
    ret_x = pca.transform(data.x)
    # 将特征向量与标签合并
    ret_data = np.hstack([ret_x, data.y.reshape(-1, 1)])
    print(ret_data.shape)
    ret_data = pd.DataFrame(ret_data)
    # 储存降维后数据 作为未来训练与测试的基础
    ret_data.to_csv("pca_mnist.csv")
