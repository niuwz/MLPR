# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd


class dataset():
    '''封装数据集类'''

    def __init__(self, data: pd.DataFrame):
        '''获取并简单处理数据'''
        # 除最后一列外均为样本特征向量
        self.x = data.iloc[:, :-1].to_numpy(dtype=np.float32)
        # 最后一列为样本标签
        self.y = data.iloc[:, -1].to_numpy(dtype=np.float32)

    def __len__(self):
        '''返回样本数量'''
        return len(self.y)

    def __getitem__(self, idx):
        '''获取样本及对应标签'''
        return self.x[idx], self.y[idx]
