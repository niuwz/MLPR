# -*- coding:utf-8 -*-
from typing import List
import numpy as np


def onehot(y):
    '''将标签y从索引值转换为onehot向量'''
    d = y.shape[0]
    # 初始为0向量
    a = np.zeros((d, 10))
    # 类别索引对应值置为1
    for i in range(d):
        a[i][int(y[i])] = 1.0
    return a


class KNN():
    def __init__(self, data, k=5):
        '''获取KNN所需相关参数'''
        self.k = k
        self.x, self.y = data[:]
        self.n = len(self.y)

    def predict(self, x):
        '''函数功能：通过比较待分类数据和已知数据之间欧氏距离的平方，获取距离最小的前K个样本标签，使用投票法给出待分类样本的类别'''
        # 计算待测试数据与已知数据之间的欧氏距离的平方，并将其与已知数据的标签拼成一个2列的数组
        dis = np.vstack([np.sum((self.x-x)**2, 1), self.y])
        # 以距离的平方为依据对其进行排序 并选择前k个样本
        nearest = np.array(sorted(dis.T, key=lambda x: x[0]))[:self.k, 1]
        # 索引值必须为int类型整数
        nearest = nearest.astype(np.int32)
        # 返回数量最多的索引
        return np.argmax(np.bincount(nearest))


class MLP():
    def __init__(self, in_channel, out_channel, lr=0.01, hidden_cells=(48,)):
        '''
        in_channel:输入维度
        out_channel:输出维度
        lr:学习率
        hidden_layers:隐藏层数量
        hidden_cells:隐藏层各层神经元数量
        '''
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.lr = lr
        self.hidden_layers = len(hidden_cells)
        self.hidden_cells = [self.in_channel]+list(hidden_cells)
        self.parameter_init()

    def parameter_init(self):
        '''参数初始化'''
        # 定义乘子s用于w的初始化，因为当w值较大时容易出现溢出问题
        s = 1e-5
        # 各层权重
        self.w = []
        # 各层偏置
        self.b = []
        # 隐藏层参数初始化
        for i in range(self.hidden_layers):
            self.w.append(s*np.random.rand(
                self.hidden_cells[i], self.hidden_cells[i+1]))
            self.b.append(np.zeros((1, self.hidden_cells[i+1])))
        # 输出层参数初始化
        self.w.append(
            s*np.random.rand(self.hidden_cells[-1], self.out_channel))
        self.b.append(np.zeros((1, self.out_channel)))
        self.gradient_init()

    def gradient_init(self):
        '''梯度初始化'''
        self.z = [None]*len(self.hidden_cells)
        self.a = [None]*len(self.hidden_cells)
        self.dz = [None]*len(self.hidden_cells)
        self.dw = [None]*len(self.w)
        self.db = [None]*len(self.b)

    def sigmoid(self, x):
        '''sigmoid激活函数'''
        return 1.0/(1.0+np.exp(-x))

    def sigmoid_derivative(self, a):
        '''sigmoid函数的导数'''
        return a*(1-a)

    def softmax(self, y):
        '''softmax激活函数'''
        # 每个输出减去其中的最大值，防止exp函数出现上溢
        c = np.max(y, 1, keepdims=True)
        p = np.exp(y-c)
        return p/np.sum(p, 1, keepdims=True)

    def softmax_derivative(self, y, label):
        '''交叉熵loss对输出层线性神经元z的导数'''
        # loss对z的导数为预测值真实类别索引处数据-1，其中label为onehot向量
        d = y-label
        return d

    def mutil_cross_entropy(self, y, label):
        '''多分类交叉熵损失'''
        # 避免因为y中含有0导致log函数出现错误
        y[y == 0] = 0.001
        return -np.sum(label*np.log2(y), 1)

    def update(self, x, label):
        '''
        反向传播及参数更新:
        x:神经元输入
        label:数据标签(onehot向量)
        '''
        # 输出层求导
        self.dz[-1] = self.softmax_derivative(self.a[-1], label)
        # 输出层w梯度
        self.dw[-1] = self.a[-2].T@self.dz[-1]
        # 输出层b梯度
        self.db[-1] = np.sum(self.dz[-1], 0, keepdims=True)
        # 隐藏层求导
        for i in range(len(self.z)-2, 0, -1):
            self.dz[i] = self.dz[i+1]@self.w[i+1].T * \
                self.sigmoid_derivative(self.a[i])
            # 第i层隐藏层w梯度
            self.dw[i] = self.a[i-1].T@self.dz[i]
            # 第i层隐藏层b梯度
            self.db[i] = np.sum(self.dz[i], 0, keepdims=True)
        # 输入层求导
        self.dz[0] = self.dz[1]@self.w[1].T*self.sigmoid_derivative(self.a[0])
        # 输入层w梯度
        self.dw[0] = x.T@self.dz[0]
        # 输入层b梯度
        self.db[0] = np.sum(self.dz[0], 0, keepdims=True)
        # 参数更新
        for i in range(len(self.w)):
            self.w[i] = self.w[i] - self.lr*self.dw[i]
            self.b[i] = self.b[i] - self.lr*self.db[i]

    def forward(self, x):
        '''
        前向传播过程:
        x: 神经元输入
        '''
        self.a[-1] = x
        # 输入层以及隐藏层前向传播
        for i in range(self.hidden_layers):
            # 线性神经元输出计算
            self.z[i] = np.dot(self.a[i-1], self.w[i])+self.b[i]
            # 非线性神经元输出计算(激活函数sigmoid)
            self.a[i] = self.sigmoid(self.z[i])
        # 输出层输出计算
        self.z[-1] = np.dot(self.a[-2], self.w[-1])+self.b[-1]
        # 使用softmax计算预测概率
        self.a[-1] = self.softmax(self.z[-1])

        return self.a[-1]

    def train(self, train_x, train_y):
        '''
        训练过程:
        train_x:用于训练的数据输入
        train_y:train_x对应标签(onehot向量)
        '''
        # 前向传播
        out = self.forward(train_x)
        # 计算误差
        loss = self.mutil_cross_entropy(out, train_y)
        # 误差反向传播以及参数更新
        self.update(train_x, train_y)
        return loss

    def test(self, test_x, test_y):
        '''
        测试过程:
        test_x:用于训练的数据输入
        test_y:test_x对应标签(onehot向量)
        '''
        # 计算分类结果
        out = self.forward(test_x)
        # 判断是否分类正确
        if np.argmax(out) == int(test_y):
            return True
        return False


class SVM():
    '''软间隔SVM 核函数为高斯核函数'''

    def __init__(self, C, s=0.5, e=0.01, max_iter=1000):
        '''
        C:惩罚项
        s:高斯核函数方差
        e:结束迭代的误差阈值
        max_iter:最大迭代次数
        '''
        self.gama = s
        self.C = C
        self.e = e
        self.nMaxIter = max_iter
        # bias为偏置 初始设置为0
        self.bias = 0

    def gaussian_kernel(self, x1, x2):
        '''高斯核函数'''
        return np.exp(-np.linalg.norm(x1-x2)**2/(2*self.gama**2))

    def train(self, x, y):
        '''
        SVM训练过程 
        x:训练数据特征向量 
        y:训练数据标签 -1或+1
        '''
        self.x = x.reshape(x.shape[0], x.shape[1], 1)
        self.y = y.reshape(-1, 1)
        self.n, self.d, _ = self.x.shape
        # 初始化a全为0
        self.a = np.zeros((self.n, 1))
        for i in range(self.nMaxIter):
            # 获取要更新的两个参数
            a1, a2 = self.get_a1()
            # 通过SMO算法对齐进行更新并返回误差
            e = self.SMO(a1, a2)
            if e < self.e:
                print(e)
                return
        print(e)

    def dec_surf(self, x):
        '''SVM的决策面 计算样本x到决策面的距离'''
        ans = 0
        for i in range(self.n):
            ans += self.a[i] * self.y[i] * self.gaussian_kernel(self.x[i], x)
        return ans + self.bias

    def SMO(self, a1_idx, a2_idx):
        '''
        SMO参数更新算法
        a1_idx:需要更新的第一个参数的索引
        a2_idx:需要更新的第二个参数的索引
        '''
        # 保存更新前的参数
        a0 = self.a.copy()
        # 获取两个参数对应的样本及其标签
        x1, x2 = self.x[a1_idx], self.x[a2_idx]
        y1, y2 = self.y[a1_idx], self.y[a2_idx]
        # 计算更新后a2的值 但由于a2的取值受到a1以及C的限制，需要进一步处理
        et = self.gaussian_kernel(x1, x1)+self.gaussian_kernel(x2,
                                                               x2)-2*self.gaussian_kernel(x1, x2)
        temp_a2 = a0[a2_idx]+y2*(self.loss(a1_idx)-self.loss(a2_idx))/et
        # 计算更新后的a2是否在限制范围内 若超出范围则进行截取
        if y1.all() == y2.all():
            self.a[a2_idx] = temp if temp_a2 > (
                temp := min(self.C, a0[a1_idx]+a0[a2_idx])) else temp_a2
            self.a[a2_idx] = temp if temp_a2 < (temp := max(
                0, a0[a1_idx]+a0[a2_idx]-self.C)) else temp_a2
        else:
            self.a[a2_idx] = temp if temp_a2 > (
                temp := min(self.C, a0[a2_idx]-a0[a1_idx]+self.C)) else temp_a2
            self.a[a2_idx] = temp if temp_a2 < (temp := max(
                0, a0[a2_idx]-a0[a1_idx])) else temp_a2
        # 计算更新后的a1的值
        self.a[a1_idx] = a0[a1_idx]+y1*y2*(a0[a2_idx]-self.a[a2_idx])
        # 计算更新后的偏置
        b1 = self.bias-self.loss(a1_idx)-y1*self.gaussian_kernel(x1, x1)*(
            self.a[a1_idx]-a0[a1_idx])-y2*self.gaussian_kernel(x1, x2)*(self.a[a2_idx]-a0[a2_idx])
        b2 = self.bias-self.loss(a2_idx)-y1*self.gaussian_kernel(x1, x2)*(
            self.a[a1_idx]-a0[a1_idx])-y2*self.gaussian_kernel(x2, x2)*(self.a[a2_idx]-a0[a2_idx])
        self.bias = (b1+b2)/2
        # 返回更新的误差 为参数a更新前后差值的二范数
        return np.linalg.norm(a0-self.a)

    def get_a1(self):
        '''
        选择SMO算法要更新的第一个参数
        选择原则:优先选择处于(0,C)之间的参数，其次为{0,C}中的参数
        '''
        # 优先选择处于(0,C)之间的参数
        for i in range(self.n):
            if 0 < self.a[i] < self.C:
                if self.dec_surf(self.x[i])*self.y[i] != 1:
                    a2 = self.get_a2(i)
                    return i, a2
        for i in range(self.n):
            if self.a[i] == 0:
                if self.dec_surf(self.x[i])*self.y[i] < 1:
                    a2 = self.get_a2(i)
                    return i, a2
            elif self.a[i] == self.C:
                if self.dec_surf(self.x[i])*self.y[i] > 1:
                    a2 = self.get_a2(i)
                    return i, a2

    def get_a2(self, a1_idx):
        '''
        选择SMO算法中要更新的第二个参数
        a1_idx:已确定的要更新的第一个参数
        选择原则:在已选择第一个参数的基础上，选择的a2应当能够使||E1-E2||能够有最大的变化
        '''
        # 记录最大loss及其对应a2的索引
        maxloss = 0
        a2_idx = 0
        for i in range(self.n):
            loss = np.abs(self.loss(a1_idx)-self.loss(i))
            if loss <= maxloss:
                continue
            a2_idx = i
            maxloss = loss
        return a2_idx

    def loss(self, idx):
        '''
        计算样本与标签之间的误差
        idx:样本索引
        '''
        return self.dec_surf(self.x[idx])-self.y[idx]

    def predict(self, x):
        '''
        预测样本的类别
        x:待分类样本
        '''
        if self.dec_surf(x) > 0:
            return np.array([1])
        else:
            return np.array([-1])

    def score(self, x_data, y_data):
        '''
        计算分类准确率
        x_data:待分类数据集输入
        y_data:待分类数据集标签
        '''
        cnt = 0
        for i in range(self.n):
            if y_data[i] == self.predict(x_data[i]):
                cnt += 1
        return cnt/self.n


class OVRSVM():
    '''采用OVR策略进行多分类的SVM分类器'''

    def __init__(self, C, s, max_iter, e, types=10):
        '''
        参数说明：
        C:每一个SVM的惩罚项
        s:高斯核函数方差
        e:结束迭代的误差阈值
        max_iter:最大迭代次数
        types:输出类别总数
        '''
        self.types = types
        # 根据类别数量n初始化n个SVM分类器
        self.svms: List[SVM] = []
        for i in range(self.types):
            self.svms.append(SVM(C, s, e, max_iter))

    def svm_train(self, data):
        '''多分类的SVM训练过程'''
        for i in range(self.types):
            train_x, train_y = self.ovr(data, i)
            print(i, end=' ')
            self.svms[i].train(train_x, train_y)

    def score(self, data):
        '''
        计算多分类正确率
        data:数据集
        '''
        output = []
        cnt = 0
        n = len(data)
        # 获取10个分类器的分类结果
        for i in range(self.types):
            output.append([])
            test_x, test_y = self.ovr(data, i)
            for j in range(n):
                output[i].append(self.svms[i].predict(test_x[j]))
        labels = data.y
        output = np.array(output)
        # 10个分类器分类结果相加，以判断正类数量
        sum_out = np.sum(output, 0)
        for i in range(n):
            y = int(labels[i])
            # 当只有一个分类器将其分为正类时 样本类别即为此类
            if output[y][i] == 1 and sum_out[i] == -8:
                cnt += 1
            # 当有多个分类器将其分为正类时，根据样本到决策面的距离确定
            else:
                max_dis = 0
                idx = -1
                for j in range(self.types):
                    dis = self.svms[j].dec_surf(data.x[i])
                    if dis > max_dis:
                        max_dis = dis
                        idx = j
                if idx == y:
                    cnt += 1
        return cnt/n

    def ovr(self, data, positive):
        '''
        对多类数据集标签进行处理
        选择一个类别作为正类 其标签替换为1 其余类别作为负类 其标签替换为-1
        data:包含多个类别的数据集
        positive:正类样本索引
        '''
        out_x = []
        out_y = []
        for i in range(len(data)):
            x, y = data[i]
            # 目标类别作为正类 其他类别作为负类
            if y == positive:
                y = 1.0
            else:
                y = -1.0
            out_x.append(x)
            out_y.append(y)
        out_x = np.array(out_x)
        out_y = np.array(out_y)
        out_x = out_x.reshape(out_x.shape[0], out_x.shape[1], 1)
        out_y = out_y.reshape(out_y.shape[0], 1)
        return out_x, out_y
