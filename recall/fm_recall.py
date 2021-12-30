import numpy
import numpy as np
import random
import pandas as pd
from numpy import *
from random import normalvariate  # 正态分布
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def loadData(base_dir):
    # '''
    #    load ori_data
    #    Returns: csv_data
    #    '''
    # movie_columns = ['movie_id', 'tag_id', 'relevance']
    # movies = pd.read_csv(base_dir + "genome-scores.csv", sep=',', header=None, names=movie_columns, engine='python')
    #
    # rating_columns = ['user_id', 'movie_id', 'rating', 'timestamp1']
    # ratings = pd.read_csv(base_dir + "ratings_shuf.csv", sep=',', header=None, names=rating_columns, engine='python')
    #
    # user_columns = ['user_id', 'movie_id', 'tag', 'timestamp2']
    # users = pd.read_csv(base_dir + "tags.csv", sep=',', header=None, names=user_columns, engine='python')
    #
    # tag_columns = ['tag_id', 'tag']
    # tags = pd.read_csv(base_dir + "genome-tags.csv", sep=',', header=None, names=tag_columns, engine='python')
    #
    # deal_columns = ['movie_id', 'name', 'year', 'genres', 'totalQuantity', 'averageRatings']
    # deal = pd.read_csv(base_dir + "deal.csv", sep=',', header=None, names=deal_columns, engine='python')
    #
    # userProfile_columns = ['user_id', 'quantity', 'avgGivingRating']
    # userProfile = pd.read_csv(base_dir + "userProfile.csv", sep=',', header=None, names=userProfile_columns,
    #                           engine='python')
    # print(movies)
    # print(ratings)
    # print(users)
    # data = pd.merge(movies, users)
    # data = pd.merge(data, tags)
    #
    # print(data)
    # data = pd.merge(ratings, data)
    # print(data)
    # data = pd.merge(data, deal)
    # data = pd.merge(data, userProfile)
    #
    # data.to_csv("load_data.csv")

    user_columns = ['id','user_id', 'movie_id', 'rating', 'timestamp1', 'tag_id', 'relevance', 'tag', 'timestamp2', 'name',
                    'year', 'genres', 'totalQuantity', 'averageRatings', 'quantity', 'avgGivingRating'
                    ]
    data = pd.read_csv("load_data.csv", sep=',', header=None, names=user_columns, engine='python')
    data.reset_index()
    return data


# 归一化
def data_norm(df, *cols):
    df_n = df.copy()
    for col in cols:
        ma = df[col].max()
        mi = df[col].min()
        df_n[col] = (df[col] - mi) / (ma - mi)
    return (df_n)


# 处理数据
def preprocessData(data, ratio):
    label = data['rating'].map(lambda x: 1 if float(x) > 4 else -1)
    print(label)

    features = ['quantity', 'year', 'avgGivingRating', 'totalQuantity', 'averageRatings', 'timestamp1', 'timestamp2',
                'tag']
    data = data[features]
    data = data_norm(data, 'quantity', 'year', 'avgGivingRating', 'totalQuantity', 'averageRatings', 'timestamp1',
                     'timestamp2')
    # onehot
    data = pd.get_dummies(data, sparse=True)
    data.to_csv("new.csv")

    # data = pd.read_csv("new.csv", sep=',', engine='python',index_col=0)
    print(data)

    num = int(data.shape[0] * ratio)

    train = data[:num]
    train_label = label[:num]

    test = data[num:]
    test_label = label[num:]
    test_label.reset_index()
    return train, train_label, test, test_label


def sigmoid(inX):
    if inX >= 0:

        return 1.0 / (1 + exp(-inX))
    else:
        return exp(inX) / (1 + exp(inX))


# 训练FM模型
def FM(dataMatrix, classLabels, k, iter, alpha):
    '''
       :param dataMatrix:  特征矩阵
       :param classLabels: 标签矩阵
       :param k:           v的维数
       :param iter:        迭代次数
       :return:            常数项w_0, 一阶特征系数w, 二阶交叉特征系数v
       '''
    # dataMatrix用的是matrix, classLabels是列表
    m, n = shape(dataMatrix)  # 矩阵的行列数，即样本数m和特征数n

    # 初始化参数
    w = zeros((n, 1))  # 一阶特征的系数
    w_0 = 0  # 常数项
    v = normalvariate(0, 0.2) * ones((n, k))  # 即生成辅助向量(n*k)，用来训练二阶交叉特征的系数
    print(v)
    for it in range(iter):

        for x in range(m):  # 随机优化，每次只使用一个样本

            # 二阶项的计算
            inter_1 = dataMatrix[x] * v  # 每个样本(1*n)x(n*k),得到k维向量（FM化简公式大括号内的第一项）
            inter_2 = multiply(dataMatrix[x], dataMatrix[x]) * multiply(v, v)  # 二阶交叉项计算，得到k维向量（FM化简公式大括号内的第二项）
            interaction = sum(multiply(inter_1, inter_1) - inter_2) / 2.  # 二阶交叉项计算完成（FM化简公式的大括号外累加）

            p = w_0 + dataMatrix[x] * w + interaction  # 计算预测的输出，即FM的全部项之和

            tmp = 1 - sigmoid(classLabels[x] * p[0, 0])  # tmp迭代公式的中间变量，便于计算
            # if np.isnan(tmp):
            #     tmp = 0

            w_0 = w_0 + alpha * tmp * classLabels[x]

            for i in range(n):
                if dataMatrix[x, i] != 0:
                    w[i, 0] = w[i, 0] + alpha * tmp * classLabels[x] * dataMatrix[x, i]
                    # if np.isnan(w[i, 0]):
                    #     w[i, 0] = 0
                for j in range(k):
                    v[i, j] = v[i, j] + alpha * tmp * classLabels[x] * (
                            dataMatrix[x, i] * inter_1[0, j] - v[i, j] * dataMatrix[x, i] * dataMatrix[x, i])

            # 计算损失函数的值
            if x%1000==0:
                print(x)
        loss = getLoss(getPrediction(mat(dataMatrix), w_0, w, v), classLabels)
        print("第{}次迭代后的损失为{}".format(it, loss))

    return w_0, w, v


# 损失函数
def getLoss(predict, classLabels):
    m = len(predict)
    loss = 0.0

    for i in range(m):
        loss -= log(sigmoid(predict[i] * classLabels[i]))
    return loss


# 预测
def getPrediction(dataMatrix, w_0, w, v):
    m = np.shape(dataMatrix)[0]
    result = []

    for x in range(m):
        inter_1 = dataMatrix[x] * v
        inter_2 = multiply(dataMatrix[x], dataMatrix[x]) * multiply(v, v)  # multiply对应元素相乘
        # 完成交叉项
        interaction = np.sum(multiply(inter_1, inter_1) - inter_2) / 2.

        p = w_0 + dataMatrix[x] * w + interaction  # 计算预测的输出

        pre = sigmoid(p[0, 0])
        result.append(pre)
    return result


# 评估预测的准确性
def getAccuracy(predict, classLabels):
    m = len(predict)
    allItem = 0
    error = 0

    for i in range(m):  # 计算每一个样本的误差

        allItem += 1
        if float(predict[i]) < 0.5 and classLabels[i] == 1.0:
            error += 1
        elif float(predict[i]) >= 0.5 and classLabels[i] == -1.0:
            error += 1
        else:
            continue

    return float(error) / allItem


if __name__ == '__main__':
    data_file = "/Users/kevin/Desktop/ml-1m/"
    Data = loadData(data_file)
    dataTrain, labelTrain, dataTest, labelTest = preprocessData(Data, 0.8)
    date_startTrain = datetime.now()
    print(dataTrain)
    print("开始训练")

    w_0, w, v = FM(mat(dataTrain.values), labelTrain, 0 ,50, 0.001)
    print("w_0:", w_0)
    print("w:", w)
    print("v:", v)
    predict_train_result = getPrediction(mat(dataTrain.values), w_0, w, v)  # 得到训练的准确性
    print("训练准确性为：%f" % (1 - getAccuracy(predict_train_result, labelTrain)))
    date_endTrain = datetime.now()
    print("训练用时为：%s" % (date_endTrain - date_startTrain))
    print("labelTrain")
    print(labelTrain)
    print('labelTest')
    print(labelTest)
    print("开始测试")
    dataTest.reset_index(drop=True, inplace=True)
    labelTest.reset_index(drop=True, inplace=True)
    print(dataTest)
    predict_test_result = getPrediction(mat(dataTest.values), w_0, w, v)  # 得到训练的准确性
    print(predict_train_result)
    print(predict_test_result)
    print("测试准确性为：%f" % (1 - getAccuracy(predict_test_result, labelTest)))
