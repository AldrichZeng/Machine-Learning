import numpy as np
import matplotlib.pyplot as plt


def loadSimpData():
    """
    创建单层决策树的数据集
    Parameters:
        无
    Returns:
        dataMat - 数据矩阵
        classLabels - 数据标签
    """
    datMat = np.matrix([[1., 2.1],
                        [1.5, 1.6],
                        [1.3, 1.],
                        [1., 1.],
                        [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels


def showDataSet(dataMat, labelMat):
    """
    数据可视化
    Parameters:
        dataMat - 数据矩阵
        labelMat - 数据标签
    Returns:
        无
    """
    data_plus = []  # 正样本
    data_minus = []  # 负样本
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)  # 转换为numpy矩阵
    data_minus_np = np.array(data_minus)  # 转换为numpy矩阵
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1])  # 正样本散点图
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1])  # 负样本散点图
    plt.show()


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    """
    单层决策树在第dimen个特征上的分类
    Parameters:
        dataMatrix - matrix，数据矩阵
        dimen - int，第dimen列，也就是第几个特征
        threshVal - float 阈值
        threshIneq - string，标志
    Returns:
        retArray - m*1的matrix  分类结果
    """
    retuanArray = np.ones((np.shape(dataMatrix)[0], 1))  # 初始化retArray为1,shape为(样本数,1)
    if threshIneq == 'lt':
        retuanArray[dataMatrix[:, dimen] <= threshVal] = -1.0  # 如果小于阈值,则赋值为-1（广播机制）
    else:
        retuanArray[dataMatrix[:, dimen] > threshVal] = -1.0  # 如果大于阈值,则赋值为-1（广播机制）
    return retuanArray


def buildStump(dataArr, classLabels, D):
    """
    找到数据集上最佳的单层决策树
    Parameters:
        dataArr - 数据矩阵
        classLabels - 数据标签
        D - 样本权重 shape=(m,1)
    Returns:
        bestStump - 字典，最佳单层决策树信息（dim，mthresh，ineq）
        minError - float，最小误差
        bestClasEst - matrix of shape =(m,1)  最佳单层决策树的分类结果
    """
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)  # m是样本数
    numSteps = 10.0
    bestStump = {}
    bestClasEst = np.mat(np.zeros((m, 1)))
    minError = float('inf')  # 最小误差初始化为正无穷大
    for i in range(n):  # 遍历所有特征
        # 找到特征中最小的值和最大值
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps  # 计算步长
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:  # 遍历大于和小于的情况，lt:less than，gt:greater than
                threshVal = (rangeMin + float(j) * stepSize)  # 计算阈值
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)  # 计算分类结果
                errArr = np.mat(np.ones((m, 1)))  # 初始化误差矩阵
                errArr[predictedVals == labelMat] = 0  # 分类正确的,赋值为0
                weightedError = D.T * errArr  # 计算误差
                # print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
                if weightedError < minError:  # 找到误差最小的分类方式
                    minError = weightedError  # 更新minError
                    bestClasEst = predictedVals.copy()
                    '''存储最佳在单层决策树'''
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    """
    使用AdaBoost算法提升弱分类器性能
    Parameters:
        dataArr - 数据矩阵 m*2
        classLabels - 数据标签 m*1
        numIt - int， 最大迭代次数
    Returns:
        weakClassArr - List,弱分类器数组
        aggClassEst - 类别估计累计值
    """
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m, 1)) / m)  # 初始化权重
    aggClassEst = np.mat(np.zeros((m, 1)))

    for i in range(numIt):  # 控制迭代次数

        '''构建单层决策树'''
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print("D:", D.T)  # 上面的函数并没有改变D的值

        '''计算弱学习算法权重alpha；max(error, 1e-16)是为了防止除零溢出'''
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))

        bestStump['alpha'] = alpha  # 存储弱学习算法权重
        weakClassArr.append(bestStump)  # 存储单层决策树到List中
        print("classEst: ", classEst.T)

        '''为下一次迭代计算D'''
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()
        '''错误率累加计算'''
        aggClassEst += alpha * classEst
        print("aggC"
              "lassEst: ", aggClassEst)

        '''计算误差'''
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
        # np.sign：大于0返回1,小于0返回-1,等于0返回0
        errorRate = aggErrors.sum() / m
        print("total error: ", errorRate)

        if errorRate == 0.0:
            break  # 误差为0，退出循环

    return weakClassArr, aggClassEst


def adaClassify(datToClass, classifierArr):
    """
    AdaBoost分类函数
    Parameters:
        datToClass - 待分类样例
        classifierArr - 训练好的分类器
    Returns:
        分类结果
    """
    dataMatrix = np.mat(datToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    '''遍历所有分类器，进行分类'''
    for i in range(len(classifierArr)):
        '''得到class的估计值'''
        classEst = stumpClassify(dataMatrix=dataMatrix,
                                 dimen=classifierArr[i]['dim'],
                                 threshVal=classifierArr[i]['thresh'],
                                 threshIneq=classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print(aggClassEst)
    return np.sign(aggClassEst)  # 返回符号


if __name__ == '__main__':
    # a, b = loadSimpData()
    # D = np.mat(np.ones((5, 1)) / 5)
    # bestStump, minError, bestClasEst = buildStump(a, b, D)
    # print(bestStump)
    # print(minError)
    # print(bestClasEst)

    dataArr, classLabels = loadSimpData()
    weakClassArr, aggClassEst = adaBoostTrainDS(dataArr, classLabels)
    print("分类函数")
    print(adaClassify([[0,0],[5,5]], weakClassArr))
