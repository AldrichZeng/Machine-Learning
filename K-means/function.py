from numpy import *


def loadDataSet(fileName):
    # 导入数据
    dataMat = []
    fileRead = open(fileName)  # filename=“testSet.txt”
    for line in fileRead.readlines():
        curLine = line.strip().split("\t")  # 每一行为tab分割的float。strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
        fltLine = list(map(float, curLine)) # 将curLine转化为float类型
        # 这里的map是将curLine（list）中的所有元素转换为float类型，并不是要返回一个map，而是强制类型转换。
        dataMat.append(fltLine)
    return dataMat  # type=list


def distEclud(a, b):
    # 计算两个向量的欧拉距离
    return sqrt(sum(power(a - b, 2)))

def randCent(dataSet, k):
    """
    随机生成质心
    :param dataSet: 数据集，type=matrix
    :param k: k个质心=k个簇, type=int
    :return:
    """
    n = shape(dataSet)[1]  # 每个样本的维数
    centroids = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])  # 找到每一维的最小值
        maxJ = max(dataSet[:, j])  # 找到每一维的最大值
        rangeJ = float(maxJ - minJ)  # 范围
        # fltLine = map(float, curLine)在python2中返回的是一个list类型数据，而在python3中该语句返回的是一个map类型的数据。
        # 因此，我们只需要将该语句改为fltLine = list(map(float, curLine)), 错误就解决啦
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
    return centroids

datMat=mat(loadDataSet('testSet.txt'))
centroids=randCent(datMat,2)
print(centroids)

