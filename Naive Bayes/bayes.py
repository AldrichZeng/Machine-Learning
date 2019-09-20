import numpy as np
from functools import reduce


def loadDataSet():
    """
    创建实验样本
    Parameters:
    	无
    Returns:
    	postingList - 实验样本切分的词条
    	classVec - 类别标签向量
    """
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],  # 词条集合
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 类别标签向量，1代表侮辱性词汇，0代表正常言论
    return postingList, classVec  # 返回实验样本切分的词条和类别标签向量


def createVocabList(dataSet):
    """
    将切分的实验样本词条整理成不重复的词条列表，也就是词汇表
    :param dataSet: 整理的样本数据集
    :return: 返回不重复的词条列表，也就是词汇表
    """
    vocabSet = set([])  # 创建一个空的不重复列表（词汇表）
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 取并集
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    """
    根据词汇表，将文档向量化，向量的每个元素为1或0
    Parameters:
    	vocabList - createVocabList返回的列表，词汇表
    	inputSet - 切分的词条列表，一个文档
    Returns:
    	returnVec - 文档向量,词集模型
    """
    returnVec = [0] * len(vocabList)  # 创建一个其中所含元素都为0的向量,和词汇表等长
    for word in inputSet:  # 遍历每个词条
        if word in vocabList:  # 如果词条存在于词汇表中，则置1
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec  # 返回文档向量


def bagOfWords2VecMN(vocabList, inputSet):
    """
    根据词汇表，将文档向量化，向量的每个元素为该词出现的次数
    Parameters:
    	vocabList - createVocabList返回的列表，词汇表
    	inputSet - 切分的词条列表，一个文档
    Returns:
    	returnVec - 文档向量,词集模型
    """
    returnVec = [0] * len(vocabList)  # 创建一个其中所含元素都为0的向量,和词汇表等长
    for word in inputSet:  # 遍历每个词条
        if word in vocabList:  # 如果词条存在于词汇表中，则置1
            returnVec[vocabList.index(word)] += 1  # 与setOfWords2Vec()唯一不同的地方
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec  # 返回文档向量


def trainNB0(trainMatrix, trainCategory):
    """
    朴素贝叶斯分类器训练函数
Parameters:
	trainMatrix - 文档向量的集合，元素为setOfWords2Vec返回的returnVec
	trainCategory - 标签，每个文档向量对应的class（1或0）
Returns:
	p0Vect - class=0的情况下，各词条出现的概率（条件概率p(x|c=0)）
	p1Vect - class=1的情况下，各词条出现的概率（条件概率p(x|c=1)）
	pAbusive - 文档属于class=1的概率
    """
    numTrainDocs = len(trainMatrix)  # 训练集的文档数目
    numWords = len(trainMatrix[0])  # 每个文档向量的长度=词汇表长度
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 文档属于class=1的概率
    p0Num = np.ones(numWords);  # class=0的各词条出现的次数
    p1Num = np.ones(numWords)  # class=1的各词条出现的次数
    p0Denom = 2.0;  # class=0的词条总数
    p1Denom = 2.0  # class=1的词条总数
    for i in range(numTrainDocs):  # 遍历所有文档
        if trainCategory[i] == 1:  # class=1的样本
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])  # class=1的词条总数
        else:  # class=0的样本
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])  # class=0的词条总数
    p1Vect = np.log(p1Num / p1Denom)  # 计算p(xi|c=1),防止下溢取对数（xi为词条）
    p0Vect = np.log(p0Num / p0Denom)  # 计算p(xi|c=0),防止下溢取对数（xi为词条）
    return p0Vect, p1Vect, pAbusive


def classifyNB(testVec, p0Vec, p1Vec, pClass1):
    """
    朴素贝叶斯分类器 分类函数
    :param testVec: 待分类的词条数组(测试样本)
    :param p0Vec: class=0的条件概率数组
    :param p1Vec: class=1的条件概率数组
    :param pClass1: 文档属于class=1的概率
    :return:0 - 属于非侮辱类
	        1 - 属于侮辱类
    """
    p1 = sum(testVec * p1Vec) + np.log(pClass1)  # 对应元素相乘  这里需要好好理解一下
    p0 = sum(testVec * p0Vec) +np.log(1.0 - pClass1)
    print('p0:', p0)
    print('p1:', p1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    """
    测试朴素贝叶斯分类器
    """
    listOPosts, listClasses = loadDataSet()  # 创建实验样本
    myVocabList = createVocabList(listOPosts)  # 创建词汇表
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))  # 将实验样本向量化
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))  # 训练朴素贝叶斯分类器

    testEntry = ['love', 'my', 'dalmation']  # 测试样本1
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))  # 测试样本向量化
    if classifyNB(thisDoc, p0V, p1V, pAb):
        print(testEntry, '属于侮辱类')  # 执行分类并打印分类结果
    else:
        print(testEntry, '属于非侮辱类')  # 执行分类并打印分类结果

    testEntry = ['stupid', 'garbage']  # 测试样本2
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))  # 测试样本向量化
    if classifyNB(thisDoc, p0V, p1V, pAb):
        print(testEntry, '属于侮辱类')  # 执行分类并打印分类结果
    else:
        print(testEntry, '属于非侮辱类')  # 执行分类并打印分类结果


if __name__ == '__main__':
    testingNB()