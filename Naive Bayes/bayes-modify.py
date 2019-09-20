import numpy as np
import random
import re


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
    return returnVec  # 返回文档向量										#返回文档向量


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
    return returnVec  # 返回文档向量													#返回词袋模型


def trainNB0(trainMatrix, trainCategory):
    """
    朴素贝叶斯分类器训练函数
Parameters:
	trainMatrix - 文档向量的集合，元素为setOfWords2Vec返回的returnVec
	trainCategory - label，每个文档向量对应的class（1或0）
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
    p1Vect = np.log(p1Num / p1Denom)  # 计算p(wi|c=1),防止下溢取对数
    p0Vect = np.log(p0Num / p0Denom)  # 计算p(wi|c=0),防止下溢取对数
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
    p0 = sum(testVec * p0Vec) + np.log(1.0 - pClass1)
    print('p0:', p0)
    print('p1:', p1)
    if p1 > p0:
        return 1
    else:
        return 0


def textParse(bigString):
    """
    接收一个大字符串并将其解析为字符串列表
    :param bigString: 大字符串
    :return: 字符串列表
    """
    # * 会匹配0个或多个规则，split会将字符串分割成单个字符【python3.5+】; 这里使用\W 或者\W+ 都可以将字符数字串分割开，产生的空字符将会在后面的列表推导式中过滤掉
    listOfTokens = re.split(r'\W+', bigString)  # 将特殊符号作为切分标志进行字符串切分，即非字母、非数字
    stringList = [tok.lower() for tok in listOfTokens if len(tok) > 2]
    return stringList  # 转化为小写，并去掉长度小于3的单词


def spamTest():
    """
    测试朴素贝叶斯分类器
    """
    docList = [];  # 保存每个邮件，元素类型String
    classList = [];  # 保存标签，元素类型int
    for i in range(1, 26):  # 遍历25个txt文件
        wordList = textParse(open('./email/spam/%d.txt' % i, 'r').read())  # 读取 垃圾邮件，并转换成字符串列表
        docList.append(wordList)
        classList.append(1)  # 标记垃圾邮件，class=1

        wordList = textParse(open('./email/ham/%d.txt' % i, 'r').read())  # 读取 非垃圾邮件，并转换成字符串列表
        docList.append(wordList)
        classList.append(0)  # 标记非垃圾邮件，class=0
    # 创建词汇表
    vocabList = createVocabList(docList)
    # 构造数据集
    trainingSet = list(range(50));  # 整数列表，训练集的索引值的列表0-49
    testSet = []  # 测试集的索引值列表
    for i in range(10):  # 从50个邮件中，随机挑选出40个作为训练集,10个做测试集
        randIndex = int(random.uniform(0, len(trainingSet)))  # 随机选取索索引值
        testSet.append(trainingSet[randIndex])  # 添加测试集的索引值
        del (trainingSet[randIndex])  # 在trainingSet中删除添加到测试集的索引值
    # 创建训练集和训练集
    trainMat = [];  # 训练集（文档向量的集合）
    trainClasses = []  # 测试集
    for docIndex in trainingSet:  # 遍历训练集
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))  # 将文档向量添加到训练集中
        trainClasses.append(classList[docIndex])  # 将label添加到训练集中
    # 训练朴素贝叶斯模型
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
    # 测试模型
    errorCount = 0  # 错误分类计数
    for docIndex in testSet:  # 遍历测试集
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])  # 测试样本（文档向量）
        result = classifyNB(np.array(wordVector), p0V, p1V, pSpam)  # 分类结果
        if result != classList[docIndex]:  # 如果分类错误
            errorCount += 1  # 错误计数加1
            print("分类错误的测试样本：", docList[docIndex])
    # 输出错误率（因为是随机选择的样本，所以每次输出结果不一样）
    print('错误率：%.2f%%' % (float(errorCount) / len(testSet) * 100))


if __name__ == '__main__':
    spamTest()
