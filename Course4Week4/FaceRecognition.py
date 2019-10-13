from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

K.set_image_data_format('channels_first')
import time
import cv2
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
import fr_utils
from inception_blocks_v2 import *


def triplet_loss(y_true, y_pred, alpha=0.2):
    """
    计算三元组损失函数
    :param y_true: 标签
    :param y_pred: List
    :param alpha: 阈值
    :return: 实数，损失函数的值
    """
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]  # 获取anchor，positive，negative的编码
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)  # anchor与positive之间的编码距离
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)  # anchor与negative之间的编码距离
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)  # 两个距离相减，然后加上alpha
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0))  # 返回0和basic_loss的最大值
    return loss
with tf.Session() as test:
    tf.set_random_seed(1)
    y_true = (None, None, None)
    y_pred = (tf.random_normal([3, 128], mean=6, stddev=0.1, seed = 1),
              tf.random_normal([3, 128], mean=1, stddev=1, seed = 1),
              tf.random_normal([3, 128], mean=3, stddev=4, seed = 1))
    loss = triplet_loss(y_true, y_pred)

    print("loss = " + str(loss.eval()))
def verify(image_path,identity,database,model):
    encoding=fr_utils.img_to_encoding(image_path,model)  # 计算图像的编码
    dist=np.linalg.norm(encoding-database[identity])*10  # 计算编码距离
    if dist<0.7:  # 判断是否要打开门
        print("欢迎"+str(identity)+"回家")
        is_door_open=True
    else:
        print("经验证，您与"+str(identity)+"不符！")
        is_door_open=False

    return dist,is_door_open


def who_is_it(image_path,database,model):
    encoding=fr_utils.img_to_encoding(image_path,model)  # 计算图像的编码
    min_dist=100  # 初始化为足够大的数字
    for (name,db_enc) in database.items():  # 遍历数据库，找到最相近的编码
        dist=np.linalg.norm(encoding-db_enc)  # 计算编码距离
        print(dist)
        if dist<min_dist:  # 更新
            min_dist=dist
            identity=name
        if min_dist>0.07:  # 判断是否在数据库中
            print("抱歉，您的信息不在数据库中")
        else:
            print("姓名："+str(identity)+" 差距："+str(min_dist))
    return min_dist,identity




if __name__ != "__main__":
    # with tf.Session() as test:
    #     tf.set_random_seed(1)
    #     y_true = (None, None, None)
    #     m=3
    #     y_pred = (tf.random_normal([m, 128], mean=6, stddev=0.1, seed=1),
    #               tf.random_normal([m, 128], mean=1, stddev=1, seed=1),
    #               tf.random_normal([m, 128], mean=3, stddev=4, seed=1))
    #     loss = triplet_loss(y_true, y_pred)
    #     print("loss=" + str(loss.eval()))

    FRmodel = faceRecoModel(input_shape=(3, 96, 96))
    # 开始时间
    start_time = time.clock()

    # 编译模型
    FRmodel.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])

    # 加载权值
    fr_utils.load_weights_from_FaceNet(FRmodel)

    # 结束时间
    end_time = time.clock()

    # 计算时差
    minium = end_time - start_time

    print("执行了：" + str(int(minium / 60)) + "分" + str(int(minium % 60)) + "秒")
