# %load mnist_loader.py
import pickle
import gzip
import mnist_loader
import network
import numpy as np
import scipy.io as scio
import random
import numpy as np

def load_data():
    imageFile = 'LAB2\\mnist_train.mat'
    labelFile = 'LAB2\\mnist_train_labels.mat'

    image = scio.loadmat(imageFile)
    label = scio.loadmat(labelFile)

    image = image['mnist_train']
    label = label['mnist_train_labels']

    image = np.array(image, dtype='float32')
    image = image / 255.0  # 数据归一化
    label = np.array(label, dtype='int64')

    data = list(zip(image, label))  # 把两个array拼接到一起，转换成list 以便利用random.shuffle函数
    random.shuffle(data)
    len_data = len(data)
    len_6 = int(0.6 * len_data)
    len_8 = int(0.8 * len_data)

    training_data = data[0:len_6]  # 把数据集分类，找到slice的下表，6 8 10三个坐标
    validation_data = data[len_6:len_8]
    test_data = data[len_8:len_data]

    training_image, training_label = zip(*training_data)  # zip(*a)等于解压 将两个list解开
    training_image = np.array(training_image, dtype='float32')
    training_label = np.array(training_label, dtype='int64')
    training_data = (training_image, training_label)

    validation_image, validation_label = zip(*validation_data)
    validation_image = np.array(validation_image, dtype='float32')
    validation_label = np.array(validation_label, dtype='int64')
    validation_data = (validation_image, validation_label)

    test_image, test_label = zip(*test_data)
    test_image = np.array(test_image, dtype='float32')
    test_label = np.array(test_label, dtype='int64')
    test_data = (test_image, test_label)


    training_inputs = [np.reshape(x, (784, 1)) for x in training_data[0]]
    training_results = [vectorized_result(y) for y in training_data[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in validation_data[0]]
    validation_data = zip(validation_inputs, validation_data[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
    test_data = zip(test_inputs, test_data[1])


    return (training_data, validation_data, test_data)


def vectorized_result(j):
    #用一个one-hot 编码的形式， 即 10个数字用10个二进制表示，是哪个数，就哪个下标标为1
    #这样神经网络输出的时候，输出神经元10个 每个输出他的概率 取最大的 和这个标签对应即可
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
