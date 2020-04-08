
import pickle
import gzip
import mnist_loader
import numpy as np
import scipy.io as scio
import random

import numpy as np

def load_data():
    imageFile = 'LAB2\\usps_train.mat'
    labelFile = 'LAB2\\usps_train_labels.mat'

    image = scio.loadmat(imageFile)
    label = scio.loadmat(labelFile)

    image = image['usps_train']
    label = label['usps_train_labels']

    image = np.array(image, dtype='double')
    # image = image / 255.0  # 数据归一化    因为usps数据集已经归一化了，所以不用再归一化了
    label = np.array(label, dtype='double')

    # print(label)
    # exit()

    data = list(zip(image, label))  # 把两个array拼接到一起，转换成list 以便利用random.shuffle函数
    random.shuffle(data)
    len_data = len(data)
    len_6 = int(0.6 * len_data)
    len_8 = int(0.8 * len_data)

    training_data = data[0:len_6]  # 把数据集分类，找到slice的下表，6 8 10三个坐标
    validation_data = data[len_6:len_8]
    test_data = data[len_8:len_data]

    training_image, training_label = zip(*training_data)  # zip(*a)等于解压 将两个list解开
    training_image = np.array(training_image, dtype='double')
    training_label = np.array(training_label, dtype='double')
    training_data = (training_image, training_label)

    validation_image, validation_label = zip(*validation_data)
    validation_image = np.array(validation_image, dtype='double')
    validation_label = np.array(validation_label, dtype='double')
    validation_data = (validation_image, validation_label)

    test_image, test_label = zip(*test_data)
    test_image = np.array(test_image, dtype='double')
    test_label = np.array(test_label, dtype='double')
    test_data = (test_image, test_label)

    return (training_data, validation_data, test_data)