import svmpy
import logging
import numpy as np
import itertools
import mnist_loader
import usps_loader
import copy

def calculate_acc(predictor, test_image, test_label):
    acc_num = 0
    for i in range(len(test_image)):
        # a = predictor.predict(test_image[i])
        # print(a.shape)
        if predictor.predict(test_image[i]) == test_label[i]:
            acc_num += 1
    acc = acc_num*1.0/len(test_image)
    return acc

def data_loader(train_data_real, number):#为了分割取出来的数据，因为每次只能二分类，所以要构建10个分类器 nsps是11个分类器
    #如果只用完全相等 即 若该分类器只在某类上鉴别为正类其他为负类才算正确样本，会导致准确率特别低，约为40%
    train_data = copy.deepcopy(train_data_real)     #这里必须用深拷贝，不然数据会被一层一层覆盖
    for i in range(len(train_data[1])):
        if(train_data[1][i] != number):
            train_data[1][i] = -1
        if (train_data[1][i] == number):
            train_data[1][i] = 1
    return train_data

def train(train_data_real, trainer, max_number):
    pre = []                        #pre存储的是n个分类器，ovr要求分类器的个数为n
    train_data_real = train_data_real
    for number in range(0, max_number+1):
        print('Now we prepare number {} dataset'.format(number))
        train_data = data_loader(train_data_real, number)
        train_image = np.array(train_data[0])
        train_label = np.array(train_data[1])
        # predictor = trainer.train(train_image[:500], train_label[:500])       #因为数据太多会执行太慢，所以测试用这个
        predictor = trainer.train(train_image, train_label)
        pre.append(predictor)
    return pre

def tst(test_data, pre, max_number):
    acc_num = 0
    test_len = len(test_data[0])
    test_image = np.array(test_data[0])
    test_label = np.array(test_data[1])
    for i in range(test_len):               #总共有多少图片进行测试
        flag_index = -1                     #记录该图片的预测到底是被预测为哪个类了
        max_dis = 0                         #最大距离，最大的才算是该类，跟分类的相比就是最接近于1的算是该类
        for j in range(max_number + 1):     #总共有多少个pre要进行测试
            flag, dis = pre[j].predict(test_image[i])   #flag代表是不是正类，dis是在距离超平面的距离
            if(flag == 1.0 and dis > max_dis):          #找到最大的距离，代表某张图片具体的分类
                flag_index = j
                max_dis = dis
        if(flag_index == test_label[i]):
            acc_num+=1
    return acc_num*1.0/test_len
if __name__ == "__main__":
    input = input()
    if(input == 'mnist'):
        train_data, validation_data, test_data = mnist_loader.load_data()
        # trainer = svmpy.SVMTrainer(svmpy.Kernel.gaussian(1), 0.1)
        trainer = svmpy.SVMTrainer(svmpy.Kernel.linear(), 0.1)
        pre = train(train_data,trainer, 9)
        print(len(pre))
        acc = tst(test_data, pre, 9)
        print("Accuracy on Mnist Dataset: {:.3f}% ".format(acc*100))
    elif(input == 'usps'):
        train_data, validation_data, test_data = usps_loader.load_data()
        # print(len(train_data[0]))
        # trainer = svmpy.SVMTrainer(svmpy.Kernel.gaussian(1), 0.1)
        trainer = svmpy.SVMTrainer(svmpy.Kernel.linear(), 0.1)
        pre = train(train_data,trainer, 10)
        print(len(pre))
        acc = tst(test_data, pre, 10)
        print("Accuracy on Usps Dataset: {:.3f}% ".format(acc*100))
    else:
        print('Please input exist dataset like: mnist or usps')
