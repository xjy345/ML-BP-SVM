import mnist_loader
import usps_loader
import network
import numpy as np
import scipy.io as scio
import random


if __name__ == '__main__':
    flag = input()
    net_arc = input()
    net_arc = int(net_arc)
    #神经网络训练总次数均为30次，mini_batch为10个，学习率为3.0
    if(flag == 'mnist'):
        training_data, validation_data, test_data = mnist_loader.load_data()
        net = network.Network([784, net_arc, 10])
        net.SGD(training_data, 30, 10, 3.0, test_data=test_data, validation_data=validation_data)
    elif(flag == 'usps'):
        training_data, validation_data, test_data = usps_loader.load_data()
        net = network.Network([256, net_arc, 11])
        net.SGD(training_data, 30, 10, 3.0, test_data=test_data, validation_data=validation_data)
    else:
        print('Please input exist dataset like: mnist or usps')