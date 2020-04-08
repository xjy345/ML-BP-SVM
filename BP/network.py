# %load network.py
import random
import numpy as np

class Network(object):

    def __init__(self, sizes):
        # 接受一个list 的输入 为sizes 例如[784,30,10] 意思是第一层784个神经元 第二层30个神经元 第三层10个神经元
        #权重w和偏置b随机初始化为 N(0,1) 另外输入层没有偏置b
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        #return 一个经过sigmoid函数的结果 范围为[0,1]
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None, validation_data=None):

        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
        if validation_data:
            validation_data = list(validation_data)
            n_validation = len(validation_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if validation_data:
                acc_val = self.evaluate(validation_data) / (1.0*n_validation) * 100
                print("Epoch {} : {:.3f}% on Validation Data ".format(j+1,acc_val))
            else:
                print("Epoch {} complete".format(j))
        if test_data:
            acc_test = self.evaluate(test_data) / (1.0*n_test) * 100
            print("Test Data Accuracy : {:.3f}% ".format(acc_test))
    def update_mini_batch(self, mini_batch, eta):
        #eta 是学习率 输入是一个mini_batch 包括b个样本
        nabla_b = [np.zeros(b.shape) for b in self.biases]          #bias 两个维度 30个隐含层的和10个输出层的
        nabla_w = [np.zeros(w.shape) for w in self.weights]         #weight 也是两个维度 x--->h的权重和 h--->y的权重

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)      #反向传播计算误差
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward   #前向传播
        activation = x
        activations = [x]                               #逐层的保存经过激活函数的中间结果

        zs = []                                         #逐层的保存每层的输入
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)                        #存储两层信息 1：隐含层的输入(未经过激活函数) 2：输出层的输入(未经过激活函数)
            activation = sigmoid(z)             #存储三层信息 1：源输入 2：隐含层的输出 3:输出层的输出
            activations.append(activation)
        # backward pass     #反向传播
        #先计算误差
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])       #第一次求导
        nabla_b[-1] = delta             #隐含层的bias
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):             #x--->h的权重更新
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):

        #np.argmax()函数 返回最大值的索引，这里为了将分类器的输出层的下标找到，以便对应真实标签Y
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        #误差函数
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):         #激活函数利用sigmoid
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):       #sigmod函数的求导
    return sigmoid(z)*(1-sigmoid(z))
