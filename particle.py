import tensorflow.keras.backend
import numpy as np
from copy import deepcopy
import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout,Dense,Flatten
from tensorflow.keras.layers import Activation,MaxPool2D,AveragePooling2D,SeparableConv2D
from tensorflow.keras.layers import BatchNormalization
import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras.optimizers import SGD
from data_prepare import get_train_batch
import random
from random import choice

class Particle:
    def __init__(self, min_layer, max_layer, max_pool_layers, input_width, input_height, input_channels,conv_prob, pool_prob, fc_prob, max_conv_kernel, max_out_ch, max_fc_neurons, output_dim):
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels

        self.num_pool_layers = 0
        self.max_pool_layers = max_pool_layers

        self.feature_width = input_width
        self.feature_height = input_height

        self.depth = np.random.randint(min_layer, max_layer)
        self.conv_prob = conv_prob
        self.pool_prob = pool_prob
        self.fc_prob = fc_prob
        self.max_conv_kernel = max_conv_kernel
        self.max_out_ch = max_out_ch

        self.max_fc_neurons = max_fc_neurons
        self.max_p = max_fc_neurons
        self.output_dim = output_dim

        self.layers = []
        self.acc = None
        self.vel = []
        self.pBest = []

        self.initialization()

        for i in range(len(self.layers)):
            if self.layers[i]["type"] != "fc":
                self.vel.append({"type" : "keep"})
            else:
                self.vel.append({"type":"keep_fc"})

        self.model = None
        self.pBest = deepcopy(self)

    def __str__(self):
        string = ""
        for z in range(len(self.layers)):
            string = string + self.layers[z]["type"] + " | "

        return string

    def initialization(self): ##初始化网络结构
        out_channel = np.random.randint(5,self.max_out_ch)
        ##改卷积核的大小操作
        ##卷积核从【3，5，7】中选择，为了减少参数量和计算量，如果卷积核为5，就换成二重3*3卷积，如果是7，就换成三重3*3卷积，同时不影响最大网络层数，即将二重或三重看作一层
        conv_kernel = choice([3,5,7])
        #conv_kernel = np.random.randint(3,self.max_conv_kernel)

        ##第一层通常是卷积层
        self.layers.append({"type":"conv","ou_c":out_channel,"kernel":conv_kernel})

        conv_prob = self.conv_prob ##初始化卷积占比
        pool_prob = conv_prob + self.pool_prob  ##每个卷积层配一个池化层，再添加一部分池化层
        fc_prob = pool_prob  ##每个池化跟一个全连接？

        for i in range(1,self.depth):
            if self.layers[-1]["type"] == "fc":
                layer_type = 1.1  ##如果最后一层是全连接层，把系数设置成大于1的数
            else:
                layer_type = np.random.rand()  ##如果不是，那么就把系数设置成0到1之间的随机值

            ##添加网络层级结构
            if layer_type < conv_prob:
                self.layers = utils.add_conv(self.layers,self.max_out_ch,self.max_conv_kernel,self.layers[-1]["ou_c"])

            elif layer_type >= conv_prob and layer_type <= pool_prob:
                self.layers, self.num_pool_layers = utils.add_pool(self.layers, self.num_pool_layers,self.max_pool_layers)

            elif layer_type >= fc_prob:
                self.layers = utils.add_fc(self.layers, self.max_fc_neurons)
                if self.layers[-1]["ou_c"] > 5:
                    self.max_fc_neurons = self.layers[-1]['ou_c']

        self.layers[-1] = {"type": "fc", "ou_c": self.output_dim, "kernel": -1}

    def velocity(self,gBest,Cg):
        self.vel = utils.computeVelocity(gBest,self.pBest.layers,self.layers,Cg)

    def update(self):
        new_p = utils.updateParticle(self.layers, self.vel)
        new_p = self.validate(new_p)

        self.layers = new_p
        self.model = None

    def validate(self,list_layers):##验证
        list_layers[-1] = {"type":"fc", "ou_c": self.output_dim,"kernel":-1} ##最后一层为全连接

        self.num_pool_layers = 0
        for i in range(len(list_layers)):
            if list_layers[i]["type"] == "max_pool" or list_layers[i]["type"]=="avg_pool":
                self.num_pool_layers += 1

                if self.num_pool_layers >= self.max_pool_layers:
                    list_layers[i]["type"] = 'remove'

        # Now, fix the inputs of each conv and pool layers
        updated_list_layers = []

        for i in range(0, len(list_layers)):
            if list_layers[i]["type"] != "remove":
                if list_layers[i]["type"] == "conv":
                    updated_list_layers.append({"type": "conv", "ou_c": list_layers[i]["ou_c"], "kernel": list_layers[i]["kernel"]})

                if list_layers[i]["type"] == "fc":
                    updated_list_layers.append(list_layers[i])

                if list_layers[i]["type"] == "max_pool":
                    updated_list_layers.append({"type": "max_pool", "ou_c": -1, "kernel": 2})

                if list_layers[i]["type"] == "avg_pool":
                    updated_list_layers.append({"type": "avg_pool", "ou_c": -1, "kernel": 2})

        return updated_list_layers


    def model_compile(self,dropout_rate):
        list_layers = self.layers
        self.model = Sequential()

        for i in range(len(list_layers)):
            if list_layers[i]["type"] == "conv":
                n_out_filters = list_layers[i]["ou_c"]
                kernel_size = list_layers[i]["kernel"]

                if i == 0: ##如果是第一层(对于卷积的第一层需要进行参数初始化)
                    in_w = self.input_width
                    in_h = self.input_height
                    in_c = self.input_channels  ##输入原始数据的参数
                    ##这里的add方式只能选择keras架构，因为keras的封装允许卷积时不输入初始图片大小
                    ##需要判断一下卷积核的大小
                    if kernel_size == 3:
                        self.model.add(SeparableConv2D(n_out_filters, kernel_size, strides=(1,1), padding="same", data_format="channels_last", kernel_initializer='he_normal', bias_initializer='he_normal', activation=None,input_shape=(in_w,in_h,in_c)))
                        self.model.add(BatchNormalization()) ##添加norm层防止数据进行Relu前因为过大而导致不稳定
                        self.model.add(Activation("relu"))
                    elif kernel_size == 5: ##二重3*3卷积
                        self.model.add(SeparableConv2D(n_out_filters, kernel_size=3, strides=(1, 1), padding="same",
                                                       data_format="channels_last", kernel_initializer='he_normal',
                                                       bias_initializer='he_normal', activation=None,
                                                       input_shape=(in_w, in_h, in_c)))
                        self.model.add(SeparableConv2D(n_out_filters, kernel_size=3, strides=(1, 1), padding="same",
                                                       data_format="channels_last", kernel_initializer='he_normal',
                                                       bias_initializer='he_normal', activation=None,
                                                       input_shape=(in_w, in_h, n_out_filters)))
                        self.model.add(BatchNormalization())  ##添加norm层防止数据进行Relu前因为过大而导致不稳定
                        self.model.add(Activation("relu"))
                    elif kernel_size == 7:  ##三重3*3卷积
                        self.model.add(SeparableConv2D(n_out_filters, kernel_size=3, strides=(1, 1), padding="same",
                                                       data_format="channels_last", kernel_initializer='he_normal',
                                                       bias_initializer='he_normal', activation=None,
                                                       input_shape=(in_w, in_h, in_c)))
                        self.model.add(SeparableConv2D(n_out_filters, kernel_size=3, strides=(1, 1), padding="same",
                                                       data_format="channels_last", kernel_initializer='he_normal',
                                                       bias_initializer='he_normal', activation=None,
                                                       input_shape=(in_w, in_h, n_out_filters)))
                        self.model.add(SeparableConv2D(n_out_filters, kernel_size=3, strides=(1, 1), padding="same",
                                                       data_format="channels_last", kernel_initializer='he_normal',
                                                       bias_initializer='he_normal', activation=None,
                                                       input_shape=(in_w, in_h, n_out_filters)))
                        self.model.add(BatchNormalization())  ##添加norm层防止数据进行Relu前因为过大而导致不稳定
                        self.model.add(Activation("relu"))
                else: ##如果不是第一层，那就先添加dropout
                    self.model.add(Dropout(dropout_rate))
                    ##需要判断一下卷积核的大小
                    if kernel_size == 3:
                        self.model.add(SeparableConv2D(n_out_filters, kernel_size=3, strides=(1, 1), padding="same",
                                                       kernel_initializer='he_normal', bias_initializer='he_normal',
                                                       activation=None))
                        self.model.add(BatchNormalization())  ##添加norm层防止数据进行Relu前因为过大而导致不稳定
                        self.model.add(Activation("relu"))
                    elif kernel_size == 5:  ##二重3*3卷积
                        self.model.add(SeparableConv2D(n_out_filters, kernel_size=3, strides=(1, 1), padding="same",
                                                       kernel_initializer='he_normal', bias_initializer='he_normal',
                                                       activation=None))
                        self.model.add(SeparableConv2D(n_out_filters, kernel_size=3, strides=(1, 1), padding="same",
                                                       kernel_initializer='he_normal', bias_initializer='he_normal',
                                                       activation=None))
                        self.model.add(BatchNormalization())  ##添加norm层防止数据进行Relu前因为过大而导致不稳定
                        self.model.add(Activation("relu"))
                    elif kernel_size == 7:  ##三重3*3卷积
                        self.model.add(SeparableConv2D(n_out_filters, kernel_size=3, strides=(1, 1), padding="same",kernel_initializer='he_normal', bias_initializer='he_normal',activation=None))
                        self.model.add(SeparableConv2D(n_out_filters, kernel_size=3, strides=(1, 1), padding="same",kernel_initializer='he_normal', bias_initializer='he_normal',activation=None))
                        self.model.add(SeparableConv2D(n_out_filters, kernel_size=3, strides=(1,1), padding="same", kernel_initializer='he_normal', bias_initializer='he_normal', activation=None))
                        self.model.add(BatchNormalization())
                        self.model.add(Activation("relu"))

            if list_layers[i]["type"] == "max_pool":
                kernel_size = list_layers[i]["kernel"]

                self.model.add(MaxPool2D(pool_size=(3,3), strides=2))

            if list_layers[i]["type"] == "avg_pool":
                self.model.add(AveragePooling2D(pool_size=(3,3), strides=2))

            if list_layers[i]["type"] == "fc":
                '''
                分为两种情况：第一次出现全连接和非第一次
                如果是第一次出现，那么需要加一个Flatten的操作
                '''
                if list_layers[i-1]["type"] != "fc": #如果前一层不是全连接
                    self.model.add(Flatten())

                self.model.add(Dropout(dropout_rate)) ##添加dropout

                if i == len(list_layers) - 1: ##如果是最后一层，输出用softmax
                    self.model.add(Dense(list_layers[i]["ou_c"], kernel_initializer='he_normal', bias_initializer='he_normal', activation=None))
                    self.model.add(BatchNormalization())
                    self.model.add(Activation("softmax"))
                else: ##如果不是最后一层，输出用relu
                    self.model.add(Dense(list_layers[i]["ou_c"], kernel_initializer='he_normal', bias_initializer='he_normal',activation=None))
                    self.model.add(BatchNormalization())
                    self.model.add(Activation("relu"))

        adam = SGD(lr=0.005,decay=0.0)
        self.model.compile(loss="sparse_categorical_crossentropy",optimizer=adam,metrics=["accuracy"],weighted_metrics=['accuracy'])
        #self.model.compile(loss="sparse_categorical_crossentropy",optimizer=adam,metrics=["accuracy"])

    def model_fit(self,x_train,y_train,batch_size,epochs,weight):
        ##用于对某一个sample
        hist = self.model.fit(get_train_batch(x_train, y_train, batch_size),epochs=epochs,
                              verbose=2,steps_per_epoch=x_train.shape[0] // batch_size,class_weight = weight)

        return hist

    def model_fit_complete(self, x_train, y_train, batch_size, epochs,weight):
        hist = self.model.fit(x=x_train, y=y_train, validation_split=0.0, batch_size=batch_size, epochs=epochs,class_weight = weight)

        return hist

    def model_delete(self):
        del self.model
        tensorflow.keras.backend.clear_session()
        self.model = None

    def crossover(self, other_particle):
        """Takes two particles and exchanges part of the solution at
        a specific point"""
        ##首先要决定交叉点
        ##获取较浅层网络的层级数量
        ##只进行卷积池化和全连接层的交换
        ##选择交叉点,选则只进行全连接层前后的交换
        length1 = len(self.layers)
        length2 = len(other_particle.layers)
        for i in range(length1):
            if self.layers[i]['type'] == 'fc':
                crossover_point_self = i
                break
        for j in range(length2):
            if other_particle.layers[j]['type'] == 'fc':
                crossover_point_other = j
                break
        new_first_half = self.layers[:crossover_point_self]
        new_second_half = other_particle.layers[crossover_point_other:]
        new_particle = deepcopy(self)
        new_layers = new_first_half
        new_layers.extend(new_second_half)
        new_particle.layers = new_layers
        '''
        存在问题：
        目前是在浅层的网络中随机选择交换的位置
        可以选择只进行卷积池化和全连接层的交换
        '''
        return new_particle

    def mutate(self):
        """Changes some parts of x based on mutation probability"""
        '''层级结构的突变，可以通过随机数的方式决定改变的类型：
        1.不改变层类型的突变
        a) 卷积层：卷积层输出通道数的变化/卷积核大小的变化
        b) 池化层：池化层pool_size变化
        c) 全连接层：输出神经元数量的变化
        2.改变层类型的突变
        a) 卷积层：第一层不能改变，只能为卷积
            中间层只能变成池化层
            最后一层卷积可以变成池化或者全连接
        b) 池化层：最后一层池化可以变成全连接
            其他层只能变成卷积
        c) 全连接层：最后一层只能是全连接层
            只有第一层可以变成卷积或者池化
        '''
        new_particle = deepcopy(self)
        ##确定层数
        length = len(self.layers)
        position = random.randint(0, length - 2)
        ##通过随机数选择变异的类型
        mutation_prob = random.random()  ##0-1之间的随机数
        if mutation_prob <= 0.5:
            ##随机选择
            if self.layers[position]["type"] == 'conv':
                out_channel = random.randint(max(3, self.layers[position - 1]['ou_c']),
                                                self.max_out_ch)  ##随机输出通道数，但比上一层要多，比最大限制小
                conv_kernel = choice([3, 5, 7])
                #conv_kernel = random.randint(3, self.max_conv_kernel)  ##随机选择卷积核大小
                new_particle.layers[position] = {"type": "conv", "ou_c": out_channel, "kernel": conv_kernel}
            elif (self.layers[position]["type"] == 'max_pool') or (self.layers[position]["type"] == 'avg_pool'):
                random_pool = random.random()
                if random_pool < 0.5:
                    new_particle.layers[position] = {"type": "max_pool", "ou_c": -1, "kernel": 2}
                else:
                    new_particle.layers[position] = {"type": "avg_pool", "ou_c": -1, "kernel": 2}
            elif self.layers[position]["type"] == 'fc':
                # new_particle.layers[position] = {"type": "max_pool", "ou_c": -1, "kernel": 2}
                if self.layers[position - 1]['type'] == 'fc':
                    new_particle.layers[position] = {"type": "fc",
                                                     "ou_c": random.randint(5, self.layers[position - 1]['ou_c']),
                                                     "kernel": -1}
                else:
                    new_particle.layers[position] = {"type": "fc", "ou_c": random.randint(5, self.max_p),
                                                     "kernel": -1}

        elif mutation_prob > 0.5:
            if self.layers[position]["type"] == 'conv':
                if position == 0:
                    pass
                else:
                    random_pool = random.random()
                    if random_pool < 0.5:
                        new_particle.layers[position] = {"type": "max_pool", "ou_c": -1, "kernel": 2}
                    else:
                        new_particle.layers[position] = {"type": "avg_pool", "ou_c": -1, "kernel": 2}
                    # new_particle.layers[position] = {"type":"avg_pool","ou_c":-1,"kernel":2}
            elif (self.layers[position]["type"] == 'max_pool') or (
                    self.layers[position]["type"] == 'avg_pool'):  ##如果是池化层，前一层必定不是全连接
                if self.layers[position + 1]['type'] == 'fc':
                    next_fc_neurons = self.layers[position + 1]['ou_c']
                    new_particle.layers[position] = {"type": "fc",
                                                     "ou_c": random.randint(next_fc_neurons, self.max_p),
                                                     "kernel": -1}
                elif self.layers[position + 1]['type'] != 'fc':
                    out_channel = random.randint(max(3, self.layers[position - 1]['ou_c']),
                                                    self.max_out_ch)  ##随机输出通道数，但比上一层要多，比最大限制小
                    conv_kernel = choice([3, 5, 7])
                    #conv_kernel = random.randint(3, self.max_conv_kernel)  ##随机选择卷积核大小
                    new_particle.layers[position] = {"type": "conv", "ou_c": out_channel, "kernel": conv_kernel}
            elif self.layers[position]["type"] == 'fc':
                if self.layers[position - 1]['type'] != 'fc':
                    random_pool = random.random()
                    if random_pool < 0.5:
                        new_particle.layers[position] = {"type": "max_pool", "ou_c": -1, "kernel": 2}
                    else:
                        new_particle.layers[position] = {"type": "avg_pool", "ou_c": -1, "kernel": 2}
                else:
                    pass
        return new_particle
