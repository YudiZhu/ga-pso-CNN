import tensorflow.keras
import tensorflow.keras.backend
import numpy as np
from population import Population
from copy import deepcopy
import random
from sklearn.utils import shuffle
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.losses import SparseCategoricalCrossentropy
CROSSOVER_PROB=0.7
import sys
def selection(swarm):
    ##想要在种群中找到用来执行交叉或变异操作的父本母本
    ##针对初始化的种群，需要在一轮训练全部完成之后对其中的优秀粒子进行选择，然后执行mutate和cross_over
    ##但是这种选择不是完全摒弃，而是要让适应度更高的粒子获得更大的被选择概率
    fitness = []
    for particle in swarm:
        fitness.append(1/particle.acc)

    # Probability over total probability
    fit_r_sum = sum(fitness)
    selection_probability = []
    for relative_fit in fitness:
        selection_probability.append(relative_fit / fit_r_sum)

    # Cumulative probability 累计概率
    cumulative_probability = []
    the_sum = 0
    for a in selection_probability:
        the_sum += a
        cumulative_probability.append(the_sum)
    #For the new generation, we compare a random number between 0 and 1
    #and we select the particle that has the next greater cumulative
    #probability
    probability = random.random()
    for i in range(0, len(cumulative_probability)):
        if probability <= cumulative_probability[i]:
            new_kid = swarm[i]
            break
    # Make new copy
    a_new_kid = deepcopy(new_kid)
    return a_new_kid

# class Logger(object):
#     def __init__(self, filename="Default.log"):
#         self.terminal = sys.stdout
#         self.log = open(filename, "a")
#         # self.log = open(filename, "a", encoding="utf-8")  # 防止编码错误
#
#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)
#
#     def flush(self):
#         pass

class psoCNN:
    def __init__(self, dataset, n_iter, pop_size, batch_size, epochs, min_layer, max_layer,
                conv_prob, pool_prob, fc_prob, max_conv_kernel, max_out_ch, max_fc_neurons, dropout_rate):
        self.pop_size = pop_size
        self.new_popsize = None
        self.n_iter = n_iter
        self.epochs = epochs

        self.batch_size = batch_size
        self.gBest_acc = np.zeros(n_iter)
        self.gBest_test_acc = np.zeros(n_iter)

        if dataset == 'turbulent':
            input_width = 128
            input_height = 128
            input_channels = 1
            output_dim = 5

            self.x_train = np.load("turbulent_train_data.npy")
            self.x_test = np.load("turbulent_test_data.npy")
            self.x_valid = np.load("turbulent_evaluate_data.npy")
            self.y_train = np.load("turbulent_train_label.npy")
            self.y_test = np.load("turbulent_test_label.npy")
            self.y_valid = np.load("turbulent_evaluate_label.npy")
            # img_list = np.load("/gemini/data-1/turbulent_data.npy")
            # labels = np.load("/gemini/data-1/turbulent_label.npy")

        if dataset == 'tao':
            input_width = 128
            input_height = 128
            input_channels = 1
            output_dim = 11

            self.x_train = np.load("tao_train_data.npy")
            self.x_test = np.load("tao_test_data.npy")
            self.x_valid = np.load("tao_evaluate_data.npy")
            self.y_train = np.load("tao_train_label.npy")
            self.y_test = np.load("tao_test_label.npy")
            self.y_valid = np.load("tao_evaluate_label.npy")
            # img_list = np.load("/gemini/data-2/tao_data.npy")
            # labels = np.load("/gemini/data-2/tao_label.npy")

        if dataset == 'WTD':
            input_width = 128
            input_height = 128
            input_channels = 1
            output_dim = 8

            self.x_train = np.load("WTD_train_data.npy")
            self.x_test = np.load("WTD_test_data.npy")
            self.x_valid = np.load("WTD_evaluate_data.npy")
            self.y_train = np.load("WTD_train_label.npy")
            self.y_test = np.load("WTD_test_label.npy")
            self.y_valid = np.load("WTD_evaluate_label.npy")
            
        self.x_train,self.y_train = shuffle(self.x_train,self.y_train,random_state=415)
        num = self.x_train.shape[0]
        counts = np.bincount(self.y_train)
        class_weight = {}
        for i in range(len(counts)):
            class_weight[i] = num * 1./counts[i]
        self.class_weight = class_weight
        
        #sys.stdout = Logger('out_log_003.txt')
        ##乱序
        # num_example = img_list.shape[0]
        # arr = np.arange(num_example)
        # np.random.shuffle(arr)
        # img_list = img_list[arr]
        # labels = labels[arr]
        ##数据切分
        ##切分为三个数据集，训练集，验证集，测试集
        ##让acc里保存在验证集上模型的表现，在test_acc里保存测试集上模型表现，即实际选择模型通过valid数据，模型的泛化性能表现通过test数据
        # self.x_split, self.x_test, self.y_split, self.y_test = train_test_split(img_list, labels, test_size=0.2,
        #                                                                         random_state=43)
        # self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(self.x_split, self.y_split, test_size=0.25,
        #                                                                         random_state=43)
        print("Initializing population...")
        ##首先初始化种群，初始化了POP_SIZE个模型结构：初始化，不涉及更新
        self.population = Population(pop_size, min_layer, max_layer, input_width, input_height, input_channels,
                                     conv_prob, pool_prob, fc_prob, max_conv_kernel, max_out_ch, max_fc_neurons,
                                     output_dim)
        self.pop_size = len(self.population.particle)
        print("verifying accuracy of the current gBEst...")
        ##原始种群的第一个粒子的结构，把目前的gBest定为第一个粒子
        print(self.population.particle[0])
        print(self.population.particle[0].layers)
        self.gBest = deepcopy(self.population.particle[0])
        self.gBest.model_compile(dropout_rate) ##创建模型，根据实际列表中存储的层级信息
        hist = self.gBest.model_fit(self.x_train, self.y_train,self.batch_size,self.epochs,self.class_weight)
        #test_metrics = self.gBest.model.evaluate(x=get_im_cv2(self.x_test), y=self.y_test, batch_size=self.batch_size)  ##evaluate返回值为 <loss，accuracy>
        #val_metrics = self.gBest.model.evaluate(x=self.x_valid, y=self.y_valid,
        #                                         batch_size=self.batch_size)  ##evaluate返回值为 <loss，accuracy>
        valid_output = np.argmax(self.gBest.model.predict(x=self.x_valid,batch_size=self.batch_size),axis=1)
        print(f'valid预测结果：{valid_output}')
        print(f'valid真实结果：{self.y_valid}')

        m = Accuracy()
        m.update_state(self.y_valid,valid_output)
        val_metrics = m.result().numpy()
        print(f'valid acc: {val_metrics}')
        m.reset_states()
        # test_metrics = self.gBest.model.evaluate(x=self.x_test, y=self.y_test,
        #                                         batch_size=self.batch_size)
        test_output = np.argmax(self.gBest.model.predict(x=self.x_test,batch_size=self.batch_size),axis=1)
        m.update_state(self.y_test,test_output)
        test_metrics = m.result().numpy()
        m.reset_states()
        print(f'test预测结果：{test_output}')
        print(f'test真实结果：{self.y_test}')
        print(f'test acc: {test_metrics}')
        self.gBest.model_delete()

        ##初始化的最优位置定为第一个粒子的位置
        self.gBest_acc[0] = val_metrics
        self.gBest_test_acc[0] = test_metrics  ##[0]为loss value
        ##评价一下两种方式的问题，这里的pbest和gbest应该是train_acc还是test_acc？？？
        # self.population.particle[0].acc = hist.history['accuracy'][-1]
        # self.population.particle[0].pBest.acc = hist.history['accuracy'][-1]
        
        self.population.particle[0].acc = val_metrics
        self.population.particle[0].pBest.acc = val_metrics
        ##目前是第一个粒子的验证集和测试集准确率
        print("Current gBest acc: " + str(self.gBest_acc[0]) + "\n")
        print("Current gBest test acc: " + str(self.gBest_test_acc[0]) + "\n")

        print("looking for a new gBest...")
        for i in range(1,self.pop_size):
            print("initialize - Particle: " + str(i+1))
            print(self.population.particle[i])
            print(self.population.particle[i].layers)

            self.population.particle[i].model_compile(dropout_rate)  ##模型构建
            hist = self.population.particle[i].model_fit(self.x_train, self.y_train,self.batch_size,self.epochs,self.class_weight)
            # val_metrics = self.population.particle[i].model.evaluate(x=self.x_valid, y=self.y_valid,
            #                                      batch_size=self.batch_size)
            valid_output = np.argmax(self.population.particle[i].model.predict(x=self.x_valid, batch_size=self.batch_size),axis=1)
            print(f'valid预测结果：{valid_output}')
            print(f'valid真实结果：{self.y_valid}')
            m.update_state(self.y_valid, valid_output)
            val_metrics = m.result().numpy()
            m.reset_states()
            print(f'valid acc: {val_metrics}')
            
            self.population.particle[i].model_delete()  #为了不占用空间，获取数据后把模型删掉了
            # self.population.particle[i].acc = hist.history['accuracy'][-1]
            # self.population.particle[i].pBest.acc = hist.history['accuracy'][-1]
            self.population.particle[i].acc = val_metrics
            self.population.particle[i].pBest.acc = val_metrics

            if self.population.particle[i].pBest.acc >= self.gBest_acc[0]:
                print("Found a new gBest.")
                self.gBest = deepcopy(self.population.particle[i])
                self.gBest_acc[0] = self.population.particle[i].pBest.acc
                print("New gBest acc: " + str(self.gBest_acc[0]))

                self.gBest.model_compile(dropout_rate)  #找到新点重新创建模型
                hist = self.gBest.model_fit(self.x_train, self.y_train,self.batch_size,self.epochs,self.class_weight)
                #test_metrics = self.gBest.model.evaluate(x=get_im_cv2(self.x_test),y=self.y_test,batch_size=self.batch_size)
                # test_metrics = self.gBest.model.evaluate(x=self.x_test, y=self.y_test,
                #                                          batch_size=self.batch_size)
                test_output = np.argmax(self.gBest.model.predict(x=self.x_test, batch_size=self.batch_size),axis=1)
                m.update_state(self.y_test, test_output)
                test_metrics = m.result().numpy()
                m.reset_states()
                print(f'test预测结果：{test_output}')
                print(f'test真实结果：{self.y_test}')

                self.gBest_test_acc[0] = test_metrics
                print("New gBest test acc:" + str(self.gBest_test_acc[0]))
            
            self.gBest.model_delete()

        print("generating a new swarm...")
        for j in range(1, pop_size):
            # Decide for crossover  ##交叉变异
            dont_crossover = random.random()
            if dont_crossover < CROSSOVER_PROB:  ##如果发生交叉互换
                parent1 = selection(self.population.particle)
                parent2 = selection(self.population.particle)
                a_new_kid = parent1.crossover(parent2)
            else:  ##如果不发生交叉互换
                a_new_kid = selection(self.population.particle)
            a_new_kid = a_new_kid.mutate()  ##发生变异
            self.population.new_particle.append(a_new_kid)
            self.population.particle.append(a_new_kid)

        self.new_popsize = len(self.population.new_particle)

        for i in range(1, self.new_popsize):
            print("initialize - Particle: " + str(self.pop_size + i))
            print(self.population.new_particle[i])
            print(self.population.new_particle[i].layers)

            self.population.new_particle[i].model_compile(dropout_rate)  ##模型构建
            hist = self.population.new_particle[i].model_fit(self.x_train, self.y_train, self.batch_size,
                                                         self.epochs,class_weight)
            # val_metrics = self.population.new_particle[i].model.evaluate(x=self.x_valid, y=self.y_valid,
            #                                                 batch_size=self.batch_size)
            valid_output = np.argmax(self.population.new_particle[i].model.predict(x=self.x_valid, batch_size=self.batch_size),axis=1)
            print(f'valid预测结果：{valid_output}')
            print(f'valid真实结果：{self.y_valid}')
            m.update_state(self.y_valid, valid_output)
            val_metrics = m.result().numpy()
            m.reset_states()
            print(f'valid acc: {val_metrics}')
            
            self.population.new_particle[i].model_delete()  # 为了不占用空间，获取数据后把模型删掉了
            # self.population.particle[i].acc = hist.history['accuracy'][-1]
            # self.population.particle[i].pBest.acc = hist.history['accuracy'][-1]
            self.population.new_particle[i].acc = val_metrics
            self.population.new_particle[i].pBest.acc = val_metrics

            if self.population.new_particle[i].pBest.acc >= self.gBest_acc[0]:
                print("Found a new gBest.")
                self.gBest = deepcopy(self.population.new_particle[i])
                self.gBest_acc[0] = self.population.new_particle[i].pBest.acc
                print("New gBest acc: " + str(self.gBest_acc[0]))

                self.gBest.model_compile(dropout_rate)  # 找到新点重新创建模型
                hist = self.gBest.model_fit(self.x_train, self.y_train, self.batch_size,
                                                         self.epochs,class_weight)
                # test_metrics = self.gBest.model.evaluate(x=get_im_cv2(self.x_test),y=self.y_test,batch_size=self.batch_size)
                # test_metrics = self.gBest.model.evaluate(x=self.x_test, y=self.y_test,
                #                                          batch_size=self.batch_size)
                test_output = np.argmax(self.gBest.model.predict(x=self.x_test, batch_size=self.batch_size),axis=1)
                m.update_state(self.y_test, test_output)
                test_metrics = m.result().numpy()
                m.reset_states()
                print(f'test预测结果：{test_output}')
                print(f'test真实结果：{self.y_test}')

                self.gBest_test_acc[0] = test_metrics
                print("New gBest test acc:" + str(self.gBest_test_acc[0]))
            
            self.gBest.model_delete()

    def fit(self, Cg, dropout_rate):
        # sys.stdout = Logger('out_log_003.txt')
        # ##目的是找到每一次迭代中的种群最优解
        for i in range(1, self.n_iter): ##每一次迭代
            gBest_acc = self.gBest_acc[i-1]
            gBest_test_acc = self.gBest_test_acc[i-1]

            for j in range(self.pop_size+self.new_popsize):  ##每一个粒子
                print("Iteration :" + str(i) + '-Particle:' +str(j+1))

                self.population.particle[j].velocity(self.gBest.layers, Cg)
                self.population.particle[j].update()

                print('Particle new architecture:')
                print(self.population.particle[j])
                print(self.population.particle[j].layers)

                self.population.particle[j].model_compile(dropout_rate)
                hist = self.population.particle[j].model_fit(self.x_train, self.y_train,self.batch_size,self.epochs,self.class_weight)
                #val_metrics = self.population.particle[j].model.evaluate(self.x_valid, self.y_valid,self.batch_size)
                valid_output = np.argmax(self.population.particle[j].model.predict(x=self.x_valid, batch_size=self.batch_size),axis=1)
                print(f'valid预测结果：{valid_output}')
                print(f'valid真实结果：{self.y_valid}')
                m = Accuracy()
                m.update_state(self.y_valid, valid_output)
                val_metrics = m.result().numpy()
                m.reset_states()
                print(f'valid acc: {val_metrics}')

                self.population.particle[j].model_delete()

                self.population.particle[j].acc = val_metrics

                f_test = self.population.particle[j].acc
                pBest_acc = self.population.particle[j].pBest.acc

                if f_test >= pBest_acc:
                    print("Found a new pBest.")
                    print("Current acc: " + str(f_test))
                    print("Past pBest acc: " + str(pBest_acc))
                    pBest_acc = f_test
                    self.population.particle[j].pBest = deepcopy(self.population.particle[j])
                    
                    if pBest_acc >= gBest_acc:
                        print("Found a new gBest.")
                        gBest_acc = pBest_acc
                        self.gBest = deepcopy(self.population.particle[j])

                        self.gBest.model_compile(dropout_rate)
                        hist = self.gBest.model_fit(self.x_train, self.y_train,self.batch_size,self.epochs,self.class_weight)
                        #test_metrics = self.gBest.model.evaluate(x=get_im_cv2(self.x_test), y=self.y_test, batch_size=self.batch_size)
                        # test_metrics = self.gBest.model.evaluate(x=self.x_test, y=self.y_test,
                        #                                          batch_size=self.batch_size)
                        test_output = np.argmax(self.gBest.model.predict(x=self.x_test, batch_size=self.batch_size),axis=1)
                        m.update_state(self.y_test, test_output)
                        test_metrics = m.result().numpy()
                        m.reset_states()
                        print(f'test预测结果：{test_output}')
                        print(f'test真实结果：{self.y_test}')

                        self.gBest.model_delete()
                        gBest_test_acc = test_metrics  ##

            self.gBest_acc[i] = gBest_acc
            self.gBest_test_acc[i] = gBest_test_acc

            print("Current gBest evaluate acc: " + str(self.gBest_acc[i]))
            print("Current gBest test acc: " + str(self.gBest_test_acc[i]))

    def fit_gBest(self,batch_size,epochs,dropout_rate):
        # sys.stdout = Logger('out_log_003.txt')
        print("\nFurther training gBest model...")
        self.gBest.model_compile(dropout_rate)

        trainable_count = 0
        for i in range(len(self.gBest.model.trainable_weights)):
            trainable_count += tensorflow.keras.backend.count_params(self.gBest.model.trainable_weights[i])

        print("gBest's number of trainable parameters: " + str(trainable_count)) ##可训练参数量
        self.gBest.model_fit(self.x_train, self.y_train,self.batch_size,self.epochs,self.class_weight)

        return trainable_count

    def evaluate_gBest(self, batch_size):
        # sys.stdout = Logger('out_log_003.txt')
        print("\nEvaluating gBest model on the test set...")
        #metrics = self.gBest.model.evaluate(x=self.x_test, y=self.y_test, batch_size=self.batch_size)
        #metrics = self.gBest.model.evaluate(x=get_im_cv2(self.x_test), y=self.y_test, batch_size=self.batch_size)
        test_output = np.argmax(self.gBest.model.predict(x=self.x_test, batch_size=self.batch_size),axis=1)
        m = Accuracy()
        m.update_state(self.y_test, test_output)
        metrics = m.result().numpy()
        m.reset_states()
        print(f'test预测结果：{test_output}')
        print(f'test真实结果：{self.y_test}')
        print("\ngBest model loss in the test set: " + " - Test set accuracy: " + str(metrics))
        return metrics
