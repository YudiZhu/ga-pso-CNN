from particle import Particle

# ##设置交叉和变异方法
# CROSSOVER_PROB = 0.7
#
# def selection(swarm):
#     ##想要在种群中找到用来执行交叉或变异操作的父本母本
#     ##针对初始化的种群，需要在一轮训练全部完成之后对其中的优秀粒子进行选择，然后执行mutate和cross_over
#     ##但是这种选择不是完全摒弃，而是要让适应度更高的粒子获得更大的被选择概率
#     ##先找交叉的粒子，再对粒子进行变异操作
#     fitness = []
#     for particle in swarm:
#         fitness.append(1/particle.acc)
#
#     # Probability over total probability
#     fit_r_sum = sum(fitness)
#     selection_probability = []
#     for relative_fit in fitness:
#         selection_probability.append(relative_fit / fit_r_sum)
#
#     # Cumulative probability 累计概率
#     cumulative_probability = []
#     the_sum = 0
#     for a in selection_probability:
#         the_sum += a
#         cumulative_probability.append(the_sum)
#     #For the new generation, we compare a random number between 0 and 1
#     #and we select the particle that has the next greater cumulative
#     #probability
#     probability = random.random()
#     for i in range(0, len(cumulative_probability)):
#         if probability <= cumulative_probability[i]:
#             new_kid = swarm[i]
#             break
#     # Make new copy
#     a_new_kid = deepcopy(new_kid)
#     return a_new_kid
#
# def crossover(one, other_particle):
#     """Takes two particles and exchanges part of the solution at
#     a specific point"""
#     ##首先要决定交叉点
#     ##获取较浅层网络的层级数量
#     ##只进行卷积池化和全连接层的交换
#     ##选择交叉点,选则只进行全连接层前后的交换
#     length1 = len(one.layers)
#     length2 = len(other_particle.layers)
#     for i in range(length1):
#         if one.layers[i]['type'] == 'fc':
#             crossover_point_self = i
#             break
#     for j in range(length2):
#         if other_particle.layers[j]['type'] == 'fc':
#             crossover_point_other = j
#             break
#     new_first_half = one.layers[:crossover_point_self]
#     new_second_half = other_particle.layers[crossover_point_other:]
#     new_particle = deepcopy(one)
#     new_layers = new_first_half
#     new_layers.extend(new_second_half)
#     new_particle.layers = new_layers
#     '''
#     存在问题：
#     目前是在浅层的网络中随机选择交换的位置
#     可以选择只进行卷积池化和全连接层的交换
#     '''
#     return new_particle
#
# def mutate(one):
#     """Changes some parts of x based on mutation probability"""
#     '''层级结构的突变，可以通过随机数的方式决定改变的类型：
#     1.不改变层类型的突变
#     a) 卷积层：卷积层输出通道数的变化/卷积核大小的变化
#     b) 池化层：池化层pool_size变化
#     c) 全连接层：输出神经元数量的变化
#     2.改变层类型的突变
#     a) 卷积层：第一层不能改变，只能为卷积
#         中间层只能变成池化层
#         最后一层卷积可以变成池化或者全连接
#     b) 池化层：最后一层池化可以变成全连接
#         其他层只能变成卷积
#     c) 全连接层：最后一层只能是池化层
#         只有第一层可以变成卷积或者池化
#     '''
#     new_particle = deepcopy(one)
#     ##确定层数
#     length = len(one.layers)
#     position = random.randint(0, length - 2)
#     ##通过随机数选择变异的类型
#     mutation_prob = random.random()  ##0-1之间的随机数
#     if mutation_prob <= 0.5:
#         ##随机选择
#         if one.layers[position]["type"] == 'conv':
#             out_channel = random.randint(max(3, one.layers[position - 1]['ou_c']),
#                                             one.max_out_ch)  ##随机输出通道数，但比上一层要多，比最大限制小
#             conv_kernel = random.randint(3, one.max_conv_kernel)  ##随机选择卷积核大小
#             new_particle.layers[position] = {"type": "conv", "ou_c": out_channel, "kernel": conv_kernel}
#         elif (one.layers[position]["type"] == 'max_pool') or (one.layers[position]["type"] == 'avg_pool'):
#             random_pool = random.random()
#             if random_pool < 0.5:
#                 new_particle.layers[position] = {"type": "max_pool", "ou_c": -1, "kernel": 2}
#             else:
#                 new_particle.layers[position] = {"type": "avg_pool", "ou_c": -1, "kernel": 2}
#         elif one.layers[position]["type"] == 'fc':
#             # new_particle.layers[position] = {"type": "max_pool", "ou_c": -1, "kernel": 2}
#             if one.layers[position - 1]['type'] == 'fc':
#                 new_particle.layers[position] = {"type": "fc",
#                                                  "ou_c": random.randint(5, one.layers[position - 1]['ou_c']),
#                                                  "kernel": -1}
#             else:
#                 new_particle.layers[position] = {"type": "fc", "ou_c": random.randint(5, one.max_p),
#                                                  "kernel": -1}
#
#     elif mutation_prob > 0.5:
#         if one.layers[position]["type"] == 'conv':
#             if position == 0:
#                 pass
#             else:
#                 random_pool = random.random()
#                 if random_pool < 0.5:
#                     new_particle.layers[position] = {"type": "max_pool", "ou_c": -1, "kernel": 2}
#                 else:
#                     new_particle.layers[position] = {"type": "avg_pool", "ou_c": -1, "kernel": 2}
#                 # new_particle.layers[position] = {"type":"avg_pool","ou_c":-1,"kernel":2}
#         elif (one.layers[position]["type"] == 'max_pool') or (
#                 one.layers[position]["type"] == 'avg_pool'):  ##如果是池化层，前一层必定不是全连接
#             if one.layers[position + 1]['type'] == 'fc':
#                 next_fc_neurons = one.layers[position + 1]['ou_c']
#                 new_particle.layers[position] = {"type": "fc",
#                                                  "ou_c": random.randint(next_fc_neurons, one.max_p),
#                                                  "kernel": -1}
#             elif one.layers[position + 1]['type'] != 'fc':
#                 out_channel = random.randint(max(3, one.layers[position - 1]['ou_c']),
#                                                 one.max_out_ch)  ##随机输出通道数，但比上一层要多，比最大限制小
#                 conv_kernel = random.randint(3, one.max_conv_kernel)  ##随机选择卷积核大小
#                 new_particle.layers[position] = {"type": "conv", "ou_c": out_channel, "kernel": conv_kernel}
#         elif one.layers[position]["type"] == 'fc':
#             if one.layers[position - 1]['type'] != 'fc':
#                 random_pool = random.random()
#                 if random_pool < 0.5:
#                     new_particle.layers[position] = {"type": "max_pool", "ou_c": -1, "kernel": 2}
#                 else:
#                     new_particle.layers[position] = {"type": "avg_pool", "ou_c": -1, "kernel": 2}
#             else:
#                 pass
#     return new_particle

class Population:
    def __init__(self, pop_size, min_layer, max_layer, input_width, input_height, input_channels, conv_prob,
                 pool_prob, fc_prob, max_conv_kernel, max_out_ch, max_fc_neurons, output_dim):
        max_pool_layers = 0
        in_w = input_width

        while in_w > 5:
            max_pool_layers += 1
            in_w = in_w/2

        self.particle = []
        self.new_particle = []
        for i in range(pop_size):
            self.particle.append(Particle(min_layer,max_layer,max_pool_layers, input_width, input_height, input_channels,
                                          conv_prob, pool_prob, fc_prob, max_conv_kernel, max_out_ch, max_fc_neurons, output_dim))
        ##执行交叉和变异的操作
        ##先找父本母本
        # for j in range(1, pop_size):
        #     #Decide for crossover  ##交叉变异
        #     dont_crossover = random()
        #     if dont_crossover < CROSSOVER_PROB:  ##如果发生交叉互换
        #         parent1 = selection(self.origin_particle)
        #         parent2 = selection(self.origin_particle)
        #         a_new_kid = crossover(parent1,parent2)
        #     else:  ##如果不发生交叉互换
        #        a_new_kid = selection(self.origin_particle)
        #     a_new_kid = a_new_kid.mutate()   ##发生变异
        #     self.particle.append(a_new_kid)

        # ##先变异
        # for each in self.origin_particle:    ##这里我是针对每一个粒子都做了变异操作
        #     self.particle.append(each)
        #     new = mutate(each)
        #     if new.layers != each.layers:
        #         self.particle.append(new)  ##相当于种群扩大两倍
        # ##再交叉
        # num = len(self.particle)
        # print(num)
        # for i in range(num):
        #     one,another = random.sample(self.particle, 2)   ##然后针对任意的两个粒子进行交叉操作  需要写一个selection粒子的方法
        #     cross_new = crossover(one,another)
        #     if (cross_new.layers != one.layers) and (cross_new.layers != another.layers):
        #         self.particle.append(cross_new)

        ###把GA和fit结合起来




