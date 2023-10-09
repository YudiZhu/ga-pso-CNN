import numpy as np
from itertools import zip_longest
from random import choice

def add_conv(layers,max_out_ch,conv_kernel,in_ou):
    out_channel = np.random.randint(max(3,in_ou),max_out_ch) ##随机输出节点数
    #conv_kernel = np.random.randint(3,conv_kernel)  ##随机选择卷积核大小
    conv_kernel = choice([3, 5, 7])

    layers.append({"type":"conv","ou_c":out_channel,"kernel":conv_kernel})

    return layers

def add_pool(layers,num_pool_layers,max_pool_layers):
    pool_layers = num_pool_layers

    if pool_layers < max_pool_layers:  ##判断池化层数量？
        random_pool = np.random.rand()  ##随机出0到1之间的数用来判断池化类型
        pool_layers += 1
        if random_pool < 0.5:
            layers.append({"type":"max_pool","ou_c":-1,"kernel":2})
        else:
            layers.append({"type":"avg_pool","ou_c":-1,"kernel":2})

    return layers, pool_layers

def add_fc(layers,max_fc_neurons):
    layers.append({"type":"fc","ou_c":np.random.randint(5,max_fc_neurons),"kernel":-1})

    return layers

def differenceConvPool(p1,p2):
    ##卷积池化层差异计算方法
    diff = []

    for comb in zip_longest(p1,p2): #按长度较长的打包，短的会填充None,实际上是在一层一层地做比较
        if comb[0] != None and comb[1] != None: ##在两个网络有对应层的情况下
            if comb[0]["type"] == comb[1]["type"]:
                diff.append({"type": "keep"})  ##如果两个网络层类型相同，则保留
            else:
                diff.append(comb[0])  ##不同，则添加左边网络的结构

        elif comb[0] != None and comb[1] == None:  #如果两边的层级结构对不上了，且是右边的网络结束了，则添加左边的网络层级
            diff.append(comb[0])

        elif comb[0] == None and comb[1] != None:  ##如果是左边网络结束了而右边网络还有，就添加remove操作
            diff.append({"type": "remove"})

    return diff

def differenceFC(p1,p2):
    ##全连接层差异计算方法
    diff = []

    # Compute the difference from the end to the begin
    for comb in zip_longest(p1[::-1], p2[::-1]):  ##[::-1]顺序取反操作
        if comb[0] != None and comb[1] != None:
            diff.append({"type": "keep_fc"})
        elif comb[0] != None and comb[1] == None:
            diff.append(comb[0])
        elif comb[0] == None and comb[1] != None:
            diff.append({"type": "remove_fc"})

    diff = diff[::-1]  ##需要保留输出层

    return diff


def computeDifference(p1,p2):
    diff = []
    ##首先找到网络结构中全连接层从什么地方开始
    '''
    next函数：返回迭代器的下一个项目，p1是一个迭代器对象，返回index和layers，这里的next获取的是全连接层出现时下一层的index
    如 pool-pool-fc-fc，返回的数就是 2
    '''
    ##这里是分别找到两个网络第一次出现全连接层的位置
    p1fc_idx = next((index for (index,d) in enumerate(p1) if d["type"]=="fc"))
    p2fc_idx = next((index for (index, d) in enumerate(p2) if d["type"] == "fc"))

    ##计算差别程度？只计算卷积池化的部分
    diff.extend(differenceConvPool(p1[0:p1fc_idx],p2[0:p2fc_idx]))  ##计算卷积池化层的区别（全连接之前）
    ##计算全连接层之间的差异
    diff.extend(differenceFC(p1[p1fc_idx:],p2[p2fc_idx:]))

    keep_all_layers = True  #判断两个网络结构是否完全一致
    for i in range(len(diff)):
        if diff[i]["type"] != "keep" or diff[i]["type"] != "keep_fc":
            keep_all_layers = False
            break

    return diff, keep_all_layers

def velocityConvPool(diff_pBest, diff_gBest, Cg):
    ##计算速度？Cg是什么
    vel = []

    for comb in zip_longest(diff_pBest,diff_gBest):
        if np.random.uniform() <= Cg:  ##随机赋值？
            if comb[1] != None:
                vel.append(comb[1])
            else:
                vel.append({"type": "remove"})
        else:
            if comb[0] != None:
                vel.append(comb[0])
            else:
                vel.append({"type":"remove"})

    return vel

def velocityFC(diff_pBest, diff_gBest, Cg):
    vel = []

    for comb in zip_longest(diff_pBest[::-1], diff_gBest[::-1]):
        if np.random.uniform() <= Cg:
            if comb[1] != None:
                vel.append(comb[1])
            else:
                vel.append({"type": "remove_fc"})
        else:
            if comb[0] != None:
                vel.append(comb[0])
            else:
                vel.append({"type": "remove_fc"})

    vel = vel[::-1]

    return vel


def computeVelocity(gBest, pBest, p, Cg):  ##分别计算当前位置和gbest与pbest之间的差别
    diff_pBest, keep_all_pBest = computeDifference(pBest,p)
    diff_gBest, keep_all_gBest = computeDifference(gBest,p)

    velocity = []
    ##首先判断结构是否相同
    if keep_all_pBest == True and keep_all_gBest ==True:  ##当前网络结构与pbest及gbest均完全一致
        for i in range(len(gBest)): ##gbest是指gbest.layers
            if np.random.uniform() <= Cg:
                velocity.append(gBest[i])
            else:
                velocity.append(pBest[i])  ##根据cg随机进行层级选择
    else:
        ##找到全连接层的位置
        dp_fc_idx = next((index for (index,d) in enumerate(diff_pBest) if d["type"]=="fc" or d["type"]=="keep_fc" or d["type"]=="remove_fc"))
        dg_fc_idx = next((index for (index,d) in enumerate(diff_gBest) if d["type"] == "fc" or d["type"] == "keep_fc" or d["type"] == "remove_fc"))

        ##计算velocity
        velocity.extend(velocityConvPool(diff_pBest[0:dp_fc_idx],diff_gBest[0:dg_fc_idx],Cg))
        # Compute the velocity between the fully connected layers
        velocity.extend(velocityFC(diff_pBest[dp_fc_idx:], diff_gBest[dg_fc_idx:], Cg))

    return velocity

def updateFC(p, vel):
    ##全连接层更新
    new_p = []

    for comb in zip_longest(p[::-1],vel[::-1]):
        if comb[1]["type"] != "remove_fc":
            if comb[1]["type"] == "keep_fc":
                new_p.append(comb[0])
            else:
                new_p.append(comb[1])
    new_p = new_p[::-1]

    return new_p

def updateConvPool(p, vel):
    new_p = []

    for comb in zip_longest(p, vel):
        if comb[1]["type"] != "remove":
            if comb[1]["type"] == "keep":
                new_p.append(comb[0])
            else:
                new_p.append(comb[1])

    return new_p

def updateParticle(p,velocity):
    new_p = []
    dp_fc_idx = next((index for (index, d) in enumerate(p) if d["type"] == "fc"))
    dg_fc_idx = next((index for (index, d) in enumerate(velocity) if d["type"] == "fc" or d["type"] == "keep_fc" or d["type"] == "remove_fc"))

    # Update only convolution and pooling layers
    new_p.extend(updateConvPool(p[0:dp_fc_idx], velocity[0:dg_fc_idx]))

    # Update only fully connected layers
    new_p.extend(updateFC(p[dp_fc_idx:], velocity[dg_fc_idx:]))

    return new_p
