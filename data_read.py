import cv2
import numpy as np
# from sklearn.model_selection import train_test_split
# img_list = np.load("img_list.npy")
# labels = np.load("label_list.npy")
# x_train, x_test, y_train, y_test = train_test_split(img_list, labels, test_size=0.3,random_state=43)

def get_im_cv2(path,color_type=1, normalize=True):
    '''
    参数：
        paths：要读取的图片路径列表
        img_rows:图片行
        img_cols:图片列
        color_type:图片颜色通道
    返回:
        imgs: 图片数组
    '''
    # Load as grayscale
    if color_type == 1:
        img = cv2.imread(path.replace('\\','/'))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif color_type == 3:
        img = cv2.imread(path.replace('\\','/'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print(img)
    # Reduce size
    resized = cv2.resize(img, (128,128))
    if normalize:
        resized = resized.astype('float32')
        resized /= 127.5
        resized -= 1. ##归一化

    resized = resized*255
    resized = resized.astype('uint8')
    return np.array(resized).reshape(128, 128, color_type)

# def get_train_batch(X_train, y_train, batch_size):
#     '''
#     参数：
#         X_train：所有图片路径列表
#         y_train: 所有图片对应的标签列表
#         batch_size:批次
#         img_w:图片宽
#         img_h:图片高
#         color_type:图片类型
#         is_argumentation:是否需要数据增强
#     返回:
#         一个generator，x: 获取的批次图片 y: 获取的图片对应的标签
#     '''
#     while 1:
#         for i in range(0, len(X_train), batch_size):
#             x = get_im_cv2(paths=X_train[i:i+batch_size],color_type=3,normalize=True)
#             y = y_train[i:i+batch_size]
#             # 最重要的就是这个yield，它代表返回，返回以后循环还是会继续，然后再返回。就比如有一个机器一直在作累加运算，但是会把每次累加中间结果告诉你一样，直到把所有数加完
#             yield(np.array(x), np.array(y))
#
# data_list = np.load('data_uint.npy')
# print(data_list[0].shape)
# data = []
# for each in data_list:
#     data.append(get_im_cv2(each))
#
# np.save('data_uint.npy',data)


# ##模型编译
# from sklearn import datasets
# import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.utils import to_categorical
# from keras.models import model_from_json
# from tensorflow.keras.optimizers import SGD
#
# # 从json文件中加载模型
# with open('C:/Users/Yudi Zhu/Desktop/best-gBest-model.json', 'r') as file:
#     model_json = file.read()
#
# new_model = model_from_json(model_json)
# new_model.load_weights('C:/Users/Yudi Zhu/Desktop/best-gBest-weights.h5')
#
# # 编译模型！！！！
# adam = SGD(lr=0.002,decay=0.0)
# # new_model.compile(loss = 'categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
# new_model.compile(loss="sparse_categorical_crossentropy",optimizer=adam,metrics=["accuracy"])
#
# dataset = datasets.load_iris()
# x = dataset.data
# y = dataset.target
# y_labels = to_categorical(y, num_classes=3)
#
# scores = new_model.evaluate(x, y_labels, verbose=0)
# print(new_model.metrics_names[1], scores[1] * 100)
