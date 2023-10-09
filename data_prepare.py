import cv2
import numpy as np
# from sklearn.model_selection import train_test_split
# img_list = np.load("img_list.npy")
# labels = np.load("label_list.npy")
# x_train, x_test, y_train, y_test = train_test_split(img_list, labels, test_size=0.3,random_state=43)

def get_im_cv2(paths,color_type=1, normalize=True):
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
    imgs = []
    for path in paths:
        if color_type == 1:
            img = cv2.imread(path.replace('\\','/'))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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

        imgs.append(resized)

    return np.array(imgs).reshape(len(paths), 128, 128, color_type)

def get_train_batch(X_train, y_train, batch_size):
    '''
    参数：
        X_train：所有图片路径列表
        y_train: 所有图片对应的标签列表
        batch_size:批次
        img_w:图片宽
        img_h:图片高
        color_type:图片类型
        is_argumentation:是否需要数据增强
    返回:
        一个generator，x: 获取的批次图片 y: 获取的图片对应的标签
    '''
    while 1:
        for i in range(0, len(X_train), batch_size):
            #x = get_im_cv2(paths=X_train[i:i+batch_size],color_type=3,normalize=True)
            x = X_train[i:i+batch_size]
            y = y_train[i:i+batch_size]
            # 最重要的就是这个yield，它代表返回，返回以后循环还是会继续，然后再返回。就比如有一个机器一直在作累加运算，但是会把每次累加中间结果告诉你一样，直到把所有数加完
            yield(np.array(x), np.array(y))

# for a,b in get_train_batch(x_train,y_train,32):
#     print(a,b)