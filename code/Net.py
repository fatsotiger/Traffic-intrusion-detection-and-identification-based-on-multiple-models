import time
from idlelib import history

start = time.time()
import tensorflow
# from tensorflow.keras.models import Sequential  #序贯模型
# from tensorflow.keras.layers import Dense    #全连接层
# from tensorflow.keras.layers import Dropout  #随机失活层
# from tensorflow.keras.layers import Flatten  #展平层，从卷积层到全连接层必须展平
# from tensorflow.keras.layers import Conv1D   #卷积层
# from tensorflow.keras.layers import MaxPooling1D  #最大值池化
from keras.models import Sequential  #序贯模型
from keras.layers import Dense    #全连接层
from keras.layers import Dropout  #随机失活层
from keras.layers import Flatten  #展平层，从卷积层到全连接层必须展平
from keras.layers import Conv1D   #卷积层
from keras.layers import MaxPooling1D  #最大值池化
import pandas as pd
from keras import backend as k
#from sklearn.cross_validation import train_test_split #随机划分为训练子集和测试子集
# from tensorflow.keras.utils import plot_model
from keras.utils.vis_utils import plot_model
#from keras.optimizers import SGD
import matplotlib.pyplot as plt


batch_size = 128  #一批训练样本128张图片
num_classes = 40  #有40个类别
epochs = 12   #一共迭代12轮

x_train = pd.read_csv('E:\\大三上\\数据挖掘\\课设\\data\\train_x.csv',header=None).values
y_train = pd.read_csv('E:\\大三上\\数据挖掘\\课设\\data\\train_y.csv',header=None).values
x_test = pd.read_csv('E:\\大三上\\数据挖掘\\课设\\data\\test_x.csv',header=None).values
y_test = pd.read_csv('E:\\大三上\\数据挖掘\\课设\\data\\test_y.csv',header=None).values

# 从训练集中手动指定验证集
# x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, test_size=0.15, random_state=2)


if k.image_data_format() == 'channels_first': #维度序列
   x_train = x_train.reshape(x_train.shape[0], 1, 41)
   x_test = x_test.reshape(x_test.shape[0], 1, 41)
   # x_dev = x_dev.reshape(x_dev.shape[0], 1, 41)
   input_shape = (1, 41)
else:
   x_train = x_train.reshape(x_train.shape[0], 41, 1)
   x_test = x_test.reshape(x_test.shape[0], 41, 1)
   # x_dev = x_dev.reshape(x_dev.shape[0], 41, 1)
   input_shape = (41, 1)


model = Sequential()  # sequential序贯模型:多个网络层的线性堆叠

# # 输出的维度（卷积滤波器的数量）filters=32；1D卷积窗口的长度kernel_size=3；激活函数activation   模型第一层需指定input_shape：
# model.add(Conv1D(32, 3, activation='relu',input_shape=input_shape))  # data_format默认channels_last
# model.add(MaxPooling1D(pool_size=(2))) # 池化层：最大池化  池化窗口大小pool_size=2
# model.add(Flatten())  # 展平一个张量，返回一个调整为1D的张量


# -----BP-全连接层----
model.add(Dense(64, input_dim=41, kernel_initializer='uniform', activation='relu',name='hidden_layer1'))  #指定第一层输入维度 input_dim 来隐含的指定输入数据的 shape。其他层的shape框架会自动推导
model.add(Dense(64, kernel_initializer='uniform', activation='relu',name='hidden_layer2'))     #第一个参数units:该层的输出维度。kernel_initializer:权值初始化的方法
model.add(Dropout(0.25))  # 需要丢弃的输入比例=0.25    dropout正则化-减少过拟合
model.add(Dense(40, kernel_initializer='uniform', activation='softmax',name='output_layer'))
# ---------------
# model.add(Dense(128, activation='relu',name='fully_connected')) #全连接层
# model.add(Dense(num_classes, activation='softmax',name='softmax'))

# 编译，损失函数:多类对数损失，用于多分类问题， 优化函数：adadelta， 模型性能评估是准确率
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
# 运行 ， verbose=1输出进度条记录      epochs训练的轮数     batch_size:指定进行梯度下降时每个batch包含的样本数
model.fit(x_train, y_train, batch_size= batch_size, epochs=epochs, verbose=1)
history = model.fit(x_train, y_train, batch_size= batch_size, epochs=epochs, verbose=0, validation_data=(x_train, y_train))




#模型的训练损失（loss）和验证损失（val_loss），以及训练准确率（acc）和验证准确率（val_acc）可以使用绘图代码绘制出来
def smooth_curve(points,factor=0.8): #定义使曲线变得平滑
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs,loss, 'bo', label = 'Training loss')
plt.plot(epochs,val_loss, 'b', label = 'Validation loss')
plt.title('Training and validatio loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.figure()
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc,'bo', label = 'Training acc')
plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
plt.title('Training and validatio accuracy')
plt.legend()
plt.show()



# 将测试集输入到训练好的模型中，查看测试集的误差
score = model.evaluate(x_test, y_test, verbose=0,batch_size= batch_size)
print('Test loss:', score[0])
print('Test accuracy: %.2f%%' % (score[1] * 100))

#运行的时间
stop = time.time()
print(str(stop-start) + "秒")

# 神经网络可视化
plot_model(model, to_file='D:/model.png',show_shapes=True)

#输出模型各层参数情况
model.summary()
