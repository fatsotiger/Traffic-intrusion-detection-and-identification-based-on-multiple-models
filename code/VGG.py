import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential
import numpy as np
import matplotlib.pyplot as plt
from numpy import float32, float64
import pandas as pd

conv_layers = [  # 5 units of conv + max pooling
    # unit 1
    layers.Conv1D(64, kernel_size=3, padding="same", activation=tf.nn.relu),
    layers.Conv1D(64, kernel_size=3, padding="same", activation=tf.nn.relu),
    layers.MaxPool1D(pool_size=2, strides=2, padding='same'),

    # unit 2
    layers.Conv1D(128, kernel_size=3, padding="same", activation=tf.nn.relu),
    layers.Conv1D(128, kernel_size=3, padding="same", activation=tf.nn.relu),
    layers.MaxPool1D(pool_size=2, strides=2, padding='same'),

    # unit 3
    layers.Conv1D(256, kernel_size=3, padding="same", activation=tf.nn.relu),
    layers.Conv1D(256, kernel_size=3, padding="same", activation=tf.nn.relu),
    layers.MaxPool1D(pool_size=2, strides=2, padding='same'),

    # unit 4
    layers.Conv1D(512, kernel_size=3, padding="same", activation=tf.nn.relu),
    layers.Conv1D(512, kernel_size=3, padding="same", activation=tf.nn.relu),
    layers.MaxPool1D(pool_size=2, strides=2, padding='same'),

    # unit 5
    layers.Conv1D(512, kernel_size=3, padding="same", activation=tf.nn.relu),
    layers.Conv1D(512, kernel_size=3, padding="same", activation=tf.nn.relu),
    layers.MaxPool1D(pool_size=2, strides=2, padding='same')
]


batch_size = 64  #一批训练样本128张图片
num_classes = 40  #有40个类别
epochs = 12#一共迭代12轮

x_train = pd.read_csv('train_x.csv',header=None).values
y_train = pd.read_csv('train_y.csv',header=None).values
x_test = pd.read_csv('test_x.csv',header=None).values
y_test = pd.read_csv('test_y.csv',header=None).values


x_train = x_train.reshape(x_train.shape[0], 41, 1)
x_test = x_test.reshape(x_test.shape[0], 41, 1)
input_shape = (41, 1)




def training():
    # [b, 32, 32, 3] => [b, 1, 1, 512]
    conv_net = Sequential(conv_layers)
    conv_net.add(layers.Flatten())
    conv_net.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))  # 隐藏层
    conv_net.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
    conv_net.add(tf.keras.layers.Dense(40, activation='softmax'))  # 输出层 tf.nn.softmax
    
    conv_net.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
# 运行 ， verbose=1输出进度条记录      epochs训练的轮数     batch_size:指定进行梯度下降时每个batch包含的样本数
    conv_net.fit(x_train, y_train, validation_split=0.2,batch_size= batch_size, epochs=epochs, verbose=1)
    score = conv_net.evaluate(x_test, y_test, verbose=0,batch_size= batch_size)
    print('Test loss:', score[0])
    print('Test accuracy: %.2f%%' % (score[1] * 100))
    conv_net.summary()

training()

