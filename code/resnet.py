
import keras
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import layers, Sequential
import matplotlib.pyplot as plt
from tensorflow.python.keras.utils.vis_utils import plot_model


class BasicBlock(layers.Layer):
    def __init__(self,filter_num,stride=1):
        super(BasicBlock,self).__init__()
        # -----------
        self.conv1=layers.Conv1D(filter_num,kernel_size=3,strides=stride,padding='same')
        self.bn1=layers.BatchNormalization()
        self.relu=layers.Activation('relu')
        # -----------
        self.conv2=layers.Conv1D(filter_num,kernel_size=3,strides=1,padding='same')
        self.bn2=layers.BatchNormalization()
        # -----保持残差数据X与处理数据维度相同-----
        if stride != 1:
            self.dowmsample=Sequential()
            self.dowmsample.add(layers.Conv1D(filter_num,1,strides=stride))
            self.dowmsample.add(layers.BatchNormalization())
        else:
            self.dowmsample=lambda x:x
        # ----------
        self.stride=stride

    def call(self,input,training=None):

        # input [b,h,w,c]
        # 跳跃数据x
        residual=self.dowmsample(input)
#
        conv1=self.conv1(input)
        bn1=self.bn1(conv1)
        relu1=self.relu(bn1)

        conv2=self.conv2(relu1)
        bn2 = self.bn2(conv2)

        add=layers.add([bn2,residual]) # f(x) + x
        output=self.relu(add)
        return output

class ResNet(keras.Model):
    # layer_dims =>ResNet18 [2,2,2,2]
    # num_classes 预设40类
    def __init__(self,layer_dims,num_classes=18):
        super(ResNet,self).__init__()
        # 预处理层
        self.stem=Sequential([
            layers.Conv1D(64,3,strides=1),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool1D(pool_size=2,strides=1,padding='same'),
        ])
        # 残差层
        self.layer1=self.build_resblock(64,layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1],stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2],stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3],stride=2)

        # 残差层output=[b,512,h,w]
        # GlobalAvgPool2D => [b,512,1,1]
        self.avgpool=layers.GlobalAvgPool1D()
        # 分类全连接层
        self.fc=layers.Dense(num_classes)

    def call(self,inputs,training=None):
        x = self.stem(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #[b,c]
        x=self.avgpool(x)
        #[b,100]
        x=self.fc(x)
        return  x

    def build_resblock(self,filter_num,blocks,stride=1):
        res_block=keras.Sequential()
        # 可能下采样
        res_block.add(BasicBlock(filter_num,stride))

        for _ in range(1,blocks):
            res_block.add(BasicBlock(filter_num, stride=1))

        return res_block

def resnet18():
    # resnet34 => [3,4,6,3]
    return ResNet([2,2,2,2])

from keras import backend as k

model=resnet18()


batch_size = 128  #一批训练样本128张图片
num_classes = 40  #有40个类别
epochs = 3  #一共迭代12轮


x_train = pd.read_csv('train_x.csv',header=None).values
y_train = pd.read_csv('train_y.csv',header=None).values
x_test = pd.read_csv('test_x.csv',header=None).values
y_test = pd.read_csv('test_y.csv',header=None).values

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

model.build(input_shape=[None,41,1])
# model=model(data)


# 编译，损失函数:多类对数损失，用于多分类问题， 优化函数：adadelta， 模型性能评估是准确率
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
# 运行 ， verbose=1输出进度条记录      epochs训练的轮数     batch_size:指定进行梯度下降时每个batch包含的样本数
model.fit(x_train, y_train, validation_split=0.2,batch_size= batch_size, epochs=epochs, verbose=1)
#model.fit( x_train, y_train,batch_size= batch_size, validation_data=(x_test, y_test), epochs=epochs, verbose=1)



# 将测试集输入到训练好的模型中，查看测试集的误差
score = model.evaluate(x_test, y_test, verbose=0,batch_size= batch_size)
print('Test loss:', score[0])
print('Test accuracy: %.2f%%' % (score[1] * 100))
model.summary()

plot_model(model, to_file='D:/modelREsNet.png',show_shapes=True)

