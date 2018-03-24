#类似VGG的卷积神经网络，例子中是分类问题，假设将图像内容分为10个类（ont-hot key）
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

# Generate dummy data
#因为CNN卷积神经网络的耗时比较长，所以生成样例数目少一些
#生成100个输入数据，再100个，再100个，3纬度的一阶向量[0.23, 0.234, 0.9833]，备注：二阶向量表示里面含有多个一阶向量
x_train = np.random.random((100, 100, 100, 3))
#生成100个标签，randint生成10以下的int（0-9）,然后通过to_categorical转化成10个分类的ont-hot key，100个，[0,0,0,1,0,...,0]
y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

#生成20个，输入
x_test = np.random.random((20, 100, 100, 3))
#生成20个标签，同输入
y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)


model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# 假设输入的是一个100*100像素的图片，并且每个像素拥有3个情况，代表100*100的彩色RGB图像
# this applies 32 convolution filters of size 3x3 each.

# 添加输入层，卷积2D，32个输出单元（filters：卷积核的数目（即输出的维度））
#(3, 3)，卷积核的长和宽是都是3。单个整数或由两个整数构成的list/tuple，卷积核的宽度和长度。如为单个整数，则表示在各个空间维度的相同长度。
# input_shape=(100, 100, 3)，代表100*100的彩色RGB图像，Conv2D通常用在图片100*100类型像素场景中
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
#为空域信号施加最大值池化，pool_size：整数或长为2的整数tuple，代表在两个方向（竖直，水平）上的下采样因子，如取（2，2）将使图片在两个维度上均变为原长的一半。为整数意为各个维度值相同且为该数字。
model.add(MaxPooling2D(pool_size=(2, 2)))
#dropout，丢失部分输入，防止过拟合
model.add(Dropout(0.25))

#再加一层神经网络
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#Flatten层，将多纬的数据压平，变成一纬（高阶变一阶），即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。
model.add(Flatten())
#此时数据已经变为一阶向量，定义输出单元为256，Dense就是常用的全连接层，所实现的运算是output = activation(dot(input, kernel)+bias)，其中activation是逐元素计算的激活函数，kernel是本层的权值矩阵，bias为偏置向量，只有当use_bias=True才会添加。
model.add(Dense(256, activation='relu'))
#dropout，丢失部分输入，防止过拟合
model.add(Dropout(0.5))
#输出变换为10单元（10个数字的一阶向量），激活函数为softmax，变换为0-1
model.add(Dense(10, activation='softmax'))

#定义梯度下降函数
#lr : 学习率 
# momentum : 梯度下降中一种常用的加速技术，控制参数更新时每次的下降幅度 
# decay ：每次更新时学习率衰减量 
# nesterov ：是否应用Nesterov momentum
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

#模型编译，并且定义loss损失函数，优化方法为sgd
model.compile(loss='categorical_crossentropy', optimizer=sgd)

#训练和执行，输入为x_train，标签为y_train，批次大小32，每批数据轮10次（自动shuffle）
model.fit(x_train, y_train, batch_size=32, epochs=10)

#预测数据成果
#本函数返回一个测试误差的标量值（如果模型没有其他评价指标），或一个标量的list（如果模型还有其他的评价指标）。model.metrics_names将给出list中各个值的含义。
score = model.evaluate(x_test, y_test, batch_size=32)
print(score)
