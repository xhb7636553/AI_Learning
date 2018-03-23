#加载keras基础框架
import keras

#加载keras基础框架中的组件，方面使用，缩短命名空间
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

#加载数据处理组件numpy，重命名为np
import numpy as np

# 生产假数据
#random产生1000个输入数据，每个有20个纬度的数组，格式：[[1...20], [1...20], [1...20]]
x_train = np.random.random((1000, 20))
#random产生1000个输入的标签数据，每个有一个纬度，通过keras.utils.to_categorical转化成one-hot key
#格式：[[0,0,..,1,0,0],[0,1,..,0,0,0],[1,0,..,0,0,0]]
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)

#random产生100个输入数据，格式同输入
x_test = np.random.random((100, 20))
#random产生10个输入数据，格式同输入的标签
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

#创建Sequential模型
model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.

#Dense 2D模型，单纬度模型，激活函数relu，输入纬度input_dim，20个，输入层64个单元
model.add(Dense(64, activation='relu', input_dim=20))
#数据丢失50%，防止过拟合
model.add(Dropout(0.5))
#中间隐含层64个单元
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
#输出层，将64个单元映射回到10个单元，激活函数softmax，将指分布为0-1，概率化，最终输出的是一个10个纬度的one-hot Key
model.add(Dense(10, activation='softmax'))

#定义优化函数，lr=learning rate学习率
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

#编译模型，设置loss函数，优化器，关注的变量列表
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

#模型训练，输入x_train和y_train，每一批数据随机轮20次，每批数据有128个记录数
model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)

#验证模型的准确结果程度x_test和y_test是测试集合
score = model.evaluate(x_test, y_test, batch_size=128)
