#MLP(Multi-layer Perceptron，多层神经网络)的二分类

import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Generate dummy data
#random产生1000个输入数据，每个有20个纬度的数组，格式：二阶数组[[0.837394,...,0.3734], [0.837394,...,0.3734], [0.837394,...,0.3734]]，每数字为0-1的浮点数
x_train = np.random.random((1000, 20))
#random产生取值范围为2以下的int整数（其实就是0或者1），数量1000个1个纬度的数组，格式：[[0],[1],[1],[0]]
y_train = np.random.randint(2, size=(1000, 1))
#random产生100个输入数据，每个有20个纬度的数组，格式：二阶数组[[0.837394,...,0.3734], [0.837394,...,0.3734], [0.837394,...,0.3734]],每数字为0-1的浮点数
x_test = np.random.random((100, 20))
#random产生取值范围为2以下的int整数（其实就是0或者1），数量100个1个纬度的数组，格式：[[0],[1],[1],[0]]
y_test = np.random.randint(2, size=(100, 1))

#创建一个Sequential对象
model = Sequential()
#添加一层，2D层，64个单元，输入纬度是一阶20纬度的输入向量input_dim=20（格式[0.3,0.7383,..,0.23834]），激活函数relu
model.add(Dense(64, input_dim=20, activation='relu'))
#抛弃部分数据
model.add(Dropout(0.5))
#添加一层，2D层，64个单元，输入纬度是一阶20纬度的输入向量input_dim=20（格式[0.3,0.7383,..,0.23834]），激活函数relu
model.add(Dense(64, activation='relu'))
#抛弃部分数据
model.add(Dropout(0.5))
#输出层，2D层，从64个单元映射回1个单元，激活函数sigmoid，变为0-1
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)
print(score)
