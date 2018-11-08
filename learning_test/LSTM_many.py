from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

data_dim = 16
timesteps = 8
num_classes = 10

# Generate dummy training data
#随机生成1000个输入数据，格式[ [[date_dim个元素], [date_dim个元素],... ],  [同前一个元素] , ... ]
x_train = np.random.random((1000, timesteps, data_dim))
#随机生成1000个标签数据，格式[ [10个float元素],  [10个float元素], ... ]
y_train = np.random.random((1000, num_classes))

# Generate dummy validation data
x_val = np.random.random((100, timesteps, data_dim))
y_val = np.random.random((100, num_classes))

# expected input data shape: (batch_size, timesteps, data_dim)
#创建一个Sequential
model = Sequential()

#第一层，输入32个单元，输入数据格式，
#return_sequences：布尔值，默认False，控制返回类型。若为True则返回整个序列，否则仅返回输出序列的最后一个输出
#input_shape输入格式，两阶数组
model.add(LSTM(32, return_sequences=True,
               input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
#第二层，32个单元
model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
#第三层，32个单元，并且返回输出序列的最后一个输出，return_sequences=False
model.add(LSTM(32))  # return a single vector of dimension 32

#第四层，Dense，从32转换为10，激活函数softmax【格式同标签】
model.add(Dense(10, activation='softmax'))

#编译模型，优化器为rmsprop，监控字段metrics=['accuracy']，刻在model.evaluate时获得
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

#执行训练，validation_data=(x_val, y_val)，每一步都用这批数据来校验准确率
model.fit(x_train, y_train,
          batch_size=64, epochs=5,
          validation_data=(x_val, y_val))