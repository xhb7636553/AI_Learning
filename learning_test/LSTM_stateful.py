#采用stateful LSTM的相同模型
#stateful LSTM的特点是，在处理过一个batch的训练数据后，其内部状态（记忆）会被作为下一个batch的训练数据的初始状态。状态LSTM使得我们可以在合理的计算复杂度内处理较长序列

from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

data_dim = 16
timesteps = 8
num_classes = 10
batch_size = 32

# Generate dummy training data
#随机生成batch_size * 10个输入数据，每个数组拥有timesteps个向量，每个向量拥有data_dim float元素，格式[ [[date_dim个元素], [date_dim个元素],... ],  [同前一个元素] , ... ]
x_train = np.random.random((batch_size * 10, timesteps, data_dim))
y_train = np.random.random((batch_size * 10, num_classes))

#随机生成batch_size * 10个输入数据，每个数组拥有timesteps个向量，每个向量拥有data_dim float元素，格式[ [[date_dim个元素], [date_dim个元素],... ],  [同前一个元素] , ... ]
x_val = np.random.random((batch_size * 3, timesteps, data_dim))
y_val = np.random.random((batch_size * 3, num_classes))

# Expected input batch shape: (batch_size, timesteps, data_dim)
# Note that we have to provide the full batch_input_shape since the network is stateful.
# the sample of index i in batch k is the follow-up for the sample i in batch k-1.

#创建模型
model = Sequential()
#第一层，32个单元，return_sequences返回全部序列，stateful状态传递，batch_input_shape=(batch_size, timesteps, data_dim)
model.add(LSTM(32, return_sequences=True, stateful=True,
               batch_input_shape=(batch_size, timesteps, data_dim)))

#第二层，LSTM，return_sequences返回全部序列，stateful状态传递
model.add(LSTM(32, return_sequences=True, stateful=True))
#第三层，LSTM，return_sequences返回全部序列的最后一个，stateful状态传递
model.add(LSTM(32, stateful=True))
#第四层，输出层，32转换为10单元，Dense层，激活函数softmax，输出[0.234,0.234,0.2334,...]，共10个float数字
model.add(Dense(10, activation='softmax'))

#模型编译，loss='categorical_crossentropy'，optimizer='rmsprop'
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

#训练程序，shuffle=False，因为有状态传递，不进行shuffle打散
model.fit(x_train, y_train,
          batch_size=batch_size, epochs=5, shuffle=False,
          validation_data=(x_val, y_val))