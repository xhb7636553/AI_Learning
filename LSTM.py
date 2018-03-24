from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM

#创建一个Sequential
model = Sequential()
#添加一个输入纬度max_features的，输出也是256的Embedding层，256的Embedding层只能作为模型的第一层
model.add(Embedding(max_features, output_dim=256))
#256变换为128纬度
model.add(LSTM(128))
#dropout
model.add(Dropout(0.5))
#从128变换为1，激活函数sigmoid，变成0-1的float
model.add(Dense(1, activation='sigmoid'))

#编译模型，metrics中的参数会被追加到score的list中
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

#执行
model.fit(x_train, y_train, batch_size=16, epochs=10)

#计算是否符合预期
score = model.evaluate(x_test, y_test, batch_size=16)