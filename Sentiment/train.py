# coding=utf-8
from __future__ import print_function
import keras
from keras.models import Sequential
from keras import layers
import numpy as np
from six.moves import range
import sys
import os
import collections
from keras.utils import to_categorical
import tools
import time

#Config
train_num = 71000
input_data_X_size = 20
batch_size = 64
dir_path=os.getcwd()
comment_file ='train_data/shuffle_avg_comment.txt'
X = []
Y = []

i = 0
with open(comment_file, "r", encoding='utf-8') as f:
	for line in f:
		try:	
			arr = line.strip().split("|")
			comment_string = arr[0]
			y = float(arr[1])
			
			#skip
			if y <= 0.6 and y >= 0.4 :
				continue

			#positive
			if y > 0.6 :
				Y.append([1,0])
			#Negative
			if y < 0.4 :
				Y.append([0,1])


			index_list = tools.string2index(comment_string)[0:input_data_X_size]
			#给末尾不足30位的补0，让输入x都是30 size
			x = np.pad(np.array(index_list), (0, (input_data_X_size - len(index_list))), 'constant')
			X.append(x)
			
			"""		
			print(comment_string)
			print(index_list)
			print(x)
			print(y)			
			"""
			i += 1
			if i % 5000 == 10 :
				print("Run Index:", i)

			if i >= 70000 :
				break
		except Exception as e:
			print(e)
			sys.exit(2)
			pass

Y = np.array(Y)
X = np.array(X)
print(X[0:10])
print(Y[0:10])


n_chunk = len(X) // batch_size
print("n_chunk:", n_chunk)

print(X.shape)
print(Y.shape)

print('Build model...')
words_num = tools.word_num
EMBEDDING_SIZE = 128
HIDDEN_LAYER_SIZE = 64

model = Sequential()
#model.add(layers.embeddings.Embedding(words_num, 64, input_length=26))
model.add(layers.embeddings.Embedding(words_num, EMBEDDING_SIZE, input_length=input_data_X_size))
model.add(layers.LSTM(HIDDEN_LAYER_SIZE, dropout=0.1, return_sequences=True))
#model.add(layers.Dropout(0.1))
model.add(layers.LSTM(64, return_sequences=True))
#model.add(layers.Dropout(0.1))
model.add(layers.Flatten())
model.add(layers.Dense(2)) #[0, 1] or [1, 0]
model.add(layers.Activation('softmax'))
#optimizer=keras.optimizers.Adam(lr=0.001)
model.compile(loss='categorical_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])
model.summary()

model.fit(X, Y, epochs=1, batch_size=64, validation_split=0.05, verbose=2)

model.save(dir_path + '/log/qinggan.save.' + time.strftime("%Y%m%d", time.localtime()))

l = [
	'商品很好，样子很不错',
	'很满意，送的也很快',
	'物流太慢了',
	'垃圾，非常差',
	'鼠标我觉得眼睛\\\\ud83d\\\\udc40可以发个光，中间的白点有点我涂黑了',
]

test_x = []
for j in l :
	index_list = tools.string2index(j)[0:input_data_X_size]
	#给末尾不足30位的补0，让输入x都是30 size
	x = np.pad(np.array(index_list), (0, (input_data_X_size - len(index_list))), 'constant')
	test_x.append(x)
test_x = np.array(test_x)
res = model.predict_on_batch(test_x)
print(res)

