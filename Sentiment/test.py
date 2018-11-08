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

#Config
train_num = 75000
input_data_X_size = 20
batch_size = 64
dir_path=os.getcwd()

model = keras.models.load_model(dir_path + '/qinggan.save.20181009')

l = [
	'商品很好，样子很不错',
	'很满意，送的也很快',
	'物流太慢了',
	'垃圾，非常差',
	'鼠标我觉得眼睛\\\\ud83d\\\\udc40可以发个光，中间的白点有点我涂黑了',
	'火影忍者',
	'不开心，悲伤',
	'赞，哈哈哈',
	'我不喜欢这个商品',
	'我喜欢这个商品',
	'我不得不说很好',
	'做得不够好',
	'金主大大一言不发给了个好评',
	'金主大大一言不发给了个差评',
	'不要怕，只是技术性调整',
	'我有点喜欢这个商品',
	'不是很喜欢，但是勉强可用吧',
	'我没有喜欢它'
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

k = 0
for item in res :
	print("--------------")
	print(l[k])
	print(round(item[0], 3))
	k += 1	



