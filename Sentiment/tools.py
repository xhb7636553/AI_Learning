#encoding=utf-8
import jieba
import sys
import re
import collections
import codecs

txt_file = '/home/user_00/keras/qinggan/word2vec/word_dict.txt'
dictionary = dict()
reversed_dictionary = dict()

def word_action(word) :
	string = re.sub("[\s+’!\"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+",'', word)
	seg_list = jieba.cut(string)
	return list(seg_list)

def sentence_to_index(seg_list) :
	#分词结果list变为index模式，为训练做准备
	index_list = []
	for word in seg_list :
		if word in dictionary:
			index_list.append(int(dictionary[word]))
		else :
			index_list.append(0)
	"""
	print("Input:")
	print(seg_list)
	print("Onput:")
	print(index_list)
	"""
	return index_list

def string2index(string) :
	return sentence_to_index(word_action(string))


i = 0
with open(txt_file,'r',encoding='UTF-8') as f:
	for line in f:
		arr = line.strip().split("|")
		word = arr[0]
		index = arr[1]
		dictionary[word] = int(index)
		i += 1
			
print("Read Data File Finish")
reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
word_num = len(dictionary)
"""

l = [
	'嗨呀  怎么这么小啊  有没有大本的啊  这么小用着不习惯啊 价格没关系啊',
	'很满意，送的也很快，火影忍者',
	'一般般，没有图片上好看。'
]

for j in l :
	#print(j)
	print(string2index(j))
"""
