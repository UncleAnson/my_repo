import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import time
from collections import Counter

start_time = time.time()

# 转换时间单位
def elapsed(sec):
    if sec < 60:
        return str(sec) + " sec"
    elif sec < (60 * 60):
        return str(sec / (60)) + " min"
    else:
        return str(sec / (60 * 60)) + " hr"


tf.reset_default_graph()
train_file = "./wordstest.txt"


# 处理多个中文文件
def readalltxt(txt_files):
    labels = []
    for txt_file in txt_files:
        target = get_ch_label(txt_file)
        labels.append(target)
    return labels


# 处理汉字，将多段连接
def get_ch_label(txt_file):
    labels = ""
    with open(txt_file, 'rb') as f:
        for label in f:
            labels += label
    return labels


# 优先转文件里的字符到向量
def get_ch_label_v(txt_file, word_num_map, txt_label=None):
    words_size = len(word_num_map)
    to_num = lambda word: word_num_map.get(word, words_size) # 在词表以外（unk）则以words_size表示
    if txt_label!=None:
        txt_label = get_ch_label(txt_file)
    labels_vector = list(map(to_num, txt_label))
    return labels_vector


train_data = get_ch_label(train_file)
print('loaded training data...')

count = Counter(train_data)
words = sorted(count)
words_size = len(words)

word_num_map = dict(zip(words, range(words_size))) # 给词表里的每个词/字编码

print('字表大小：', words_size)
wordlabel = get_ch_label_v(train_file, word_num_map)


# 构建模型
learning_rate = 0.001
training_iters = 10000
display_step = 1000
n_input = 4

n_hidden_1 = 256
n_hidden_2 = 512
n_hidden_3 = 512

# sx代表输入的连续4个文字
x = tf.placeholder("float", [None, n_input, 1])
# wordy代表一个字，one_hot编码
wordy = tf.placeholder("float", [None, words_size])

# 定义网络结构
