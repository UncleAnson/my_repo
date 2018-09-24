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
            labels += label.decode('utf-8')
    return labels


# 优先转文件里的字符到向量
def get_ch_label_v(txt_file, word_num_map, txt_label=None):
    words_size = len(word_num_map)
    to_num = lambda word: word_num_map.get(word, words_size) # 在词表以外（unk）则以words_size表示
    if txt_file!=None:
        txt_label = get_ch_label(txt_file)
    labels_vector = list(map(to_num, txt_label))
    return labels_vector


train_data = get_ch_label(train_file)
print('loaded training data...')

count = Counter(train_data)
words = sorted(count)
words_size = len(words)

word_num_map = dict(zip(words, range(words_size))) # 给词表里的每个词/字编序号，从0开始

print('字表大小：', words_size)
wordlabel = get_ch_label_v(train_file, word_num_map) # 将文件中的文字转换为序号表示，unk以len表示


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
x1 =  tf.reshape(x, [-1, n_input])
x2 = tf.split(x1, n_input, 1)

rnn_cell = rnn.MultiRNNCell([rnn.LSTMCell(n_hidden_1), rnn.LSTMCell(n_hidden_2), rnn.LSTMCell(n_hidden_3)])

# 通过rnn得到输出
outputs, states = rnn.static_rnn(rnn_cell, x2, dtype=tf.float32)

# 通过全连阶层输出指定维度
pred = tf.contrib.layers.fully_connected(outputs[-1], words_size, activation_fn=None)

# 定义损失函数，即优化器
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=wordy))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# 模型评估
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(wordy, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

savedir = "./log/rnnword"
saver = tf.train.Saver(max_to_keep=1)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    step = 0
    offset = random.randint(0, n_input+1)
    end_offset = n_input + 1
    acc_total = 0
    loss_total = 0

    kpt = tf.train.latest_checkpoint(savedir)
    print("kpt：", kpt)
    startepo = 0

    if kpt!=None:
        saver.restore(session, kpt)
        ind = kpt.find("-")
        startepo = int(kpt[ind+1: ])
        print(startepo)
        step = startepo

    while step<training_iters:
        if offset>(len(train_data)-end_offset):
            offset = random.randint(0, n_input+1)
        inwords = [[wordlabel[i]]for i in range(offset, offset+n_input)]
        inwords = np.reshape(np.array(inwords), [-1, n_input, 1])

        out_onehot = np.zeros([words_size], dtype=float)
        out_onehot[wordlabel[offset+n_input]] = 1.0
        out_onehot = np.reshape(out_onehot, [1, -1])

        _, acc, lossval, onehot_pred = session.run([optimizer, accuracy, loss, pred], feed_dict={x: inwords, wordy: out_onehot})
        loss_total += lossval
        acc_total += acc

        if (step+1)%display_step == 0:
            print("Iter= ", str(step+1), ", Average Loss=", " {:.6f}".format(loss_total/display_step), ", Average Accuracy= ", "{:.2f}%".format(100*acc_total/display_step))
            acc_total = 0
            loss_total = 0

            in2 = [words[wordlabel[i]] for i in range(offset, offset+n_input)] # 前面n_input个
            out2 = words[wordlabel[offset+n_input]] # 下一个实际是什么
            onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
            print("%s - [%s] vs [%s]"%(in2, out2, words[onehot_pred_index]))
            saver.save(session, savedir+"rnnwordtest.cpkt", global_step=step)
        step += 1
        offset += (n_input+1)
    print("finished！")
    saver.save(session, savedir + "rnnwordtest.cpkt", global_step=step)
    print("Elapsed time: ", elapsed(time.time()-start_time))





# 运行模型生成句子
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    while True:
        prompt = "请输入%s个字："%n_input
        sentence = input(prompt)
        inputword = sentence.strip()

        if len(inputword)!=n_input:
            print("你输入的字符长度为：",len(inputword),"，请输入%s个字"%n_input)
            continue
        # try:
        inputword = get_ch_label_v(None, word_num_map, inputword)

        for i in range(32):
            keys = np.reshape(np.array(inputword), [-1, n_input, 1])
            onehot_pred = session.run(pred, feed_dict={x: keys})
            onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
            sentence = "%s%s"%(sentence, words[onehot_pred_index])
            inputword = inputword[1: ]
            inputword.append(onehot_pred_index)
        print(sentence)
        # except:
        #     print('该字我还没学会')
