import time
from collections import Counter

import numpy as np
from tensorflow.python.ops import ctc_ops

from wav.yuyinutils import get_audio_and_transcriptch, pad_sequences
# from yuyinutils import get_wavs_labels
from wav.yuyinutils import get_wavs_labels_v2
from wav.yuyinutils import sparse_tuple_from
from wav.yuyinutils import sparse_tuple_to_texts_ch, ndarray_to_text_ch

# wav_path = r'D:/data_thchs30/data_thchs30/train'
# label_file = r'D:/data_thchs30/doc/trans/train.word.txt'
# wav_files, labels = get_wavs_labels(wav_path, label_file)

train_data_path = r'D:\data_thchs30\train'
wav_files, labels = get_wavs_labels_v2(train_data_path)

print(wav_files[0], labels[0])
print("wav:", len(wav_files), "label", len(labels))

b_stddev = 0.046875
h_stddev = 0.046875

n_hidden = 1024
n_hidden_1 = 1024
n_hidden_2 = 1024
n_hidden_5 = 1024
n_cell_dim = 1024
n_hidden_3 = 2*1024

keep_dropout_rate = 0.95
relu_clip = 20

# 建立批次获取样本函数

all_words = []
for label in labels:
    all_words += [word for word in label]
counter = Counter(all_words)
words = sorted(counter)
words_size = len(words)
word_num_map = dict(zip(words, range(words_size)))

print('字表大小:', words_size)
print(wav_files)
n_input = 26  # 计算梅尔倒谱系数的个数——以13或26个梅尔倒谱系数表示一段音频
n_context = 9
batch_size = 8

def next_batch(labels, wav_files, start_idx=0, batch_size=1):
    filesize = len(labels)
    end_idx = min(filesize, start_idx + batch_size)
    idx_list = range(start_idx, end_idx)
    txt_labels = [labels[i] for i in idx_list] # 当前batch取出来

    wav_files =  [wav_files[i] for i in idx_list]
    source, audio_len, target, transcriptch_len = get_audio_and_transcriptch(None, wav_files, n_input, n_context, word_num_map, txt_labels)
    start_idx += batch_size
    # 验证start_idx
    if start_idx >= filesize:
        start_idx = -1

    # 使用pad方式对其输入序列
    source, source_length = pad_sequences(source) # source是填补后的数据，source_length是原数据原本的长度

    sparse_labels = sparse_tuple_from(target)
    return start_idx, source, source_length, sparse_labels

next_idx, source, source_len, sparse_lab = next_batch(labels, wav_files, 0, batch_size)
print(len(sparse_lab)) # 3
print(np.shape(source)) #(8, 1168, 494)
t = sparse_tuple_to_texts_ch(sparse_lab, words)
print(t[0])

import tensorflow as tf

def BiRNN_model(batch_x, seq_length, n_input, n_context, n_character, keep_dropout):
    def variable_on_cpu(name, shape, initializer):
        with tf.device('/cpu:0'):
            var = tf.get_variable(name=name, shape=shape, initializer=initializer)
        return var

    batch_x_shape = tf.shape(batch_x)
    # 将输入转成时间序列优先
    batch_x = tf.transpose(batch_x, [1, 0, 2])
    batch_x = tf.reshape(batch_x, [-1, n_input+2*n_input*n_context])

    with tf.name_scope('fc1'):
        b1 = variable_on_cpu('b1', [n_hidden_1], tf.random_normal_initializer(stddev=b_stddev))
        h1 = variable_on_cpu('h1', [n_input+2*n_input*n_context, n_hidden_1], tf.random_normal_initializer(stddev=h_stddev))
        layer_1 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(batch_x, h1), b1)), relu_clip)
        layer_1 = tf.nn.dropout(layer_1, keep_dropout)

    with tf.name_scope('fc2'):
        b2 = variable_on_cpu('b2', [n_hidden_2], tf.random_normal_initializer(stddev=b_stddev))
        h2 = variable_on_cpu('h2', [n_hidden_1, n_hidden_2], tf.random_normal_initializer(stddev=h_stddev))
        layer_2 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_1, h2), b2)), relu_clip)
        layer_2 = tf.nn.dropout(layer_2, keep_dropout)

    with tf.name_scope('fc3'):
        b3 = variable_on_cpu('b3', [n_hidden_3], tf.random_normal_initializer(stddev=b_stddev))
        h3 = variable_on_cpu('h3', [n_hidden_2, n_hidden_3], tf.random_normal_initializer(stddev=h_stddev))
        layer_3 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_2, h3), b3)), relu_clip)
        layer_3 = tf.nn.dropout(layer_3, keep_dropout)

    # 双向rnn
    with tf.name_scope('lstm'):
        # 前向
        lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=True)
        lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, input_keep_prob=keep_dropout)

        # 反向
        lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=True)
        lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, input_keep_prob=keep_dropout)

        # layer_3 : [n_steps, batch_size, 2*n_cell_dim]
        layer_3 = tf.reshape(layer_3, [-1, batch_x_shape[0], n_hidden_3])

        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                                 cell_bw=lstm_bw_cell,
                                                                 inputs=layer_3,
                                                                 dtype=tf.float32,
                                                                 time_major=True,
                                                                 sequence_length=seq_length)

        outputs = tf.concat(outputs, 2)
        outputs = tf.reshape(outputs, [-1, 2*n_cell_dim]) # [2*n_cell_dim]

    with tf.name_scope('fc5'):
        b5 = variable_on_cpu('b5', [n_hidden_5], tf.random_normal_initializer(stddev=b_stddev))
        h5 = variable_on_cpu('h5', [2*n_cell_dim, n_hidden_5], tf.random_normal_initializer(stddev=h_stddev))
        layer_5 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(outputs, h5), b5)), relu_clip)
        layer_5 = tf.nn.dropout(layer_5, keep_dropout)

    with tf.name_scope('fc6'):
        # softmax分类层
        b6 = variable_on_cpu('b6', [n_character], tf.random_normal_initializer(stddev=b_stddev))
        h6 = variable_on_cpu('h6', [n_hidden_5, n_character], tf.random_normal_initializer(stddev=h_stddev))
        layer_6 = tf.add(tf.matmul(layer_5, h6), b6)

        # [n_steps, batch_size, n_character]
        layer_6 = tf.reshape(layer_6, [-1, batch_x_shape[0], n_character])

    return layer_6


# 定义占位符
input_tensor = tf.placeholder(tf.float32, [None, None, n_input+(2*n_input*n_context)], name='input')
# ctc_loss计算需要使用sparse_placeholder来生成sparsetensor
targets = tf.sparse_placeholder(tf.int32, name='targets')
seq_length = tf.placeholder(tf.int32, [None], name='seq_length')
keep_dropout = tf.placeholder(tf.float32)

# 构建网络模型
logits = BiRNN_model(input_tensor, tf.to_int64(seq_length), n_input, n_context, words_size+1, keep_dropout)

# 调用ctc_loss
avg_loss = tf.reduce_mean(ctc_ops.ctc_loss(targets, logits, seq_length))  #logits是预测结果，

# 优化器
learning_rate = 0.001
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(avg_loss)

with tf.name_scope('decode'):
    decoded, log_prob = ctc_ops.ctc_beam_search_decoder(logits, seq_length, merge_repeated=False)

with tf.name_scope('accuracy'):
    distance = tf.edit_distance(tf.cast(decoded[0], tf.int32), targets)
    # 计算label error rate（accuracy）
    ler = tf.reduce_mean(distance, name='label_error_rate')

epochs = 100
savedir = 'log/yuyinchalltest/'
saver = tf.train.Saver(max_to_keep=1)

sess = tf.Session()
# 在会话中初始化图中的变量
sess.run(tf.global_variables_initializer())

# 检查指定路径，如果有文件则载入模型并更新迭代次数
kpt = tf.train.latest_checkpoint(savedir)
print('kpt:', kpt)
startepo = 0
if kpt!=None:
    saver.restore(sess, kpt)
    ind = kpt.find('-')
    startepo = int(kpt[ind+1: ])
    print(startepo)

section = '\n{0:=^40}\n'
print(section.format('run training epoch'))

train_start = time.time()
for epoch in range(epochs):
    epoch_start = time.time()
    if epoch<startepo:
        continue
    print('epoch start:', epoch, 'total epochs=', epochs)

    n_batches_per_epoch = int(np.ceil(len(labels)/batch_size))
    print('total loop ', n_batches_per_epoch, ' in one epoch, ', batch_size, ' items in one loop.')

    train_cost = 0
    train_ler = 0

    for batch in range(n_batches_per_epoch):
        next_idx, source, source_lengths, sparse_labels = next_batch(labels, wav_files, next_idx, batch_size)
        feed = {input_tensor: source, targets: sparse_labels,
                seq_length: source_lengths, keep_dropout: keep_dropout_rate}

        # 计算avg_loss和optimizer
        batch_cost, _ = sess.run([avg_loss, optimizer], feed_dict=feed)
        train_cost += batch_cost

        # 定期评估模型
        if (batch+1)%20==0:
            print('loop: ', batch, 'Train cost: ', train_cost/(batch+1))
            feed2 = {input_tensor: source, targets: sparse_labels,
                seq_length: source_lengths, keep_dropout: 1.0}
            d, train_ler = sess.run([decoded[0], ler], feed_dict=feed2)
            dense_decoded = tf.sparse_tensor_to_dense(d, default_value=-1).eval(sess)

            dense_labels = sparse_tuple_to_texts_ch(sparse_labels, words)

            counter = 0
            print('Label err rate: ', train_ler)
            for orig, decoded_arr in zip(dense_labels, dense_decoded):
                decoded_str = ndarray_to_text_ch(decoded_arr, words)
                print('File ', counter)
                print('Orignal: ', orig)
                print('Decoded: ', decoded_str)
                counter += 1
                break

    epoch_duration = time.time() - epoch_start
    log = 'Epoch {}/{}, train_cost: {:.3f}, train_ler: {:.3f}, time: {:.2f} sec'
    print(log.format(epoch, epochs,train_cost, train_ler, epoch_duration))
    saver.save(sess, savedir+'yuyinch.cpkt', global_step=epoch)
train_duration = time.time() - train_start
print('Training complete, total duration: {:.2f} min'.format(train_duration/60))
sess.close()

