import os
import numpy as np

def get_wavs_labels(wav_path, label_file):
    # 获取训练用的wav文件路径列表
    wav_files=[]
    for dirpath, dirnames, filenames in os.walk(wav_path):
        for filename in filenames:
            if filename.endswith('.wav') or filename.endswith('.WAV'):
                filename_path = os.sep.join([dirpath, filename])
                if os.stat(filename_path).st_size<240000:
                    continue
                wav_files.append(filename_path)
    # 构造label字典
    labels_dict = {}
    with open(label_file, 'rb') as f:
        for label in f:
            label = label.strip(b'\n')
            label_id = label.split(b' ', 1)[0]
            label_text = label.split(b' ', 1)[1]
            labels_dict[label_id.decode('ascii')] = label_text.decode('utf-8')

    # wav_id关联了wav文件和label
    labels = []
    new_wav_files = []
    for wav_file in wav_files:
        wav_id = os.path.basename(wav_file).split('.')[0]

        if wav_id in labels_dict:
            labels.append(labels_dict[wav_id])
            new_wav_files.append(wav_file)

    return new_wav_files, labels

def get_wavs_labels_v2(data_path):
    # 获取训练用的wav文件路径列表
    wav_files=[]
    labels_dict = {}
    for filename in os.listdir(data_path)[:30]:
        if filename.endswith('.wav'):
            filename_path = os.sep.join([data_path, filename])
            if os.stat(filename_path).st_size<240000:
                continue
            wav_files.append(filename_path)
        # 构造label字典
        elif filename.endswith('.trn'):
            filename_path = os.sep.join([data_path, filename])
            label_id = filename.split('.')[0]
            sub_path = open(filename_path).read().replace('/', '\\').strip()
            label_text = open(os.sep.join([data_path, sub_path]), encoding='utf-8').read().split('\n')[0]
            labels_dict[label_id] = label_text

    # wav_id关联了wav文件和label
    labels = []
    new_wav_files = []
    for wav_file in wav_files:
        wav_id = os.path.basename(wav_file).split('.')[0]

        if wav_id in labels_dict:
            labels.append(labels_dict[wav_id])
            new_wav_files.append(wav_file)

    return new_wav_files, labels


import numpy as nps
from python_speech_features import mfcc
import scipy.io.wavfile as wav

def get_audio_and_transcriptch(txt_files, wav_files, n_input, n_context, word_num_map, txt_labels=None):
    # n_input 计算梅尔倒谱系数的个数
    # n_context 对于每个时间点，要包含上下文样本的个数
    audio = []
    audio_len = []
    transcript = []
    transcript_len = []
    if txt_files!=None:
        txt_labels = txt_files

    for txt_obj, wav_file in zip(txt_labels, wav_files):
        audio_data = audiofile_to_input_vector(wav_file, n_input, n_context)
        audio_data = audio_data.astype('float32')

        audio.append(audio_data)
        audio_len.append(np.int32(len(audio_data)))  # 为什么要加np.int32

        # 载入音频对应的文本
        target = []
        if txt_files!=None: # txt_obj是文件
            target = get_ch_label_v(txt_obj, word_num_map)
        else: # txt_obj是labels
            target = get_ch_label_v(None, word_num_map, txt_obj)

        transcript.append(target)
        transcript_len.append(len(target))

    audio = np.asarray(audio)
    audio_len = np.asarray(audio_len)
    transcript = np.asarray(transcript)
    transcript_len = np.asarray(transcript_len)
    return audio, audio_len, transcript, transcript_len

def audiofile_to_input_vector(audio_filename, numcep, numcontext):
    # 加载wav文件
    fs, audio = wav.read(audio_filename)
    # 获得mfcc coefficients
    orig_inputs = mfcc(audio, samplerate=fs, numcep=numcep)
    orig_inputs = orig_inputs[::2]  # (139, 26 )

    train_inputs = np.array([], np.float32)
    train_inputs.resize((orig_inputs.shape[0], numcep+2*numcep*numcontext))

    empty_mfcc = np.array([])
    empty_mfcc.resize((numcep))
    # 准备输入数据： 当前样本前九个，当前样本，当前样本后九个序列样本
    time_slices = range(train_inputs.shape[0]) # 139个切片
    context_past_min = time_slices[0] + numcontext
    context_future_max = time_slices[-1] - numcontext

    for time_slice in time_slices:
        # 前9个补0，mfcc features
        need_empty_past = max(0, (context_past_min - time_slice))
        empty_source_past = list(empty_mfcc for empty_slots in range(need_empty_past))
        data_source_past = orig_inputs[max(0, time_slice - numcontext): time_slice]
        assert len(empty_source_past)+ len(data_source_past)== numcontext

        # 后9个补0，mfcc features
        need_empty_future = max(0, time_slice - context_future_max)
        empty_source_future = list(empty_mfcc for empty_slots in range(need_empty_future))
        data_source_future = orig_inputs[time_slice+1: time_slice+numcontext+1]
        assert len(empty_source_future) + len(data_source_future) == numcontext

        if need_empty_past:
            past = np.concatenate((empty_source_past, data_source_past))
        else:
            past = data_source_past

        if need_empty_future:
            future = np.concatenate((data_source_future, empty_source_future))
        else:
            future = data_source_future

        past = np.reshape(past, numcontext*numcep)
        now = orig_inputs[time_slice]
        future = np.reshape(future, numcontext*numcep)

        train_inputs[time_slice] = np.concatenate((past, now, future))
        assert len(train_inputs[time_slice]) == numcep+2*numcep*numcontext
    train_inputs = (train_inputs - np.mean(train_inputs))/np.std(train_inputs)
    return train_inputs


def pad_sequences(sequences, maxlen=None, dtype=np.float32, padding='post', truncating='post', value=0.):
    lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # 从第一个非空的序列中得到样本形状
    sample_shape = tuple()
    for s in sequences:
        if len(s)>0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape)*value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s)==0:
            continue
        if truncating =='pre':
            trunc = s[-maxlen:]
        elif truncating=='post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood'%truncating)

        # 检查trunc
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:]!=sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s'%(trunc.shape[1:], idx, sample_shape))

        if padding=='post':
            x[idx, :len(trunc)] = trunc
        elif padding=='pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood'%padding)
    return x, lengths

# 文本样本转向量
def get_ch_label_v(txt_file, word_num_map, txt_label=None):
    words_size = len(word_num_map)
    to_num = lambda word: word_num_map.get(word, words_size)

    if txt_file!=None:
        txt_label = get_ch_label(txt_file)

    labels_vector = list(map(to_num, txt_label))
    return labels_vector

def get_ch_label(txt_file):
    labels = ""
    with open(txt_file, 'rb') as f:
        for label in f:
            labels = labels + label.decode('gb2312')
    return labels

# 密集矩阵转成稀疏矩阵
def sparse_tuple_from(sequences, dtype=np.int32):
    indices = []
    values = []
    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), indices.max(0)[1]+1], dtype=np.int64)

    return indices, values, shape

# 将字向量转换成文字
SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1

def sparse_tuple_to_texts_ch(tuple, words):
    # 将稀疏矩阵的字向量转成文字
    indices = tuple[0]
    values = tuple[1]
    results = ['']*tuple[2][0]
    for i in range(len(indices)):
        index = indices[i][0] # 第i条数据
        c = values[i]
#
        c = ' ' if c == SPACE_INDEX else words[c]
        results[index] = results[index] + c

    return results

def ndarray_to_text_ch(values, words):
    # 将密集矩阵的字向量转成文字
    results = ''
    for i in range(len(values)):
        results += words[values[i]]
    return results.replace('`', ' ')



