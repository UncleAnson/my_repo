#! -*- coding: utf-8 -*-
from tqdm import tqdm
import os, json
import pandas as pd

from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam

min_count = 32
maxlen = 400
batch_size = 64
epochs = 100
char_size = 128

data = pd.read_csv('./news.csv')
if os.path.exists('./seq2seq_config.json'):
    print('读取配置字文件')
    chars, id2char, char2id = json.load(open('./seq2seq_config.json'))
    id2char = {int(i): j for i, j in id2char.items()}
else:
    chars = {}
    for t, c in tqdm(data[['title', 'content']].values): # 按字不按词
        t, c = t.strip(), c.strip()
        for w in t:
            chars[w] = chars.get(w,0) + 1
        for w in c:
            chars[w] = chars.get(w,0) + 1
    chars = {i: j for i, j in chars.items() if j >= min_count}  # 按频次过滤
    # 0: mask  1: unk  2: start  3: end
    id2char = {i+4:j for i,j in enumerate(chars)}
    char2id = {j:i for i,j in id2char.items()}
    json.dump([chars,id2char,char2id], open('seq2seq_config.json', 'w'))


def str2id(s, start_end=False):
    # 文字转整数id
    if start_end: # 补上<start>和<end>标记
        ids = [char2id.get(c, 1) for c in s[: maxlen-2]] # 允许的最大长度减二
        ids = [2] + ids + [3]
    else: # 普通转化
        ids = [char2id.get(c, 1) for c in s[: maxlen]]
    return ids


def id2str(ids):
    # ids列表转句子，id转文字，找不到的用空字符代替
    return ''.join([id2char.get(i, '') for i in ids])


def padding(x):
    # 右补0，padding至batch内的最大长度，x是数据集
    ml = max([len(i) for i in x]) # maxlen
    return [i + [0] * (ml-len(i)) for i in x]


def data_generator():
    # 数据生成器
    X,Y = [],[]
    while True:
        for t, c in data[['title', 'content']].values:
            X.append(str2id(c))
            Y.append(str2id(t, start_end=True))
            if len(X) == batch_size:
                X = np.array(padding(X))
                Y = np.array(padding(Y))
                yield [X, Y], None # None?
                X,Y = [],[]


# 搭建seq2seq模型, x和y都是一维数组
x_in = Input(shape=(None,))
y_in = Input(shape=(None,))
x = x_in
y = y_in
# K相当于tf. 或T.
# 't2' is a tensor of shape [2, 3, 5]
# shape(expand_dims(t2, 0)) ==> [1, 2, 3, 5]
# greater(a,b) a>b->True a<b->False
# Lambda 对x执行匿名函数，转成0，1（大于0则是1，0是补的）
x_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x)
y_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(y)


def to_one_hot(x): # 输出一个词表大小的向量，来标记该词是否在文章出现过
    x, x_mask = x
    x = K.cast(x, 'int32')
    x = K.one_hot(x, len(chars)+4)   # +4，因为0-3是代表四个特殊标记 https://blog.csdn.net/nini_coded/article/details/79250600
    x = K.sum(x_mask * x, 1, keepdims=True) # 独热双保险
    x = K.cast(K.greater(x, 0.5), 'float32') # 超过0.5则是1，低于0.5则是0
    return x


class ScaleShift(Layer): # 继承layer类
    """缩放平移变换层（Scale and shift）
    """
    def __init__(self, **kwargs):
        super(ScaleShift, self).__init__(**kwargs)

    def build(self, input_shape):
        kernel_shape = (1,)*(len(input_shape)-1) + (input_shape[-1],)
        self.log_scale = self.add_weight(name='log_scale',
                                         shape=kernel_shape,
                                         initializer='zeros')
        self.shift = self.add_weight(name='shift',
                                     shape=kernel_shape,
                                     initializer='zeros')

    def call(self, inputs):
        # 对当前层的输入进行操作，通过求对数进行缩放，通过加减进行平移
        # exp 逐元素求自然指数
        x_outs = K.exp(self.log_scale) * inputs + self.shift
        return x_outs


x_one_hot = Lambda(to_one_hot)([x, x_mask])
x_prior = ScaleShift()(x_one_hot) # 学习输出的先验分布（标题的字词很可能在文章出现过）

embedding = Embedding(len(chars)+4, char_size)
# x，y共用一个中文嵌入层
x = embedding(x)
y = embedding(y)

# encoder，双层双向LSTM
# 非对齐时序标注任务（不是简单的分类任务），如果是在gpu环境下训练可以使用CuDNNLSTM

x = Bidirectional(LSTM(int(char_size/2), return_sequences=True))(x)
x = Bidirectional(LSTM(int(char_size/2), return_sequences=True))(x)

# decoder，双层单向LSTM，如果是在gpu环境下训练可以使用CuDNNLSTM
y = LSTM(char_size, return_sequences=True)(y)
y = LSTM(char_size, return_sequences=True)(y)

# =======================================================
class Interact(Layer):
    """交互层，负责融合encoder和decoder的信息
    """
    def __init__(self, **kwargs):
        super(Interact, self).__init__(**kwargs)

    def build(self, input_shape):
        in_dim = input_shape[0][-1]
        out_dim = input_shape[1][-1]
        self.kernel = self.add_weight(name='kernel',
                                      shape=(in_dim, out_dim),
                                      initializer='glorot_normal')

    def call(self, inputs):
        q, v, v_mask = inputs
        k = v
        mv = K.max(v - (1. - v_mask) * 1e10, axis=1, keepdims=True) # maxpooling1d
        mv = mv + K.zeros_like(q[:,:,:1]) # 将mv重复至“q的timesteps”份
        # 下面几步只是实现了一个乘性attention
        qw = K.dot(q, self.kernel)
        a = K.batch_dot(qw, k, [2, 2]) / 10.
        a -= (1. - K.permute_dimensions(v_mask, [0, 2, 1])) * 1e10
        a = K.softmax(a)
        o = K.batch_dot(a, v, [2, 1])
        # 将各步结果拼接
        return K.concatenate([o, q, mv], 2)

    def compute_output_shape(self, input_shape):
        return (None, input_shape[0][1],
                input_shape[0][2]+input_shape[1][2]*2)


xy = Interact()([y, x, x_mask])
xy = Dense(512, activation='relu')(xy)
xy = Dense(len(chars)+4)(xy)
xy = Lambda(lambda x: (x[0]+x[1])/2)([xy, x_prior]) # 与先验结果平均
xy = Activation('softmax')(xy)

# 交叉熵作为loss，但mask掉padding部分
cross_entropy = K.sparse_categorical_crossentropy(y_in[:, 1:], xy[:, :-1])
loss = K.sum(cross_entropy * y_mask[:, 1:, 0]) / K.sum(y_mask[:, 1:, 0])

model = Model([x_in, y_in], xy)
model.add_loss(loss)
model.compile(optimizer=Adam(1e-3))


def gen_title(s, topk=3):
    """beam search解码
    每次只保留topk个最优候选结果；如果topk=1，那么就是贪心搜索
    """
    xid = np.array([str2id(s)] * topk) # 输入转id
    yid = np.array([[2]] * topk) # 解码均以<start>开通，这里<start>的id为2
    scores = [0] * topk # 候选答案分数
    for i in range(50): # 强制要求标题不超过50字
        proba = model.predict([xid, yid])[:, i, 3:] # 直接忽略<padding>、<unk>、<start>
        log_proba = np.log(proba + 1e-6) # 取对数，方便计算
        arg_topk = log_proba.argsort(axis=1)[:,-topk:] # 每一项选出topk
        _yid = [] # 暂存的候选目标序列
        _scores = [] # 暂存的候选目标序列得分
        if i == 0:
            for j in range(topk):
                _yid.append(list(yid[j]) + [arg_topk[0][j]+3])
                _scores.append(scores[j] + log_proba[0][arg_topk[0][j]])
        else:
            for j in range(len(xid)):
                for k in range(topk): # 遍历topk*topk的组合
                    _yid.append(list(yid[j]) + [arg_topk[j][k]+3])
                    _scores.append(scores[j] + log_proba[j][arg_topk[j][k]])
            _arg_topk = np.argsort(_scores)[-topk:] # 从中选出新的topk
            _yid = [_yid[k] for k in _arg_topk]
            _scores = [_scores[k] for k in _arg_topk]
        yid = []
        scores = []
        for k in range(len(xid)):
            if _yid[k][-1] == 3: # 找到<end>就返回
                return id2str(_yid[k])
            else:
                yid.append(_yid[k])
                scores.append(_scores[k])
        yid = np.array(yid)
    # 如果50字都找不到<end>，直接返回
    return id2str(yid[np.argmax(scores)])


s1 = u'夏天来临，皮肤在强烈紫外线的照射下，晒伤不可避免，因此，晒后及时修复显得尤为重要，否则可能会造成长期伤害。专家表示，选择晒后护肤品要慎重，芦荟凝胶是最安全，有效的一种选择，晒伤严重者，还请及时就医。'
s2 = u'8月28日，网络爆料称，华住集团旗下连锁酒店用户数据疑似发生泄露。从卖家发布的内容看，数据包含华住旗下汉庭、禧玥、桔子、宜必思等10余个品牌酒店的住客信息。泄露的信息包括华住官网注册资料、酒店入住登记的身份信息及酒店开房记录，住客姓名、手机号、邮箱、身份证号、登录账号密码等。卖家对这个约5亿条数据打包出售。第三方安全平台威胁猎人对信息出售者提供的三万条数据进行验证，认为数据真实性非常高。当天下午，华住集团发声明称，已在内部迅速开展核查，并第一时间报警。当晚，上海警方消息称，接到华住集团报案，警方已经介入调查。'

class Evaluate(Callback):
    def __init__(self):
        self.lowest = 1e10
    def on_epoch_end(self, epoch, logs=None):
        # 训练过程中观察一两个例子，显示标题质量提高的过程
        print(gen_title(s1))
        print(gen_title(s2))
        # 保存最优结果
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save_weights('./best_model.weights')

evaluator = Evaluate()
model.fit_generator(data_generator(),
                    steps_per_epoch=1000,
                    epochs=epochs,
                    callbacks=[evaluator])