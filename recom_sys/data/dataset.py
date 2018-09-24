import tensorflow as tf
import os
'''
构建从tfrecord文件中读取数据的函数
'''
def _get_training_data(FLAGS):
    # 获取训练用的tfrecord文件的绝对地址
    filenames = [FLAGS.tf_records_train_path+f for f in os.listdir(FLAGS.tf_records_train_path)]
    # 训练
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse)
    dataset = dataset.shuffle(buffer_size=500) # 以500条数据为最小单位进行打乱
    dataset = dataset.repeat() # 默认无限重复/循环
    dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True) # n%batch_size 舍去
    dataset = dataset.prefetch(buffer_size=1) # 返回一个可迭代对象
    # 推理
    dataset2 = tf.data.TFRecordDataset(filenames)
    dataset2 = dataset2.map(parse)
    dataset2 = dataset2.shuffle(buffer_size=1)
    dataset2 = dataset2.repeat()
    dataset2 = dataset2.batch(1)
    dataset2 = dataset2.prefetch(buffer_size=1)
    
    return dataset, dataset2

def _get_test_data(FLAGS):
    filenames = [FLAGS.tf_records_test_path+f for f in os.listdir(FLAGS.tf_records_test_path)]
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse)
    dataset = dataset.shuffle(buffer_size=1)
    dataset = dataset.repeat()
    dataset = dataset.batch(1)
    dataset = dataset.prefetch(buffer_size=1)

    return dataset

def parse(serialized):
    # tfrecord文件内数据的解析器
    features = {'movie_ratings':tf.FixedLenFeature([3952], tf.float32)}  # 每个用户有3952个特征（电影）
    parsed_example = tf.parse_single_example(serialized, features=features)
    movie_ratings = tf.cast(parsed_example['movie_ratings'], tf.float32) # 获得 用户-电影 矩阵
     
    return movie_ratings