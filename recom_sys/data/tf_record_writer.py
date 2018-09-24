import numpy as np
import tensorflow as tf
import sys
from preprocess_data import _get_dataset

def _add_to_tfrecord(data_sample, tfrecord_writer):
    data_sample=list(data_sample.astype(dtype=np.float32))
    example = tf.train.Example(features=tf.train.Features(feature={'movie_ratings': float_feature(data_sample)}))                                          
    tfrecord_writer.write(example.SerializeToString())
    
# 返回tfrecord文件名
def _get_output_filename(output_dir, idx, name):
    return '%s/%s_%03d.tfrecord' % (output_dir, name, idx)

def float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def main():
    SAMPLES_PER_FILES = 100 # 限制每个tfrecord文件100条数据
    training_set, test_set = _get_dataset(sys.argv[1])

    for data_set, name, dir_ in zip([training_set, test_set], ['train', 'test'], [sys.argv[2], sys.argv[3]]):
        # [(training_set, 'train', sys.argv[2]),(test_set, 'test', sys.argv[3])]
        num_samples=len(data_set)
        i, fidx = 0, 1
        while i < num_samples:
            tf_filename = _get_output_filename(dir_, fidx, name)
            with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
                j = 0
                while i < num_samples and j < SAMPLES_PER_FILES:
                    sys.stdout.write('\r>> Converting sample %d/%d' % (i+1, num_samples))
                    sys.stdout.flush()

                    sample = data_set[i]
                    _add_to_tfrecord(sample, tfrecord_writer)
                    i += 1
                    j += 1
                fidx += 1
        print()
    print('Finished converting the dataset!')
    
if __name__ == "__main__":
    main()
            
    







