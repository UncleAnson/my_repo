import tensorflow as tf
import tensorboard
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('mnist_data/', one_hot=True)

print(mnist.train.images)
print(mnist.train.images.shape) # 50000条

import pylab
# im = mnist.train.images[1]
# im = im.reshape(-1, 28)#*255 # 28*28
# pylab.imshow(im, cmap='binary')
# pylab.show()

print(mnist.test.images.shape) # 10000
print(mnist.validation.images.shape) # 5000

# 查看标签
print(mnist.train.labels.shape)
print(mnist.train.labels)

tf.reset_default_graph()
# 构建模型
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.random_normal([784, 10]))
# b = tf.Variable(tf.random_normal([10]))
b = tf.Variable(tf.zeros([10]))
s = tf.add(tf.matmul(x, W), b)
pred = tf.nn.softmax(s)

# 定义损失函数&优化器
# cost = -tf.reduce_sum(y*tf.log(pred))
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
# cost = tf.reduce_sum(tf.square(y-pred)) # deal
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

# 设定基本训练参数
train_epoch = 2
batch_size = 100#64
display_step = 1
print(mnist.train.num_examples)
print(mnist.train.images.shape[0])

mnist_data = mnist.train.images*1
print(mnist_data[0])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('-------------', sess.run(W))
    for epoch in range(train_epoch):
        avg_cost = 0.
        total_batch = int(mnist.train.images.shape[0] / batch_size)

        for i in range(total_batch):
            # batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            x_data = mnist_data[i*batch_size: (i+1)*batch_size]
            # x_data = mnist.train.images[i*batch_size: (i+1)*batch_size]
            y_data = mnist.train.labels[i*batch_size: (i+1)*batch_size]

            # _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})
            # c = sess.run([optimizer], feed_dict={x: x_data, y: y_data})
            _, c, p, argmax = sess.run([optimizer, cost, pred, tf.argmax(pred, 1)], feed_dict={x: x_data, y: y_data})
            # print('\n==========p========\n', p, '\n==========s========\n', s_)
            # print('\n==========w========\n', w)
            # print('\n==========p========\n', p, argmax)

        #     avg_cost += c
        #     # print(c)
        # avg_cost = avg_cost/total_batch
            avg_cost += c/total_batch

        if (epoch+1)%display_step == 0:
            print('Epoch:', '%04d'%(epoch+1), 'cost=', '{:.9f}'.format(avg_cost))
    print('Finished!')
    #
    # # 测试
    # correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    # # pred:
    # # [[1.36713106e-02 8.70788377e-03 4.97930259e-01 2.40342855e-01
    # #  6.32920649e-09 1.06183987e-03 9.43126466e-09 9.76515934e-02
    # #  1.81352871e-12 1.40634254e-01]]
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # print('Accuracy:', sess.run(accuracy, {x:mnist.test.images, y:mnist.test.labels}))

    # 保存模型
    saver = tf.train.Saver()
    path = 'model/model.ckpt' # 存储路径/文件名称
    saver.save(sess, path)
    print('Model saved in {}'.format(path))


with tf.Session() as sess:
    saver = tf.train.Saver()
    path = 'model/model.ckpt'
    saver.restore(sess, path)

    # 测试
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # pred:
    # [[1.36713106e-02 8.70788377e-03 4.97930259e-01 2.40342855e-01
    #  6.32920649e-09 1.06183987e-03 9.43126466e-09 9.76515934e-02
    #  1.81352871e-12 1.40634254e-01]]
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy:', sess.run(accuracy, {x: mnist.test.images, y: mnist.test.labels}))

    batch_x, batch_y = mnist.validation.next_batch(2)
    p = sess.run(pred, feed_dict={x: batch_x})

    x0 = batch_x[0]
    x0 = x0.reshape(-1, 28)

    x1 = batch_x[1]
    x1 = x1.reshape(-1, 28)

    print(batch_y)
    print(p)
    pylab.imshow(x0, cmap='binary')
    pylab.show()

    pylab.imshow(x1, cmap='binary')
    pylab.show()