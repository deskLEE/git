import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pylab as pl


class Network():
    x = tf.placeholder(shape=[1, None], dtype=tf.float32)
    y = tf.placeholder(shape=[1, None], dtype=tf.float32)

    inputW = tf.Variable(tf.random_normal([10, 1]))
    inputB = tf.Variable(tf.random_normal([10, 1]))

    hideW = tf.Variable(tf.random_normal([1, 10]))
    hideB = tf.Variable(tf.random_normal([1, 1]))

    h1 = tf.nn.sigmoid(tf.add(tf.matmul(inputW, x), inputB))
    output = tf.add(tf.matmul(hideW, h1), hideB)

    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - output)))

    opt = tf.train.AdamOptimizer(1)

    train_step = opt.minimize(loss)


if __name__ == '__main__':

    x_data = np.linspace(-1, 1, 100).reshape(1, 100)
    noise = np.random.normal(0, 0.05, x_data.shape)
    y_data = x_data ** 3 + 1 + noise

    net = Network()
    sess = tf.Session()
    init = tf.global_variables_initializer()

    sess.run(init)
    saver = tf.train.Saver()  # 保存模型的saver函数

    fig = pl.figure(1)
    pl.ion()

    train_step = 200
    for step in range(train_step):
        print('第', step + 1, '次训练')
        sess.run(net.train_step, feed_dict={net.x: x_data, net.y: y_data})
        pre = sess.run(net.output, feed_dict={net.x: x_data})

        pl.clf()
        pl.scatter(x_data, y_data)
        pl.plot(x_data.T, pre.T, 'r')  # 显示的红线是预测值 显示的点图是实际值
        pl.pause(0.01)

    saver.save(sess, "./tmp/model.ckpt")    # 利用saver.save 函数保存model的ckpt文件