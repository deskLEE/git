from for_save import Network  # 引入训练网络的类
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pylab as pl

sess = tf.Session()     # 建立sess
saver = tf.train.Saver().restore(sess, "./tmp/model.ckpt")
# 利用函数restore将ckpt模型存储下来
net = Network()     # net是网络的一个实例化

x_data = np.linspace(-1,1,100).reshape(1,100)
pre = sess.run(net.output,feed_dict={net.x:x_data})  # 直接将x_data输入进去
# 最终得到pre的运行结果

#### 这里切记一点，在恢复模型前，一定要自己定义一下你要恢复的变量

pl.plot(x_data.T,pre.T)
pl.grid()
pl.show()