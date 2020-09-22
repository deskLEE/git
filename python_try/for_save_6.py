import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# 从model.ckpt.meta中直接加载已经持久化的图
saver = tf.train.import_meta_graph(
    './model/model.ckpt.meta'
)
with tf.Session() as sess:
#    saver.restore(sess, './model/model.ckpt')
    saver.restore(sess, "./model/model.ckpt")
    print (sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))