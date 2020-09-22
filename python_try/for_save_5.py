import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

v1 = tf.get_variable(
    name = 'v1',
    shape = [1],
    initializer = tf.constant_initializer(1.0)
)
v2 = tf.get_variable(
    name = 'v2',
    shape = [1],
    initializer = tf.constant_initializer(0.0)
)

result = v1 + v2
init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    saver.save(sess, './model/model.ckpt')
# ------- 模型的训练 --------

graph = tf.get_default_graph()
for op in graph.get_operations():
    print(op.name)