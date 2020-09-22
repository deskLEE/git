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

variable_names = [v.name for v in tf.trainable_variables()]
print(variable_names)


# tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
# print(tensor_name_list)
#for tensor_name in tensor_name_list:
#    print(tensor_name,'\n')
#graph = tf.get_default_graph()
#for op in graph.get_operations():
#    print(op.name)

