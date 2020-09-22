1. tensorboard_test 中介绍了如何打开tensorboard的打开方式
以及tensorboard中显示内容的展示方法

2. tensorboard_test_2是更加详细的关于tensorboard的具体方法展示

3. for_save中展示了将模型保存为.ckpt格式的方法

4. for_save_2中展示了利用restore将模型加载出来得到结果的方法

5. for_save_3中展示了在保存模型代码中输出代码中的张量名的代码

variable_names = [ v.name for v in tf.trainable_variables()]
print(variable_names)
----------------------------------------------------
结果为：
['v1:0', 'v2:0']

6. for_save_4中展示了在保存模型代码中输出网络中所有节点的代码

tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
---------------------------
结果为：
v1/Initializer/Const 

v1 

v1/Assign 

v1/read 

7. for_save_5中的作用和6.中类似：
graph = tf.get_default_graph()
for op in graph.get_operations():
    print(op.name)
---------------------------
结果为：
v1/Initializer/Const
v1
v1/Assign
v1/read
v2/Initi

8. for_save_6中给出了通过直接变量提取的方式来进行结果恢复
saver = tf.train.import_meta_graph(
    './model/model.ckpt.meta'
)
    print (sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))

9. see_key_test_1和see_key_test_2给出了通过保存的模型来观察所有的张量的方法