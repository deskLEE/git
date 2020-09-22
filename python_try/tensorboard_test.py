#coding=utf-8

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# # 不加这几句，则CONV 报错
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

input_img   = tf.placeholder(dtype=tf.float32)
input_label = tf.placeholder(dtype=tf.float32)

param_kernel1   = tf.get_variable(name='param_kernel1',shape=[3,3,1,8])
param_bias1     = tf.get_variable(name='param_bias1',shape=[8])
param_kernel2   = tf.get_variable(name='param_kernel2',shape=[3,3,8,8])
param_bias2     = tf.get_variable(name='param_bias2',shape=[8])
param_kernel3   = tf.get_variable(name='param_kernel3',shape=[3,3,8,1])

output1 = tf.nn.conv2d(input=input_img,filter=param_kernel1,strides=[1,1,1,1],padding='SAME')
output1_bias    = tf.add(output1,param_bias1)
output2 = tf.nn.conv2d(input=output1_bias,filter=param_kernel2,strides=[1,1,1,1],padding='SAME')
output2_bias    = tf.add(output2,param_bias2)
output_end_tmp  = tf.nn.conv2d(input=output2_bias,filter=param_kernel3,strides=[1,1,1,1],padding='SAME')
output_end  = tf.squeeze(output_end_tmp)

loss=tf.reduce_mean(tf.square(input_label-output_end))
train_step  = tf.train.AdamOptimizer(0.001).minimize(loss)

config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True

sess=tf.Session(config=config)
sess.run(tf.global_variables_initializer())

tf.summary.scalar('Loss',loss)   # 捕捉loss的变化   最终会在scalar中显示出这条线
tf.summary.image('output2',tf.transpose(output2,perm=[3,1,2,0]),max_outputs=8)  # 具体请查看tf.summary.image()的api介绍
# tf.summary.image的含义
merged_summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('./Log',sess.graph)
# 图片的保存
for i in range(0,100):
    img=np.random.random((1,32,32,1))     # 随机生成32x32x1的图片一张
    label=np.random.random((32,32))       # label 是随机的32x32的矩阵
    [a,theloss]=sess.run([train_step,loss],feed_dict={input_img:img,input_label:label})
    print(theloss) # 输出theloss的结果
    summary=sess.run(merged_summary_op,feed_dict={input_img:img,input_label:label})
    summary_writer.add_summary(summary,i) # 表示以 i 的次数保存

variable_names = [v.name for v in tf.trainable_variables()]
print(variable_names)

#运行本代码后，按照tensorboard的使用步骤查看即可

# 分析：
# 1、tf.summary.scalar(name,tensor,collections=None,family=None)
# name表示生成节点的名字，也会作为TensorBoard中的系列的名字
# tensor:包含一个值的实数Tensor
# 一般在画loss,accuary时会用到这个函数。
# -------------------------------------------------------------
# 2、输出带图像的probuf，汇总数据的图像的的形式如下： ' tag /image/0', ' tag /image/1'...，如：input/image/0等。
#
# 格式：tf.summary.image(name, tensor, max_images=3, collections=None, name=None)
# name 节点名字
# tensor 格式为[batch_size, height, width, channels]
# channels=1 为灰度图像 channels=3为RGB图像 channels=4为RGBA图像，包含透明度信息
# max_outputs表示生成图像的最大批处理元素数
# -------------------------------------------------------------
# 3、tf.transpose(
#     a,
#     perm=None,
#     name='transpose',
#     conjugate=False
# )
#
# a:表示的是需要变换的张量
# perm:a的新的维度序列
# -------------------------------------------------------------
# 4、tf.summary.merge_all() 用于对图片进行自动管理
# -------------------------------------------------------------
# 5、tf.summary.FileWriter(dir,sess.graph)
# dir表示地址 用于定义一个写入图像的目标文件
# -------------------------------------------------------------
# 6、显示tensorboard的命令
# tensorboard --logdir ./Log --host=127.0.0.1
# -------------------------------------------------------------
# 7、输出所有的变量名
# variable_names = [v.name for v in tf.trainable_variables()]
# print(variable_names)
# -------------------------------------------------------------
# 8、tensor的数据格式：
# Tensor("add:0", shape=(2,), dtype=float32)
# 保留的三个属性：名字name/维度shape/类型type
# 张量的第一个属性名字不仅是张量的唯一标识符，同样也给出了这个张量是如何计算出来的。
# 张量的命名形式“node：src_output”，
# node：是节点名称，
# src_output:表示当前张量来自节点的第几个输出。
# 9、关于tensor的命名规则：
# 总结：<张量> tensor就是指输出的数据，名字就是其对应节点名字加：0。
# 节点（op）包括两种属性：1.name(这个那么可以自己指定，不指定话就被默认把op换成小写的字母作为name). 2.OP(有时也叫type,可以理解为tensorflow中的基本类型，常见的有Add,Mul。。。。。)
# -------------------------------------------------------------
