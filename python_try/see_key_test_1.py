import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import os
from tensorflow.python import pywrap_tensorflow

checkpoint_path = os.path.join('.\model\model.ckpt')
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key)
    print(reader.get_tensor(key))
    # print(reader.get_tensor(key)) #相应的值
# 将上述图存成pb文件，这个文件包含了模型的图结构也包含了模型的参数值，便于部署和inference