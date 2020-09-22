# 使用print_tensors_in_checkpoint_file打印ckpt里的内容
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
file_name = '.\model\model.ckpt'

print_tensors_in_checkpoint_file(file_name,  # ckpt文件名字
                                 None,# ,  # 如果为None,则默认为ckpt里的所有变量
                                 True,  # bool 是否打印所有的tensor，这里打印出的是tensor的值，一般不推荐这里设置为False
                                 True)  # bool 是否打印所有的tensor的name
# 上面的打印ckpt的内部使用的是pywrap_tensorflow.NewCheckpointReader所以，掌握NewCheckpointReader才是王道