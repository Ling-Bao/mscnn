# -*- coding:utf-8 -*-
"""
@Function: Structure of MSCNN crowd counting
@Source: Multi-scale Convolution Neural Networks for Crowd Counting
         https://arxiv.org/abs/1702.02359
@Data set: https://pan.baidu.com/s/12EqB1XDyFBB0kyinMA7Pqw 密码: sags  --> Have some problems

@Author: Ling Bao
@Code verification: Ling Bao
@说明：
    学习率：1e-4
    平均loss : 14.

@Data: Sep. 11, 2017
@Version: 0.1
"""

# 系统模块
import re

# 机器学习库
import tensorflow as tf

# 项目模块
import mscnn_train

# 模型参数设置
MP_NAME = 'mp'
train_log = 'train_log'
model = 'model'
output = 'output'
data_train_gt = 'Data_original/Data_gt/train_gt/'
data_train_im = 'Data_original/Data_im/train_im/'
data_train_index = 'Data_original/dir_name.txt'

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 1, """批次处理图片数目""")
tf.app.flags.DEFINE_string('train_log', train_log, """训练日志""")
tf.app.flags.DEFINE_string('model_dir', model, """模型保存""")
tf.app.flags.DEFINE_string('output_dir', output, """输出中间结果""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """是否记录设备布局""")
tf.app.flags.DEFINE_string('data_train_gt', data_train_gt, """训练集标签""")
tf.app.flags.DEFINE_string('data_train_im', data_train_im, """训练集图片""")
tf.app.flags.DEFINE_string('data_train_index', data_train_index, """训练集图片""")


def _activation_summary(x):
    """
    概要汇总函数
    :param x: 待保存变量
    :return: None
    """
    tensor_name = re.sub('%s_[0-9]*/' % MP_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
    """
    创建变量
    :param name: name_scope
    :param shape: tensor维度
    :param initializer: 初始化值
    :return: tensor变量
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)

    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """
    创建有权重衰减项的变量
    :param name: name_scope
    :param shape: tensor维度
    :param stddev: 用于初始化的标准差
    :param wd: 权重
    :return: tensor变量
    """
    # wd 为衰减因子,若为None则无衰减项
    var = _variable_on_cpu(name, shape, tf.random_normal_initializer(stddev=stddev))
    if wd:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)

    return var


class BatchNorm(object):
    """
    BN操作类
    """
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        """
        初始化函数
        :param epsilon: 精度
        :param momentum: 动量因子
        :param name: name_scope
        """
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x):
        """
        BN算子
        :param x: 输入变量
        :return:
        """
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None,
                                            epsilon=self.epsilon, scale=True, scope=self.name)


def multi_scale_block(in_con, in_dim, out_dim, is_bn=False):
    """
    多尺度块MSB
    :param in_con: 输入tensor变量 [batch_size, filter_w, filter_h, in_dim]
    :param in_dim: 输入通道数
    :param out_dim: 输出通道数
    :param is_bn: 是否增加Batch Normal
    :return: 输出tensor变量 [4 * batch_size, filter_w, filter_h, in_dim]
    """
    with tf.variable_scope('con_9') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[9, 9, in_dim, out_dim],
                                             stddev=0.01, wd=0.0005)
        con_9 = tf.nn.conv2d(in_con, kernel, [1, 1, 1, 1], padding='SAME', name=scope.name)
        _activation_summary(con_9)

    with tf.variable_scope('con_7') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[7, 7, in_dim, out_dim],
                                             stddev=0.01, wd=0.0005)
        con_7 = tf.nn.conv2d(in_con, kernel, [1, 1, 1, 1], padding='SAME', name=scope.name)
        _activation_summary(con_7)

    with tf.variable_scope('con_5') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, in_dim, out_dim],
                                             stddev=0.01, wd=0.0005)
        con_5 = tf.nn.conv2d(in_con, kernel, [1, 1, 1, 1], padding='SAME', name=scope.name)
        _activation_summary(con_5)

    with tf.variable_scope('con_3') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, in_dim, out_dim],
                                             stddev=0.01, wd=0.0005)
        con_3 = tf.nn.conv2d(in_con, kernel, [1, 1, 1, 1], padding='SAME', name=scope.name)
        _activation_summary(con_3)

    with tf.variable_scope('concat') as scope:
        concat = tf.concat([con_9, con_7, con_5, con_3], 3, name=scope.name)
        biases = _variable_on_cpu('biases', [out_dim * 4], tf.constant_initializer(0))
        bias = tf.nn.bias_add(concat, biases)

        if is_bn:
            bn = BatchNorm()
            bias = bn(bias)

        msb = tf.nn.relu(bias)
        _activation_summary(msb)

    return msb


def inference(images):
    """
    构建MSCNN模型
    :param images: 原始图像
    :return: 人群密度估计图像
    """
    # -------------------------------------------------------------------------------------------- #
    # 创建模型
    # con1_1
    with tf.variable_scope('con1') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[9, 9, 3, 64],
                                             stddev=0.01, wd=0.0005)
        con = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME', name=scope.name)
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0))
        bias = tf.nn.bias_add(con, biases)
        con1 = tf.nn.relu(bias)
        _activation_summary(con1)

    # msb_con2
    with tf.variable_scope('msb_con2'):
        msb_con2 = multi_scale_block(con1, 64, 16)

    # pool_msb_con2
    with tf.variable_scope('pool_msb_con2') as scope:
        pool_msb_con2 = tf.nn.max_pool(msb_con2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                       name=scope.name)

    # msb_con3
    with tf.variable_scope('msb_con3'):
        msb_con3 = multi_scale_block(pool_msb_con2, 64, 32)

    # msb_con4
    with tf.variable_scope('msb_con4'):
        msb_con4 = multi_scale_block(msb_con3, 128, 32)

    # pool_msb_con4
    with tf.variable_scope('pool_msb_con4') as scope:
        pool_msb_con4 = tf.nn.max_pool(msb_con4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                       name=scope.name)

    # msb_con5
    with tf.variable_scope('msb_con5'):
        msb_con5 = multi_scale_block(pool_msb_con4, 128, 64)

    # msb_con6
    with tf.variable_scope('msb_con6'):
        msb_con6 = multi_scale_block(msb_con5, 256, 64)

    # mpl_con7
    with tf.variable_scope('mpl_con7') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[1, 1, 256, 1000], stddev=0.001, wd=0.0005)
        con = tf.nn.conv2d(msb_con6, kernel, [1, 1, 1, 1], padding='SAME', name=scope.name)
        biases = _variable_on_cpu('biases', [1000], tf.constant_initializer(0))
        bias = tf.nn.bias_add(con, biases)
        mpl_con7 = tf.nn.relu(bias)
        _activation_summary(mpl_con7)

    # con_out
    with tf.variable_scope('con_out') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[1, 1, 1000, 1], stddev=0.001, wd=0.0005)
        con = tf.nn.conv2d(mpl_con7, kernel, [1, 1, 1, 1], padding='SAME', name=scope.name)
        biases = _variable_on_cpu('biases', [1], tf.constant_initializer(0))
        bias = tf.nn.bias_add(con, biases)

        con_out = tf.nn.relu(bias)
        _activation_summary(con_out)

    # 删除第四维度channel, channel=1
    image_out = con_out

    tf.summary.image("con_img", image_out)

    return image_out


def inference_bn(images):
    """
    在MSCNN模型的cnn层后增加Batch Normal; 对输出的激活函数进行了改进f(x)=relu(sigmoid(x))
    $$sigmod(x)=\frac{1}{1+e^{-x}}$$
    $$relu(x)=
    \begin{equation}
    \begin{cases}
     x, & x \geq 0 \\
    0, & x < 0
    \end{cases}
    \end{equation}$$
    $$f(x)=relu(sigmod(x))$$

    :param images: 原始图像
    :return: 人群密度估计图像
    """
    # -------------------------------------------------------------------------------------------- #
    # 创建模型
    # con1_1
    with tf.variable_scope('con1') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[9, 9, 3, 64],
                                             stddev=0.01, wd=0.0005)
        con = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME', name=scope.name)
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0))
        bias = tf.nn.bias_add(con, biases)
        con1 = tf.nn.relu(bias)
        _activation_summary(con1)

    # msb_con2
    with tf.variable_scope('msb_con2'):
        msb_con2 = multi_scale_block(con1, 64, 16, is_bn=True)

    # pool_msb_con2
    with tf.variable_scope('pool_msb_con2') as scope:
        pool_msb_con2 = tf.nn.max_pool(msb_con2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                       name=scope.name)

    # msb_con3
    with tf.variable_scope('msb_con3'):
        msb_con3 = multi_scale_block(pool_msb_con2, 64, 32, is_bn=True)

    # msb_con4
    with tf.variable_scope('msb_con4'):
        msb_con4 = multi_scale_block(msb_con3, 128, 32, is_bn=True)

    # pool_msb_con4
    with tf.variable_scope('pool_msb_con4') as scope:
        pool_msb_con4 = tf.nn.max_pool(msb_con4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                       name=scope.name)

    # msb_con5
    with tf.variable_scope('msb_con5'):
        msb_con5 = multi_scale_block(pool_msb_con4, 128, 64, is_bn=True)

    # msb_con6
    with tf.variable_scope('msb_con6'):
        msb_con6 = multi_scale_block(msb_con5, 256, 64, is_bn=True)

    # mpl_con7
    with tf.variable_scope('mpl_con7') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[1, 1, 256, 1000], stddev=0.001, wd=0.0005)
        con = tf.nn.conv2d(msb_con6, kernel, [1, 1, 1, 1], padding='SAME', name=scope.name)
        biases = _variable_on_cpu('biases', [1000], tf.constant_initializer(0))
        bias = tf.nn.bias_add(con, biases)
        mpl_con7 = tf.nn.relu(bias)
        _activation_summary(mpl_con7)

    # con_out
    with tf.variable_scope('con_out') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[1, 1, 1000, 1], stddev=0.001, wd=0.0005)
        con = tf.nn.conv2d(mpl_con7, kernel, [1, 1, 1, 1], padding='SAME', name=scope.name)
        biases = _variable_on_cpu('biases', [1], tf.constant_initializer(0))
        bias = tf.nn.bias_add(con, biases)

        bn = BatchNorm()
        bias = bn(bias)

        con_out = tf.nn.relu(tf.nn.sigmoid(bias))
        _activation_summary(con_out)

    # 删除第四维度channel, channel=1
    image_out = con_out

    tf.summary.image("con_img", image_out)

    return image_out


def loss(predict, label):
    """
    计算损失
    :param predict: mscnn估计密度图
    :param label: ground truth crowd counting map
    :return: L2 loss
    """
    # L2 Loss
    predict = tf.squeeze(predict, 3)
    l2_loss = tf.reduce_sum((predict - label) * (predict - label))

    # 增加概要
    tf.summary.histogram('loss', l2_loss)

    return l2_loss


def add_avg_loss(avg_loss):
    """
    计算平均损失
    :param avg_loss:
    :return:
    """
    add_avg_loss_op = avg_loss * 1
    tf.summary.histogram('avg_loss', avg_loss)

    return add_avg_loss_op


def _add_loss_summaries(total_loss):
    """
    增加损失概要信息
    :param total_loss:
    :return:
    """
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def train(total_loss, global_step, nums_per_train):
    """
    根据损失构建RMSProp优化算子
    :param total_loss: 损失
    :param global_step:
    :param nums_per_train:
    :return: RMSProp优化算子
    """
    num_batches_per_epoch = nums_per_train / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * mscnn_train.num_epochs_per_decay)

    lr = tf.train.exponential_decay(mscnn_train.initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    mscnn_train.learning_rate_per_decay,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # 优化算法
    opt = tf.train.RMSPropOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

    # 应用梯度
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    train_op = apply_gradient_op

    # 添加概要
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    return train_op
