# -*- coding:utf-8 -*-
"""
@Function: MSCNN crowd counting model training
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

# 系统库
import os.path
import random
import cv2
from six.moves import xrange

# 机器学习库
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np

# 项目库
import mscnn

# 模型参数设置
FLAGS = tf.app.flags.FLAGS

# 模型训练参数
num_epochs_per_decay = 20
learning_rate_per_decay = 0.9
initial_learning_rate = 1.0e-5


def train():
    """
    在ShanghaiTech测试集上对mscnn模型训练
    :return:
    """
    with tf.Graph().as_default():
        # 读取文件目录txt
        dir_file = open(FLAGS.data_train_index)
        dir_name = dir_file.readlines()

        # 参数设置
        nums_train = len(dir_name)  # 训练一批次的图片数量
        global_step = tf.Variable(0, trainable=False)  # 定义全局衰减步数

        # 用于训练数据的place_holder
        image = tf.placeholder("float")
        label = tf.placeholder("float")
        avg_loss1 = tf.placeholder("float")

        # 模型训练相关的初始化工作
        # predicts = mscnn.inference(image)  # 构建mscnn模型
        predicts = mscnn.inference_bn(image)  # 构建改进mscnn模型
        loss = mscnn.loss(predicts, label)  # 计算损失
        train_op = mscnn.train(loss, global_step, nums_train)  # 获取训练算子

        sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))  # 创建一个会话
        saver = tf.train.Saver(tf.all_variables())  # 创建保存器

        init = tf.initialize_all_variables()  # 变量初始化
        sess.run(init)  # 初始化模型所有变量

        checkpoint_dir = tf.train.get_checkpoint_state(FLAGS.model_dir)
        if checkpoint_dir and checkpoint_dir.model_checkpoint_path:
            saver.restore(sess, checkpoint_dir.model_checkpoint_path)
        else:
            print('Not found checkpoint file')

        summary_op = tf.summary.merge_all()  # 概要汇总
        add_avg_loss_op = mscnn.add_avg_loss(avg_loss1)  # 添加平均loss的op
        summary_writer = tf.summary.FileWriter(FLAGS.train_log, graph_def=sess.graph_def)  # 创建一个概要器

        # 参数设置
        steps = 100000
        avg_loss = 0

        for step in xrange(steps):
            # 批次数
            index = step % nums_train
            tmp1 = None
            while step % nums_train == 0:
                tmp = range(nums_train)
                tmp1 = random.sample(tmp, nums_train)
                break

            num_batch = tmp1[index]
            file_name = dir_name[num_batch]
            im_name, gt_name = file_name.split(' ')
            gt_name = gt_name.split('\n')[0]  # 分离出最后一个回车符

            # 训练数据(图片)
            batch_xs = cv2.imread(FLAGS.data_train_im + im_name)
            batch_xs = np.array(batch_xs, dtype=np.float32)
            batch_xs = batch_xs.reshape(1, len(batch_xs), -1, 3)

            # 训练数据(密度图)
            batch_ys = np.array(np.load(FLAGS.data_train_gt + gt_name))
            batch_ys = np.array(batch_ys, dtype=np.float32)
            batch_ys = batch_ys.reshape(1, len(batch_ys), -1)

            # 获取损失值以及预测密度图
            _, loss_value = sess.run([train_op, loss], feed_dict={image: batch_xs, label: batch_ys})
            output = sess.run(predicts, feed_dict={image: batch_xs})
            avg_loss += loss_value

            # 保存概述数据
            if step % 100 == 0:
                summary_str = sess.run(summary_op, feed_dict={image: batch_xs, label: batch_ys,
                                                              avg_loss1: avg_loss / 100})
                summary_writer.add_summary(summary_str, step)
                avg_loss = 0

            if step % 10 == 0:
                print("avg_loss:%.7f\t counting:%.7f\t predict:%.7f" % \
                      (loss_value, sum(sum(sum(batch_ys))), sum(sum(sum(output)))))
                sess.run(add_avg_loss_op, feed_dict={avg_loss1: loss_value})

            # 保存模型参数
            if step % 2000 == 0:
                checkpoint_path = os.path.join(FLAGS.model_dir, 'skip_mcnn.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

            # 输出预测密度图(用于测试)
            if step % 500 == 0:
                out_path = os.path.join(FLAGS.output_dir, str(step) + "out.npy")
                np.save(out_path, output)


def main(argv=None):
    if gfile.Exists(FLAGS.train_log):
        gfile.DeleteRecursively(FLAGS.train_log)
    gfile.MakeDirs(FLAGS.train_log)

    if not gfile.Exists(FLAGS.model_dir):
        gfile.MakeDirs(FLAGS.model_dir)

    if not gfile.Exists(FLAGS.output_dir):
        gfile.MakeDirs(FLAGS.output_dir)

    train()


if __name__ == '__main__':
    tf.app.run()
