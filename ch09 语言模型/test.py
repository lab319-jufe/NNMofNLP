import tensorflow as tf
import os
import math
import numpy as np
from tqdm import tqdm
from dataset import Dataset
from model import NNLM,cost
from Config import Config

config = Config()
# test模式
config.Training = False

# 测试有没有“model”文件夹，如果没有，则新建
'''
if not os.path.exists(os.path.join(config.save_path, config.model_name)):
    print("model not found")
'''
def MakeDataset(dataset, num_batches, batch_size, num_step, shuffle=False):
    # 将数据整理称为一个维度为[batch_size, num_batches * num_step]
    data = np.array(dataset[: num_batches * batch_size * num_step])
    data = np.reshape(data, [batch_size, num_batches * num_step])
    # 沿着第二个维度将数据切分成num_batches个batch,存入一个数组
    data_batches = np.split(data, num_batches, axis=1)

    # 重复上述操作，但是每个位置向右移动一位，这里得到的时RNN每一步输出所需要预测的下一个单词
    label = np.array(dataset[1: num_batches * batch_size * num_step + 1])
    label = np.reshape(label, [batch_size, num_batches * num_step])
    label_batches = np.split(label, num_batches, axis=1)
    # 得到一个长度为num_batches的数组，其中每一项包含一个data矩阵和一个label矩阵
    # 直接返回使用或者yield生产都可以
    while True:
        for idx in range(num_batches):
            yield data_batches[idx], label_batches[idx]

'''
准备测试集（采用yield方式）
'''
with  open(os.path.join(config.ptb_dir, config.ptb_test)) as fin:
    # 将整个文档读进一个长字符串
    test_id_string = ' '.join([line.strip() for line in fin.readlines()])
# 实际上是id_list
test_set = [int(w) for w in test_id_string.split()]  # 将读取的单词编号转为整数
# 计算总的batch数量，每个batch包含的单词数量是batch_size * num_step
test_num_batches = (len(test_set) - 1) // (config.batch_size * config.num_step)

# input_data的类型一定要定义为tf.int32,tf.nn.embedding_lookup(embedding, ids=input_data)要求的
input_data = tf.placeholder(dtype=tf.int32, shape=[None, config.num_step], name="input_data")
labels = tf.placeholder(dtype=tf.int32, shape=[None, config.num_step], name="labels")
logits = NNLM(input_data, config)
loss, train_op = cost(logits, labels, config)
saver = tf.train.Saver()
model_file = tf.train.latest_checkpoint(config.save_path)
print("==============================="+model_file)
with tf.Session() as sess:
    # 加载模型
    saver.restore(sess, model_file)
    # 训练数据生成器
    test_gen = MakeDataset(test_set, test_num_batches, config.batch_size, config.num_step)
    # 训练数据拟合
    iters = 0
    total_costs = 0
    for i in range(test_num_batches):
        X, y = test_gen.__next__()
        cost = sess.run(loss, feed_dict={input_data:X, labels:y})
        iters += config.num_step
        total_costs += cost
        perlexity = np.exp(total_costs / iters)
        if 0 == i % 20:
            print("iter:%d-----loss:%f-------perplexity:%f" % (i, total_costs, perlexity))
    print("loss:%f,perplexity:%f" % (total_costs, perlexity))