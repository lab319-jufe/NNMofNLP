import tensorflow as tf
import os
import math
import numpy as np
from tqdm import tqdm

os.chdir(r'E:\OneDrive\Documents\Ma&St-learning\NNMofNLP\ch09 语言模型')
from dataset import Dataset
from model import NNLM,cost
from Config import Config

### 修改使其自动调整学习率

config = Config()

# 测试有没有“model”文件夹，如果没有，则新建
if os.path.exists(os.path.join(config.save_path)):
    os.mkdir(config.save_path)

def MakeDataset(dataset, num_batches, batch_size, num_step, shuffle=False):
    # 将数据整理称为维度为[batch_size, num_batches * num_step]的数组
    # 这样处理可能会漏掉一些数据没有用
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
准备训练集（采用yield方式）
'''
with  open(os.path.join(config.ptb_dir, config.ptb_train)) as fin:
    # 将整个文档读进一个长字符串
    train_id_string = ' '.join([line.strip() for line in fin.readlines()])
# 实际上是id_list
train_set = [int(w) for w in train_id_string.split()]  # 将读取的单词编号转为整数
# 计算总的batch数量，每个batch包含的单词数量是batch_size * num_step
# len(train_set) - 1是考虑到label的制作
train_num_batches = (len(train_set) - 1) // (config.batch_size * config.num_step)

'''
准备验证集
'''
with open(os.path.join(config.ptb_dir, config.ptb_valid)) as fin:
    # 将整个文档读进一个长字符串
    valid_id_string = ' '.join([line.strip() for line in fin.readlines()])
# 实际上是id_list
valid_set = [int(w) for w in valid_id_string.split()]  # 将读取的单词编号转为整数
# 计算总的batch数量，每个batch包含的单词数量是batch_size * num_step
valid_num_batches = (len(valid_set) - 1) // (config.batch_size * config.num_step)


# input_data的类型一定要定义为tf.int32,tf.nn.embedding_lookup(embedding, ids=input_data)要求的
input_data = tf.placeholder(dtype=tf.int32, shape=[None, config.num_step], name="input_data")
labels = tf.placeholder(dtype=tf.int32, shape=[None, config.num_step], name="labels")
logits = NNLM(input_data, config)
loss, train_op = cost(logits, labels, config)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 训练数据生成器
    train_gen = MakeDataset(train_set, train_num_batches, config.batch_size, config.num_step)
    # 验证数据生成器
    valid_gen = MakeDataset(valid_set, valid_num_batches, config.batch_size, config.num_step)
    for epoch in range(config.epoches):
        # 训练数据拟合
        iters = 0
        total_costs = 0
        for i in range(train_num_batches):
            X, y = train_gen.__next__()
            cost, _ = sess.run([loss, train_op], feed_dict={input_data:X, labels:y})
            iters += config.num_step
            total_costs += cost
            perlexity = np.exp(total_costs / iters)
            if 0 == i%20:
                print("epoch:%d-----iter:%d-----loss:%f-------perplexity:%f" % (epoch, i, cost, perlexity))
        # 每20个epoch进行一次验证
        if 0 == (epoch+1) % 10:
            iters = 0
            total_costs = 0
            for i in range(valid_num_batches):
                X, y = valid_gen.__next__()
                cost = sess.run(loss, feed_dict={input_data:X, labels:y})
                iters += config.num_step
                total_costs += cost
                iters += config.num_step
                perlexity = np.exp(total_costs / iters)
            print("valid perplexity:", perlexity)
    saver = tf.train.Saver()
    saver.save(sess,save_path=os.path.join(config.save_path, config.model_name))