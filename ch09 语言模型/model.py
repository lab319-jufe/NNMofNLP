import tensorflow as tf

'''
修改成类
class NNLM(object):
    def __init__(self, config, training=True):
        self.config = config
        self.training = training
    
    def NNLM(self, input_data):
        embedding = tf.get_variable(name="embedding", shape=[self.config.vocab_size, self.config.embedding_size])
        inputs = tf.nn.embedding_lookup(embedding, ids=input_data)
        # 输入数据接dropout层
        if config.Training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.BasicLSTMCell(config.hidden_size) for _ in range(config.number_layers)])
        inital_state = cell.zero_state(config.batch_size, tf.float32)
        state = inital_state
        # 构建LSTM网络
        # outputs, states = tf.nn.dynamic_rnn(cell, inputs, src_size, dtype=tf.float32)
        outputs = []
        with tf.variable_scope('RNN'):
            for time_step in range(config.num_step):
                if time_step > 0: 
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
        # 把输出队列展开成[batch, hidden_size * num_steps]的形状，
        # 然后再reshape成[batch*num_steps, hidden_size]的形状
        output = tf.reshape(tf.concat(outputs, 1), shape=[-1, config.hidden_size])
        # softmax层分类
        if config.share_emb_and_softmax:
            weight = tf.transpose(embedding)
        else:
            weight = tf.get_variable('weight', [config.hidden_size,config.vocab_size])
        bias = tf.get_variable("bias", [config.vocab_size])
        logits = tf.nn.xw_plus_b(output, weight, bias)
        # Reshape logits to be a 3-D tensor for a sequence loss
        #logits = tf.reshape(logits, [-1, config.num_step, config.vocab_size])
        return logits
'''

def NNLM(input_data, config):
    embedding = tf.get_variable(name="embedding", shape=[config.vocab_size, config.embedding_size])
    inputs = tf.nn.embedding_lookup(embedding, ids=input_data)
    # 输入数据接dropout层
    if config.Training and config.keep_prob < 1:
        inputs = tf.nn.dropout(inputs, config.keep_prob)
    cell = tf.nn.rnn_cell.MultiRNNCell(
        [tf.nn.rnn_cell.BasicLSTMCell(config.hidden_size) for _ in range(config.number_layers)])
    inital_state = cell.zero_state(config.batch_size, tf.float32)
    state = inital_state
    # 构建LSTM网络
    # outputs, states = tf.nn.dynamic_rnn(cell, inputs, src_size, dtype=tf.float32)
    outputs = []
    with tf.variable_scope('RNN'):
        for time_step in range(config.num_step):
            if time_step > 0: 
                tf.get_variable_scope().reuse_variables()
            (cell_output, state) = cell(inputs[:, time_step, :], state)
            outputs.append(cell_output)
    # 把输出队列展开成[batch, hidden_size * num_steps]的形状，
    # 然后再reshape成[batch*num_steps, hidden_size]的形状
    output = tf.reshape(tf.concat(outputs, 1), shape=[-1, config.hidden_size])
    # softmax层分类
    if config.share_emb_and_softmax:
        weight = tf.transpose(embedding)
    else:
        weight = tf.get_variable('weight', [config.hidden_size,config.vocab_size])
    bias = tf.get_variable("bias", [config.vocab_size])
    logits = tf.nn.xw_plus_b(output, weight, bias)
    # Reshape logits to be a 3-D tensor for a sequence loss
    #logits = tf.reshape(logits, [-1, config.num_step, config.vocab_size])
    return logits


def cost(logits, labels, config):
    # logits为神经网络层的输出，shape为[batch_size, num_step, num_classes]
    # label为一个二维的向量，shape为[batch_size, num_step]
    # 如果label已经是one-hot格式，则采用tf.nn.softmax_cross_entroy_with_logits()
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(labels, [-1]), logits=logits)
    # cost 已经在batch上取平均，后边计算困惑度要注意
    cost = tf.reduce_sum(loss) / config.batch_size
    '''
    if not config.Training:
        trainable_variables = tf.trainable_variables()
        # 控制梯度大小，定义优化方法和训练步骤
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, trainable_variables), config.max_grad_norm)
        optimizer = tf.train.AdamOptimizer(learning_rate=config.lr)
        train_op = optimizer.apply_gradients(zip(grads, trainable_variables))
    '''
    # 自适应学习率
    #global_step = tf.Variable(0, trainable=False)
    # tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=False, name=None
    # decayed_learning_rate = learning_rate × decay_rate(global_step / decay_steps)
    # 每走1000步，学习率调整为原来的0.5倍,指数部分采用整除策略
    #tf.train.exponential_decay(learning_rate=config.lr, global_step=global_step, decay_steps=1000, decay_rate=0.5, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=config.lr)
    train_op = optimizer.minimize(cost)
    return cost, train_op