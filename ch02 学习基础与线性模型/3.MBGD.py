# 综合前两者的优缺点
# 小批量随机梯度下降法（Mini-batch Stochastic Gradient Decent）是对速度和稳定性进行妥协后的产物。
# 样本大小定为b
# 当b=1时，小批量随机下降法就等价与SGD；当b=N时，小批量就等价于全批量。所以小批量梯度下降法的效果也和b的选择相关，这个数值被称为批量尺寸(batch size)。
# 如何确定b 参见 https://arxiv.org/abs/1609.04836

def compute_grad_batch(beta, batch_size, x, y):
    grad = [0, 0]
    r = np.random.choice(range(len(x)), batch_size, replace=False)
    grad[0] = 2. * np.mean(beta[0] + beta[1] * x[r] - y[r])
    grad[1] = 2. * np.mean(x[r] * (beta[0] + beta[1] * x[r] - y[r]))
    return np.array(grad)
