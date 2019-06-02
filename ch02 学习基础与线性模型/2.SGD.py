#Reference: http://sofasofa.io/tutorials/python_gradient_descent/4.php
#对于最小二乘法来说，每计算一次梯度的代价是O(N)，运算次数与N成线性关系。
#而随机梯度下降法能将计算一次梯度的代价降低到O(1)，也就是运算次数为常数次，与N无关。

import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submit = pd.read_csv('sample_submit.csv')

# 初始设置
beta = [1, 1]
alpha = 0.2
tol_L = 0.1

# 对x进行归一化
max_x = max(train['id'])
x = train['id'] / max_x
y = train['questions']

def compute_grad_SGD(beta, x, y):
    grad = [0, 0]
    r = np.random.randint(0, len(x))
    grad[0] = 2. * np.mean(beta[0] + beta[1] * x[r] - y[r])
    grad[1] = 2. * np.mean(x[r] * (beta[0] + beta[1] * x[r] - y[r]))
    return np.array(grad)

# 定义更新beta的函数
def update_beta(beta, alpha, grad):
    new_beta = np.array(beta) - alpha * grad
    return new_beta

# 定义计算RMSE的函数
def rmse(beta, x, y):
    squared_err = (beta[0] + beta[1] * x - y) ** 2
    res = np.sqrt(np.mean(squared_err))
    return res

# 进行第一次计算
np.random.seed(10)
grad = compute_grad_SGD(beta, x, y)
loss = rmse(beta, x, y)
beta = update_beta(beta, alpha, grad)
loss_new = rmse(beta, x, y)

# 开始迭代
i = 1
while np.abs(loss_new - loss) > tol_L:
    beta = update_beta(beta, alpha, grad)
    grad = compute_grad_SGD(beta, x, y)
    if i % 100 == 0:
        loss = loss_new
        loss_new = rmse(beta, x, y)
        print('Round %s Diff RMSE %s'%(i, abs(loss_new - loss)))
    i += 1

print('Coef: %s \nIntercept %s'%(beta[1], beta[0]))
#Coef: 4636.290683345392 
#Intercept 805.078482892621
#经过28900次迭代（对于全批量梯度下降法来说相当于是28900/2253=12.83次迭代）

#真实的参数为
print('Our Coef: %s \nOur Intercept %s'%(beta[1] / max_x, beta[0]))
#Our Coef: 2.057829863890542 
#Our Intercept 805.078482892621

#训练误差为
res = rmse(beta, x, y)
print('Our RMSE: %s'%res)
#Our RMSE: 610.1332779883645
