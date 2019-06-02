##########################全批量梯度下降法
##########################例：最小二乘法一元线性回归模型
########################## Reference: http://sofasofa.io/tutorials/python_gradient_descent/3.php
import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submit = pd.read_csv('sample_submit.csv')

#初始设置
beta = [1, 1]#初值
alpha = 0.2#步长
tol_L = 0.1#容忍度

# 对x进行归一化
max_x = max(train['id'])
x = train['id'] / max_x
y = train['questions']

#根据最小二乘公式计算某一点上的两个beta参数的梯度
def compute_grad(beta, x, y):
    grad = [0, 0]
    grad[0] = 2. * np.mean(beta[0] + beta[1] * x - y)
    grad[1] = 2. * np.mean(x * (beta[0] + beta[1] * x - y))
    return np.array(grad)

#对参数进行一次更新
def update_beta(beta, alpha, grad):
    new_beta = np.array(beta) - alpha * grad
    return new_beta

# 定义计算RMSE的函数
def rmse(beta, x, y):
    squared_err = (beta[0] + beta[1] * x - y) ** 2
    res = np.sqrt(np.mean(squared_err))
    return res

# 进行第一次计算
grad = compute_grad(beta, x, y)
loss = rmse(beta, x, y)
beta = update_beta(beta, alpha, grad)
loss_new = rmse(beta, x, y)#经过一次迭代，损失下降了

# 开始迭代
i = 1
while np.abs(loss_new - loss) > tol_L:
    beta = update_beta(beta, alpha, grad)
    grad = compute_grad(beta, x, y)
    loss = loss_new
    loss_new = rmse(beta, x, y)
    i += 1
    print('Round %s Diff RMSE %s'%(i, abs(loss_new - loss)))
print('Coef: %s \nIntercept %s'%(beta[1], beta[0]))

#经过118次迭代，最终达到收敛条件

# 由于我们对x进行了归一化，上面得到的Coef其实是真实的系数乘以max_x。
# 我们可以还原得到最终的回归系数。
print('Our Coef: %s \nOur Intercept %s'%(beta[1] / max_x, beta[0]))
# Our Coef: 2.12883541445 
# Our Intercept 1015.70899949

res = rmse(beta, x, y)
print('Our RMSE: %s'%res)
#Our RMSE: 533.59831397379


#我们可以用标准模块sklearn.linear_model.LinearRegression来检验我们通过梯度下降法得到的系数。
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train[['id']], train[['questions']])
print('Sklearn Coef: %s'%lr.coef_[0][0])
print('Sklearn Coef: %s'%lr.intercept_[0])
#结果比较接近
#Sklearn Coef: 2.19487084445
#Sklearn Coef: 936.051219649

res = rmse([lr.intercept_[0], lr.coef_[0][0]], train['id'], y)
print('Sklearn RMSE: %s'%res)
#Sklearn RMSE: 531.841307949
#RMSE的预测结果也比较接近