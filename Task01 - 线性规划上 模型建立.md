# Task01 - 线性规划（上）模型建立 

根据已经提出的题目简介，本项目的核心是最优决策问题，即使得最终总费用最小

在决策中需要考虑诸多关系错综复杂的限制条件：

- 管线通道选择（A1\A2...\A15)
- 管道提供钢厂（S1\S2...\S7)
- 钢管从钢厂运往各节点的运行成本：1单位钢管每公里0.1万元
- 管道铺设费用：每公里0.1万元



在初看题目中，会发现问题复杂烦乱，需要首先考虑简化并建立模型

因此，先简化题目到最简单的情况：节点少、钢厂少、路线上（三少），解决以后再慢慢还原题目



## 建立优化模型过程

**1.1	确定目标函数**

令 **𝑊**𝑡𝑜𝑡𝑎𝑙 表示总费用**,𝑊**𝑜𝑟𝑑𝑒𝑟,**𝑊**𝑡𝑟𝑎𝑛𝑠分别表示订购与运输费用,, 那么我们的目标即是求出

min **𝑊**𝑡𝑜𝑡𝑎𝑙= min (**𝑊**𝑜𝑟𝑑𝑒𝑟+**𝑊**𝑡𝑟𝑎𝑛𝑠)

**1.2	确定决策变量**

列出**𝑊**𝑜𝑟𝑑𝑒𝑟,**𝑊**𝑡𝑟𝑎𝑛𝑠的表达公式，并将其带入目标函数中，得到目标函数关于决策变量的解析式

**1.3	确定约束关系**

定义决策变量的范围区间（>=0 or/and <=?)，以及不同决策变量间的相互关系



基于Scipy优化工具，进行简单线性规划的求解：

```python
import numpy as np												# 导入使用到的lib
from scipy.optimize import minimize

fun = lambda x : (228.3 * x[0]) + (253.1 * x[1])				# 定义目标函数

cons = ({'type': 'eq', 'fun': lambda x: x[0] + x[1] - 2000}, 	# 等式约束，x0+x1-2000 = 0
        {'type': 'ineq', 'fun': lambda x: x[0]}, 				# 不等式约束 x0 >=0 
        {'type': 'ineq', 'fun': lambda x: x[1]},
        {'type': 'ineq', 'fun': lambda x: 1000 - x[0]}, 		# 1600-x0 >= 0
        {'type': 'ineq', 'fun': lambda x: 1600 - x[1]}
       )

# eq 表示函数结果等于0； ineq 表示表达式结果大于等于0

x0 = np.array((1.0, 1.0)) 										# 设置初始值(随机设置即可)

res = minimize(fun, x0, method='SLSQP', constraints=cons) 		# 调用最小值模块 minimize()
# SLSQP	序列最小二乘二次规划

# minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)

# fun：目标函数，返回单值

# x0：初始迭代值

# args：要输入到目标函数中的参数

# method：求解的算法，目前可选的有‘Nelder-Mead’、‘Powell’、‘CG’、‘BFGS’、‘Newton-CG’、‘L-BFGS-B’、‘TNC’、‘COBYLA’、‘SLSQP’、‘dogleg’、‘trust-ncg’ （ver 0.14.0支持自定义算法）

# jac：目标函数的雅可比矩阵。可选项，仅适用于CG，BFGS，Newton-CG，L-BFGS-B，TNC，SLSQP，dogleg，trust-ncg。如果jac是布尔值并且为True，则假定fun与目标函数一起返回梯度。如果为False，将以数字方式估计梯度。jac也可以返回目标的梯度。此时，它的参数必须与fun相同。

# hess，hessp：可选项，目标函数的Hessian(二阶导数矩阵)或目标函数的Hessian乘以任意向量p。仅适用于Newton-CG，dogleg，trust-ncg。

# bounds：可选项，变量的边界(仅适用于L-BFGS-B，TNC和SLSQP)。以(min，max)对的形式定义 x 中每个元素的边界。如果某个参数在 min 或者 max 的一个方向上没有边界，则用 None 标识。如(None, max)

# constraints：约束条件(只对 COBYLA 和 SLSQP)。dict 类型。

# tol：迭代停止的精度。

# callback(xk)：每次迭代要回调的函数，需要有参数 xk

# options：其他选项

print("最优值：", res.fun)
print("最优解：", res.x)


```



完成简单过程对象的模型建立之后，开始转向考虑更加复杂的线性规划模型

第一步：简单增加节点数（即引入变量数组）

第二步：完全还原“三多”

7个钢厂(S)	  i = 1,2,3...7

15个节点(A)	j = 1,2,3...15

Zi,j-			 第i个钢厂运输到节点Aj后，沿左侧铺设的钢管量

Zi,j+			第i个钢厂运输到节点Aj后，沿右侧铺设的钢管量

由于起点和终点各有一边不需要铺设，Zi,1- = Zi,15+ = 0

pi		钢管厂Si的单位订购价（i = 1,2,3...7)

si		钢管厂Si的订购上限（i = 1,2,3...7)

Ci,j	  钢管厂Si到Aj的公路最短路线费用 = 0.1*C‘i,j   (C'i,j为最短运输路线)

通过计算出Ci,j以后实现其求解

求解过程：

```python
# 请写下你的代码
import math
 
nodes = ('A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11'
         ,'A12','A13','A14','A15','A16','A17','A18','A19','A20','A21','A22','A23','A24',
        'A25','A26','A27','A28','A29','A30','A31','A32','A33','A34','A35','A36','A37','A38')
# dis矩阵为方阵
dis = [[0,3,450,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf],
       [3,0,math.inf,math.inf,301,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf],
       [450,math.inf,0,80,math.inf,1150,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf],
       [math.inf,math.inf,80,0,2,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf],
       [math.inf,301,math.inf,2,0,math.inf,750,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf],
       [math.inf,math.inf,1150,math.inf,math.inf,0,600,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,1100,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf],
       [math.inf,math.inf,math.inf,math.inf,750,600,0,math.inf,606,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf],
       [math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,0,10,306,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf],
       [math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,606,10,0,math.inf,194,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf],
       [math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,306,math.inf,0,5,195,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf],
       [math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,194,5,0,math.inf,205,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf],
       [math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,195,math.inf,0,10,math.inf,math.inf,20,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf],
       [math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,205,10,0,math.inf,math.inf,31,201,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf],
       [math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,0,1200,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf],
       [math.inf,math.inf,math.inf,math.inf,math.inf,1100,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,1200,0,202,12,math.inf,720,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf],
       [math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,20,31,202,0,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf],
       [math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,201,12,math.inf,math.inf,0,math.inf,math.inf,680,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf],
       [math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,0,690,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf],
       [math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,720,math.inf,math.inf,690,0,42,520,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf],
       [math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,680,math.inf,42,0,math.inf,480,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf],
       [math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,520,math.inf,0,70,math.inf,170,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf],
       [math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,480,70,0,math.inf,math.inf,math.inf,math.inf,300,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf],
       [math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,0,690,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf],
       [math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,170,math.inf,690,0,88,math.inf,math.inf,160,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf],
       [math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,88,0,462,10,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf],
       [math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,462,0,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf],
       [math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,300,math.inf,math.inf,10,math.inf,0,math.inf,math.inf,220,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf],
       [math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,160,math.inf,math.inf,math.inf,0,70,320,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf],
       [math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,70,0,10,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf],
       [math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,220,math.inf,10,0,math.inf,210,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf],
       [math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,320,math.inf,math.inf,0,62,160,math.inf,math.inf,math.inf,math.inf,math.inf],
       [math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,210,62,0,math.inf,math.inf,420,math.inf,math.inf,math.inf],
       [math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,160,math.inf,0,70,30,290,math.inf,math.inf],
       [math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,70,0,110,math.inf,math.inf,math.inf],
       [math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,420,30,110,0,math.inf,500,math.inf],
       [math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,290,math.inf,math.inf,0,20,30],
       [math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,500,20,0,20],
       [math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,30,20,0]
      ]
 
def shortDistance(dis):
    for i in range(38):         # 十字交叉法的位置位置，先列后行
        for j in range(38):     # 列 表示dis[j][i]的值，即j->i
            for k in range(j+1, 38): # 行 表示dis[i][k]的值，即i->k，i只是一个桥梁而已
                # 先列后行，形成一个传递关系，若比原来距离小，则更新
                if dis[j][k] > dis[j][i] + dis[i][k]:
                    dis[j][k] = dis[j][i] + dis[i][k]
                    dis[k][j] = dis[j][i] + dis[i][k]
 
shortDistance(dis)
 
a = [16,14,18,23,26,34,38]
b = [2,5,7,9,11,13,17,20,22,27,30,32,35,37]
for i in a:
    print()
    for j in b:
        print(dis[i-1][j-1],end=' ')
 
dis1 = [
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
]
 
k = 0
l = 0
for i in a:
    k += 1
    for j in b:
        l = (l%14)+1
        dis1[k][l] = 0.1*dis[i-1][j-1]
        #print(k,l)

```

```python
import math
import numpy as np
from scipy.optimize import minimize

fun = lambda x :((160 * x[0] + 155*x[1] + 155*x[2]+ 160*x[3] +155*x[4] +150*x[5]+ 160*x[6]) 
    + 
        dis1[1][1] * (x[7]+x[8]) +  dis1[1][2] * (x[9]+x[10])+  dis1[1][3] * (x[11]+x[12])+  dis1[1][4] * (x[13]+x[14])+  dis1[1][5] * (x[15]+x[16])+  dis1[1][6] * (x[17]+x[18]) +  dis1[1][7] * (x[19]+x[20])+  
        dis1[1][8] * (x[21]+x[22])+  dis1[1][9] * (x[23]+x[24])+  dis1[1][10] * (x[25]+x[26])+  dis1[1][11] * (x[27]+x[28])+  dis1[1][12] * (x[29]+x[30])+  dis1[1][13] * (x[31]+x[32])+  dis1[1][14] * (x[33]+x[34])
    
    +   dis1[2][1] * (x[35]+x[36])+  dis1[2][2] * (x[37]+x[38])+  dis1[2][3] * (x[39]+x[40])+  dis1[2][4] * (x[41]+x[42])+  dis1[2][5] * (x[43]+x[44]) +  dis1[2][6] * (x[45]+x[46])+ dis1[2][7] * (x[47]+x[48])+
        dis1[2][8] * (x[49]+x[50])+  dis1[2][9] * (x[51]+x[52])+  dis1[2][10] * (x[53]+x[54])+  dis1[2][11] * (x[55]+x[56])+  dis1[2][12] * (x[57]+x[58])+  dis1[2][13] * (x[59]+x[60]) + dis1[2][14] * (x[61]+x[62])
   
    +   dis1[3][1] * (x[63]+x[64])+  dis1[3][2] * (x[65]+x[66])+  dis1[3][3] * (x[67]+x[68])+  dis1[3][4] * (x[69]+x[70])+  dis1[3][5] * (x[71]+x[72]) +  dis1[3][6] * (x[73]+x[74])+ dis1[3][7] * (x[75]+x[76])+
        dis1[3][8] * (x[77]+x[78])+  dis1[3][9] * (x[79]+x[80])+  dis1[3][10] * (x[81]+x[82])+  dis1[3][11] * (x[83]+x[84])+  dis1[3][12] * (x[85]+x[86])+  dis1[3][13] * (x[87]+x[88]) + dis1[3][14] * (x[89]+x[90])
    
    
    +   dis1[4][1] * (x[91]+x[92])+  dis1[4][2] * (x[93]+x[94])+  dis1[4][3] * (x[95]+x[96])+  dis1[4][4] * (x[97]+x[98])+  dis1[4][5] * (x[99]+x[100]) +  dis1[4][6] * (x[101]+x[102])+ dis1[4][7] * (x[103]+x[104])+
        dis1[4][8] * (x[105]+x[106])+  dis1[4][9] * (x[107]+x[108])+  dis1[4][10] * (x[109]+x[110])+  dis1[4][11] * (x[111]+x[112])+  dis1[4][12] * (x[113]+x[114])+  dis1[4][13] * (x[115]+x[116]) + dis1[4][14] * (x[117]+x[118])
        
    +   dis1[5][1] * (x[119]+x[120])+  dis1[5][2] * (x[121]+x[122])+  dis1[5][3] * (x[123]+x[124])+  dis1[5][4] * (x[125]+x[126])+  dis1[5][5] * (x[127]+x[128]) +  dis1[5][6] * (x[129]+x[130])+ dis1[5][7] * (x[131]+x[132])+
        dis1[5][8] * (x[133]+x[134])+  dis1[5][9] * (x[135]+x[136])+  dis1[5][10] * (x[137]+x[138])+  dis1[5][11] * (x[139]+x[140])+  dis1[5][12] * (x[141]+x[142])+  dis1[5][13] * (x[143]+x[144]) + dis1[5][14] * (x[145]+x[146])
        
    +   dis1[6][1] * (x[147]+x[148])+  dis1[6][2] * (x[149]+x[150])+  dis1[6][3] * (x[151]+x[152])+  dis1[6][4] * (x[153]+x[154])+  dis1[6][5] * (x[155]+x[156]) +  dis1[6][6] * (x[157]+x[158])+ dis1[6][7] * (x[159]+x[160])+
        dis1[6][8] * (x[161]+x[162])+  dis1[6][9] * (x[163]+x[164])+  dis1[6][10] * (x[165]+x[166])+  dis1[6][11] * (x[167]+x[168])+  dis1[6][12] * (x[169]+x[170])+  dis1[6][13] * (x[171]+x[172]) + dis1[6][14] * (x[173]+x[174])
        
    +   dis1[7][1] * (x[175]+x[176])+  dis1[7][2] * (x[177]+x[178])+  dis1[7][3] * (x[179]+x[180])+  dis1[7][4] * (x[181]+x[182])+  dis1[7][5] * (x[183]+x[184]) +  dis1[7][6] * (x[185]+x[186])+ dis1[7][7] * (x[187]+x[188])+
        dis1[7][8] * (x[189]+x[190])+  dis1[7][9] * (x[191]+x[192])+  dis1[7][10] * (x[193]+x[194])+  dis1[7][11] * (x[195]+x[196])+  dis1[7][12] * (x[197]+x[198])+  dis1[7][13] * (x[199]+x[200]) + dis1[7][14] * (x[201]+x[202])
        )
cons = ({'type': 'eq', 'fun': lambda x: x[0] - x[7] - x[8] - x[9]- x[10]- x[11] - x[12] - x[13]- x[14]- x[15] - x[16] - x[17]- x[18]- x[19] - x[20] - x[21]- x[22]- x[23] - x[24] - x[25]- x[26]- x[27] - x[28] - x[29]- x[30]- x[31] - x[32] - x[33]- x[34]}, # 等式约束
        {'type': 'eq', 'fun': lambda x: x[1] - x[35] - x[36] - x[37]- x[38]- x[39] - x[40] - x[41]- x[42]- x[43] - x[44] - x[45]- x[46]- x[47] - x[48] - x[49]- x[50]- x[51] - x[52] - x[53]- x[54]- x[55] - x[56] - x[57]- x[58]- x[59] - x[60] - x[61]- x[62]},
        {'type': 'eq', 'fun': lambda x: x[2] - x[63] - x[64] - x[65]- x[66]- x[67] - x[68] - x[69]- x[70]- x[71] - x[72] - x[73]- x[74]- x[75] - x[76] - x[77]- x[78]- x[79] - x[80] - x[81]- x[82]- x[83] - x[84] - x[85]- x[86]- x[87] - x[88] - x[89]- x[90]},
        {'type': 'eq', 'fun': lambda x: x[3] - x[91] - x[92] - x[93]- x[94]- x[95] - x[96] - x[97]- x[98]- x[99] - x[100] - x[101]- x[102]- x[103] - x[104] - x[105]- x[106]- x[107] - x[108] - x[109]- x[110]- x[111] - x[112] - x[113]- x[114]- x[115] - x[116] - x[117]- x[118]},
        {'type': 'eq', 'fun': lambda x: x[4] - x[119] - x[120] - x[121]- x[122]- x[123] - x[124] - x[125]- x[126]- x[127] - x[128] - x[129]- x[130]- x[131] - x[132] - x[133]- x[134]- x[135] - x[136] - x[137]- x[138]- x[139] - x[140] - x[141]- x[142]- x[143] - x[144] - x[145]- x[146]},
        {'type': 'eq', 'fun': lambda x: x[5] - x[147] - x[148] - x[149]- x[150]- x[151] - x[152] - x[153]- x[154]- x[155] - x[156] - x[157]- x[158]- x[159] - x[160] - x[161]- x[162]- x[163] - x[164] - x[165]- x[166]- x[167] - x[168] - x[169]- x[170]- x[171] - x[172] - x[173]- x[174]},
        {'type': 'eq', 'fun': lambda x: x[6] - x[175] - x[176] - x[177]- x[178]- x[179] - x[180] - x[181]- x[182]- x[183] - x[184] - x[185]- x[186]- x[187] - x[188] - x[189]- x[190]- x[191] - x[192] - x[193]- x[194]- x[195] - x[196] - x[197]- x[198]- x[199] - x[200] - x[201]- x[202]},
        
        
        {'type': 'ineq', 'fun': lambda x: 800 - x[0]}, # 不等式约束
        {'type': 'ineq', 'fun': lambda x: 800 - x[1]}, # 不等式约束
        {'type': 'ineq', 'fun': lambda x: 1000 - x[2]}, # 不等式约束
        {'type': 'ineq', 'fun': lambda x: 2000 - x[3]}, # 不等式约束
        {'type': 'ineq', 'fun': lambda x: 2000 - x[4]}, # 不等式约束
        {'type': 'ineq', 'fun': lambda x: 2000 - x[5]}, # 不等式约束
        {'type': 'ineq', 'fun': lambda x: 3000 - x[6]}, # 不等式约束
        
        
        
        {'type': 'eq', 'fun': lambda x: x[0] + x[1]+ x[2]+ x[3]+ x[4]+ x[5]+ x[6] - 5171},
 
        {'type': 'ineq', 'fun': lambda x: x[7]},
        {'type': 'ineq', 'fun': lambda x: x[8]},
        {'type': 'ineq', 'fun': lambda x: x[9]},
        {'type': 'ineq', 'fun': lambda x: x[10]},
        {'type': 'ineq', 'fun': lambda x: x[11]},
        {'type': 'ineq', 'fun': lambda x: x[12]},
        {'type': 'ineq', 'fun': lambda x: x[13]},
        {'type': 'ineq', 'fun': lambda x: x[14]},
        {'type': 'ineq', 'fun': lambda x: x[15]},
        {'type': 'ineq', 'fun': lambda x: x[16]},
        {'type': 'ineq', 'fun': lambda x: x[17]},
        {'type': 'ineq', 'fun': lambda x: x[18]},
        {'type': 'ineq', 'fun': lambda x: x[19]},
        {'type': 'ineq', 'fun': lambda x: x[20]},
        {'type': 'ineq', 'fun': lambda x: x[21]},
        {'type': 'ineq', 'fun': lambda x: x[22]},
        {'type': 'ineq', 'fun': lambda x: x[23]},
        {'type': 'ineq', 'fun': lambda x: x[24]},
        {'type': 'ineq', 'fun': lambda x: x[25]},
        {'type': 'ineq', 'fun': lambda x: x[26]},
        {'type': 'ineq', 'fun': lambda x: x[27]},
        {'type': 'ineq', 'fun': lambda x: x[28]},
        {'type': 'ineq', 'fun': lambda x: x[29]},
        {'type': 'ineq', 'fun': lambda x: x[30]},
        {'type': 'ineq', 'fun': lambda x: x[31]},
        {'type': 'ineq', 'fun': lambda x: x[32]},
        {'type': 'ineq', 'fun': lambda x: x[33]},
        {'type': 'ineq', 'fun': lambda x: x[34]},
        
        {'type': 'ineq', 'fun': lambda x: x[35]},
        {'type': 'ineq', 'fun': lambda x: x[36]},
        {'type': 'ineq', 'fun': lambda x: x[37]},
        {'type': 'ineq', 'fun': lambda x: x[38]},
        {'type': 'ineq', 'fun': lambda x: x[39]},
        {'type': 'ineq', 'fun': lambda x: x[40]},
        {'type': 'ineq', 'fun': lambda x: x[41]},
        {'type': 'ineq', 'fun': lambda x: x[42]},
        {'type': 'ineq', 'fun': lambda x: x[43]},
        {'type': 'ineq', 'fun': lambda x: x[44]},
        {'type': 'ineq', 'fun': lambda x: x[45]},
        {'type': 'ineq', 'fun': lambda x: x[46]},
        {'type': 'ineq', 'fun': lambda x: x[47]},
        {'type': 'ineq', 'fun': lambda x: x[48]},
        {'type': 'ineq', 'fun': lambda x: x[49]},
        {'type': 'ineq', 'fun': lambda x: x[50]},
        {'type': 'ineq', 'fun': lambda x: x[51]},
        {'type': 'ineq', 'fun': lambda x: x[52]},
        {'type': 'ineq', 'fun': lambda x: x[53]},
        {'type': 'ineq', 'fun': lambda x: x[54]},
        {'type': 'ineq', 'fun': lambda x: x[55]},
        {'type': 'ineq', 'fun': lambda x: x[56]},
        {'type': 'ineq', 'fun': lambda x: x[57]},
        {'type': 'ineq', 'fun': lambda x: x[58]},
        {'type': 'ineq', 'fun': lambda x: x[59]},
        {'type': 'ineq', 'fun': lambda x: x[60]},
        {'type': 'ineq', 'fun': lambda x: x[61]},
        {'type': 'ineq', 'fun': lambda x: x[62]},
        
        {'type': 'ineq', 'fun': lambda x: x[63]},
        {'type': 'ineq', 'fun': lambda x: x[64]},
        {'type': 'ineq', 'fun': lambda x: x[65]},
        {'type': 'ineq', 'fun': lambda x: x[66]},
        {'type': 'ineq', 'fun': lambda x: x[67]},
        {'type': 'ineq', 'fun': lambda x: x[68]},
        {'type': 'ineq', 'fun': lambda x: x[69]},
        {'type': 'ineq', 'fun': lambda x: x[70]},
        {'type': 'ineq', 'fun': lambda x: x[71]},
        {'type': 'ineq', 'fun': lambda x: x[72]},
        {'type': 'ineq', 'fun': lambda x: x[73]},
        {'type': 'ineq', 'fun': lambda x: x[74]},
        {'type': 'ineq', 'fun': lambda x: x[75]},
        {'type': 'ineq', 'fun': lambda x: x[76]},
        {'type': 'ineq', 'fun': lambda x: x[77]},
        {'type': 'ineq', 'fun': lambda x: x[78]},
        {'type': 'ineq', 'fun': lambda x: x[79]},
        {'type': 'ineq', 'fun': lambda x: x[80]},
        {'type': 'ineq', 'fun': lambda x: x[81]},
        {'type': 'ineq', 'fun': lambda x: x[82]},
        {'type': 'ineq', 'fun': lambda x: x[83]},
        {'type': 'ineq', 'fun': lambda x: x[84]},
        {'type': 'ineq', 'fun': lambda x: x[85]},
        {'type': 'ineq', 'fun': lambda x: x[86]},
        {'type': 'ineq', 'fun': lambda x: x[87]},
        {'type': 'ineq', 'fun': lambda x: x[88]},
        {'type': 'ineq', 'fun': lambda x: x[89]},
        {'type': 'ineq', 'fun': lambda x: x[90]},
        
        {'type': 'ineq', 'fun': lambda x: x[91]},
        {'type': 'ineq', 'fun': lambda x: x[92]},
        {'type': 'ineq', 'fun': lambda x: x[93]},
        {'type': 'ineq', 'fun': lambda x: x[94]},
        {'type': 'ineq', 'fun': lambda x: x[95]},
        {'type': 'ineq', 'fun': lambda x: x[96]},
        {'type': 'ineq', 'fun': lambda x: x[97]},
        {'type': 'ineq', 'fun': lambda x: x[98]},
        {'type': 'ineq', 'fun': lambda x: x[99]},
        {'type': 'ineq', 'fun': lambda x: x[100]},
        {'type': 'ineq', 'fun': lambda x: x[101]},
        {'type': 'ineq', 'fun': lambda x: x[102]},
        {'type': 'ineq', 'fun': lambda x: x[103]},
        {'type': 'ineq', 'fun': lambda x: x[104]},
        {'type': 'ineq', 'fun': lambda x: x[105]},
        {'type': 'ineq', 'fun': lambda x: x[106]},
        {'type': 'ineq', 'fun': lambda x: x[107]},
        {'type': 'ineq', 'fun': lambda x: x[108]},
        {'type': 'ineq', 'fun': lambda x: x[109]},
        {'type': 'ineq', 'fun': lambda x: x[110]},
        {'type': 'ineq', 'fun': lambda x: x[111]},
        {'type': 'ineq', 'fun': lambda x: x[112]},
        {'type': 'ineq', 'fun': lambda x: x[113]},
        {'type': 'ineq', 'fun': lambda x: x[114]},
        {'type': 'ineq', 'fun': lambda x: x[115]},
        {'type': 'ineq', 'fun': lambda x: x[116]},
        {'type': 'ineq', 'fun': lambda x: x[117]},
        {'type': 'ineq', 'fun': lambda x: x[118]},
        
        {'type': 'ineq', 'fun': lambda x: x[119]},
        {'type': 'ineq', 'fun': lambda x: x[120]},
        {'type': 'ineq', 'fun': lambda x: x[121]},
        {'type': 'ineq', 'fun': lambda x: x[122]},
        {'type': 'ineq', 'fun': lambda x: x[123]},
        {'type': 'ineq', 'fun': lambda x: x[124]},
        {'type': 'ineq', 'fun': lambda x: x[125]},
        {'type': 'ineq', 'fun': lambda x: x[126]},
        {'type': 'ineq', 'fun': lambda x: x[127]},
        {'type': 'ineq', 'fun': lambda x: x[128]},
        {'type': 'ineq', 'fun': lambda x: x[129]},
        {'type': 'ineq', 'fun': lambda x: x[130]},
        {'type': 'ineq', 'fun': lambda x: x[131]},
        {'type': 'ineq', 'fun': lambda x: x[132]},
        {'type': 'ineq', 'fun': lambda x: x[133]},
        {'type': 'ineq', 'fun': lambda x: x[134]},
        {'type': 'ineq', 'fun': lambda x: x[135]},
        {'type': 'ineq', 'fun': lambda x: x[136]},
        {'type': 'ineq', 'fun': lambda x: x[137]},
        {'type': 'ineq', 'fun': lambda x: x[138]},
        {'type': 'ineq', 'fun': lambda x: x[139]},
        {'type': 'ineq', 'fun': lambda x: x[140]},
        {'type': 'ineq', 'fun': lambda x: x[141]},
        {'type': 'ineq', 'fun': lambda x: x[142]},
        {'type': 'ineq', 'fun': lambda x: x[143]},
        {'type': 'ineq', 'fun': lambda x: x[144]},
        {'type': 'ineq', 'fun': lambda x: x[145]},
        {'type': 'ineq', 'fun': lambda x: x[146]},
        
        
        {'type': 'ineq', 'fun': lambda x: x[147]},
        {'type': 'ineq', 'fun': lambda x: x[148]},
        {'type': 'ineq', 'fun': lambda x: x[149]},
        {'type': 'ineq', 'fun': lambda x: x[150]},
        {'type': 'ineq', 'fun': lambda x: x[151]},
        {'type': 'ineq', 'fun': lambda x: x[152]},
        {'type': 'ineq', 'fun': lambda x: x[153]},
        {'type': 'ineq', 'fun': lambda x: x[154]},
        {'type': 'ineq', 'fun': lambda x: x[155]},
        {'type': 'ineq', 'fun': lambda x: x[156]},
        {'type': 'ineq', 'fun': lambda x: x[157]},
        {'type': 'ineq', 'fun': lambda x: x[158]},
        {'type': 'ineq', 'fun': lambda x: x[159]},
        {'type': 'ineq', 'fun': lambda x: x[160]},
        {'type': 'ineq', 'fun': lambda x: x[161]},
        {'type': 'ineq', 'fun': lambda x: x[162]},
        {'type': 'ineq', 'fun': lambda x: x[163]},
        {'type': 'ineq', 'fun': lambda x: x[164]},
        {'type': 'ineq', 'fun': lambda x: x[165]},
        {'type': 'ineq', 'fun': lambda x: x[166]},
        {'type': 'ineq', 'fun': lambda x: x[167]},
        {'type': 'ineq', 'fun': lambda x: x[168]},
        {'type': 'ineq', 'fun': lambda x: x[169]},
        {'type': 'ineq', 'fun': lambda x: x[170]},
        {'type': 'ineq', 'fun': lambda x: x[171]},
        {'type': 'ineq', 'fun': lambda x: x[172]},
        {'type': 'ineq', 'fun': lambda x: x[173]},
        {'type': 'ineq', 'fun': lambda x: x[174]},
        
        {'type': 'ineq', 'fun': lambda x: x[175]},
        {'type': 'ineq', 'fun': lambda x: x[176]},
        {'type': 'ineq', 'fun': lambda x: x[177]},
        {'type': 'ineq', 'fun': lambda x: x[178]},
        {'type': 'ineq', 'fun': lambda x: x[179]},
        {'type': 'ineq', 'fun': lambda x: x[180]},
        {'type': 'ineq', 'fun': lambda x: x[181]},
        {'type': 'ineq', 'fun': lambda x: x[182]},
        {'type': 'ineq', 'fun': lambda x: x[183]},
        {'type': 'ineq', 'fun': lambda x: x[184]},
        {'type': 'ineq', 'fun': lambda x: x[185]},
        {'type': 'ineq', 'fun': lambda x: x[186]},
        {'type': 'ineq', 'fun': lambda x: x[187]},
        {'type': 'ineq', 'fun': lambda x: x[188]},
        {'type': 'ineq', 'fun': lambda x: x[189]},
        {'type': 'ineq', 'fun': lambda x: x[190]},
        {'type': 'ineq', 'fun': lambda x: x[191]},
        {'type': 'ineq', 'fun': lambda x: x[192]},
        {'type': 'ineq', 'fun': lambda x: x[193]},
        {'type': 'ineq', 'fun': lambda x: x[194]},
        {'type': 'ineq', 'fun': lambda x: x[195]},
        {'type': 'ineq', 'fun': lambda x: x[196]},
        {'type': 'ineq', 'fun': lambda x: x[197]},
        {'type': 'ineq', 'fun': lambda x: x[198]},
        {'type': 'ineq', 'fun': lambda x: x[199]},
        {'type': 'ineq', 'fun': lambda x: x[200]},
        {'type': 'ineq', 'fun': lambda x: x[201]},
        {'type': 'ineq', 'fun': lambda x: x[202]}
        
       )
 
x0 = np.ones(203) # 设置初始值(随机设置即可)
 
res = minimize(fun, x0, method='SLSQP', constraints=cons) # 调用最小值模块
res
```



### Google ORTools求解

```Python
# 导入线性求解器，线性和整数求解的API都在其中
from ortools.linear_solver import pywraplp
import math

# 声明求解器
solver = pywraplp.Solver.CreateSolver('GLOP')

# 创建变量 
x3 = solver.NumVar(0, 1000, 'x3') 
x4 = solver.NumVar(0, 1600, 'x4')
x5 = solver.NumVar(0, math.inf, 'x5') 
x6 = solver.NumVar(0, math.inf, 'x6')
x7 = solver.NumVar(0, math.inf, 'x7') 
x8 = solver.NumVar(0, math.inf, 'x8')
x9 = solver.NumVar(0, math.inf, 'x9') 
x10 = solver.NumVar(0, math.inf, 'x10')
print('变量个数 =', solver.NumVariables())

# 添加约束
solver.Add(x3 + x4 == 780)
solver.Add(x3 == x5+x6+x7)
solver.Add(x4 == x8+x9+x10)
print('约束个数 =', solver.NumConstraints())

# 定义目标函数
solver.Minimize(155*x3 + 160*x4 + 73.3*x5 + 101.1*(x6+x7) + 93.1*(x8+x9) + 78.9*x10)

# 调用求解器
status = solver.Solve()

if status == pywraplp.Solver.OPTIMAL:
    print('最优值 =', solver.Objective().Value())
    print('x3', x3.solution_value())
    print('x4', x4.solution_value())
    print('x5', x5.solution_value())
    print('x6', x6.solution_value())
    print('x7', x7.solution_value())
    print('x8', x8.solution_value())
    print('x9', x9.solution_value())
    print('x10', x10.solution_value())
else:
    print('The problem does not have an optimal solution.')
```

