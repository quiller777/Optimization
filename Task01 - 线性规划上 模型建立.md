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


