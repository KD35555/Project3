import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from matplotlib import cm

iris = load_iris()
# 这里只要前两个特征当底面坐标(x, y)
X = iris.data[:100, :2]
y = iris.target[:100]

# 跑逻辑回归
model = LogisticRegression()
model.fit(X, y)

# 设定画图范围，稍微留点边距
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
# 打网格，步长0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# 算出网格上每个点的概率
# [:, 1] 表示取属于类别1的概率值
Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)  # 变回网格形状

# 画3D曲面
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
# cmap选个冷暖色对比，coolwarm比较好看
surf = ax.plot_surface(xx, yy, Z, cmap=cm.coolwarm, alpha=0.8)

ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
ax.set_zlabel('Probability of Class 1') # Z轴高度就是概率
ax.set_title('Task 3: 3D Probability Map')

# 加个色条说明数值
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()