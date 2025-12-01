import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC

# 拿数据
iris = load_iris()
# 为了演示二分类，只取前100个数据(类别0和1)
# 选前3个特征，正好对应X, Y, Z轴
X = iris.data[:100, :3]
y = iris.target[:100]

# 必须用线性核(linear)，不然决策边界不是平面的
clf = SVC(kernel='linear')
clf.fit(X, y)

# 搞个3D画布
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 分别把两类点画上去，蓝色是0，红色是1
ax.scatter(X[y==0, 0], X[y==0, 1], X[y==0, 2], c='b', label='Class 0')
ax.scatter(X[y==1, 0], X[y==1, 1], X[y==1, 2], c='r', label='Class 1')

# 核心公式：w1*x + w2*y + w3*z + b = 0
# 反推 z = -(w1*x + w2*y + b) / w3
# 这样才能画出那个分割平面
z = lambda x, y: (-clf.intercept_[0] - clf.coef_[0][0] * x - clf.coef_[0][1] * y) / clf.coef_[0][2]

# 生成网格坐标
tmp_x = np.linspace(X[:, 0].min(), X[:, 0].max(), 30)
tmp_y = np.linspace(X[:, 1].min(), X[:, 1].max(), 30)
x_surf, y_surf = np.meshgrid(tmp_x, tmp_y)

# 画平面，透明度调低点方便看后面的点
ax.plot_surface(x_surf, y_surf, z(x_surf, y_surf), alpha=0.3, color='grey')

ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
ax.set_zlabel('Petal Length')
ax.set_title('Task 2: 3D Decision Boundary')
plt.show()