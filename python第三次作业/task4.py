import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from matplotlib import cm

# 1. 准备数据
iris = load_iris()
# 还是只取前100个数据(两类)和前3个特征
X = iris.data[:100, :3]
y = iris.target[:100]

# 2. 训练模型
# 重点：要开 probability=True 才能算概率
# 重点：kernel='linear' 还是画平面
clf = SVC(kernel='linear', probability=True)
clf.fit(X, y)

# 3. 准备3D画布
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 4. 画原始数据点
ax.scatter(X[y==0, 0], X[y==0, 1], X[y==0, 2], c='b', s=30, label='Class 0 (Setosa)')
ax.scatter(X[y==1, 0], X[y==1, 1], X[y==1, 2], c='r', s=30, label='Class 1 (Versicolor)')

# 5. 加分点：圈出"支持向量" (Support Vectors)
# 这些点是真正撑起分类边界的关键数据
sv = clf.support_vectors_
ax.scatter(sv[:, 0], sv[:, 1], sv[:, 2], c='k', marker='x', s=100, linewidths=1.5, label='Support Vectors')

# 6. 计算平面
# 还是那个公式：z = -(w1x + w2y + b) / w3
z = lambda x, y: (-clf.intercept_[0] - clf.coef_[0][0] * x - clf.coef_[0][1] * y) / clf.coef_[0][2]

# 生成网格
tmp_x = np.linspace(X[:, 0].min(), X[:, 0].max(), 50)
tmp_y = np.linspace(X[:, 1].min(), X[:, 1].max(), 50)
xx, yy = np.meshgrid(tmp_x, tmp_y)
zz = z(xx, yy)

# 7. 再次加分点：计算平面上每个点的概率，用来上色
# 我们把网格展平，喂给模型算概率
grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
probs = clf.predict_proba(grid_points)[:, 1] # 取类别1的概率
probs = probs.reshape(xx.shape)

# 8. 画出"彩色"平面
# facecolors=cm.coolwarm(probs) 让平面颜色随概率变化
surf = ax.plot_surface(xx, yy, zz, facecolors=cm.coolwarm(probs), alpha=0.6, shade=False)

# 设置坐标轴
ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
ax.set_zlabel('Petal Length')
ax.set_title('Task 4 Bonus: Decision Plane with Probability Heatmap')

# 加上图例
ax.legend()

# 调整视角让平面看清楚点
ax.view_init(elev=20, azim=30)

plt.show()