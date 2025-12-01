# Project 3: 鸢尾花数据分类与可视化 (Iris Data Classification and Visualization)

本实验利用 Python 机器学习库（Scikit-learn）对鸢尾花数据集进行分类分析，并通过 Matplotlib 和 Seaborn 实现从二维到三维的数据可视化，直观展示分类器的决策边界与概率分布。

## 🛠️ 环境配置 (Prerequisites)

本项目基于 Python 3 开发。为了确保代码能够顺利运行，请安装以下必要的依赖库。

### 1. 依赖库列表
* **numpy**: 数值计算
* **pandas**: 数据处理
* **matplotlib**: 基础绘图 (2D/3D)
* **seaborn**: 统计绘图 (箱线图等)
* **scikit-learn**: 机器学习算法 (SVM, 逻辑回归等)
* **plotly**: 交互式绘图 (用于数据预览)

### 2. 安装命令
你可以使用 `pip` 一键安装所有依赖：

```bash
pip install numpy pandas matplotlib seaborn scikit-learn plotly





📂 文件说明 (File Structure)
本项目包含以下 5 个 Python 脚本，分别对应实验报告中的不同任务环节：

data_preview.py (对应 Data Exploration) 功能：数据预处理、绘制特征分布箱线图及交互式散点矩阵图。

classifier2d.py (对应 Task 1) 功能：实现逻辑回归多分类，绘制 2D 决策边界 及各类别概率热力图。

task2.py (对应 Task 2) 功能：基于线性 SVM，在三维空间中绘制 决策超平面 (Decision Hyperplane)。

task3.py (对应 Task 3) 功能：基于逻辑回归，在三维空间中绘制 概率曲面 (Probability Map)。

task4.py (对应 Task 4 Bonus) 功能：进阶可视化：融合了 3D 决策平面与概率热力图，并标记了支持向量。





🚀 运行指南 (How to Run)
请按照以下顺序运行脚本以复现实验报告中的结果。

步骤 1: 数据探索
运行 data_preview.py 查看数据分布情况。

Bash

python data_preview.py
输出: 将弹出一个包含 4 个子图的箱线图窗口；并在浏览器中打开 Plotly 生成的交互式散点图。

步骤 2: 二维分类可视化 (Task 1)
运行 classifier2d.py 查看二维平面下的分类效果。

Bash

python classifier2d.py
输出: 弹出一个窗口，包含 1 张整体决策边界图和 3 张分列别的概率热力图。

步骤 3: 三维决策平面 (Task 2)
运行 task2.py 查看 SVM 在三维空间中的线性分割平面。

Bash

python task2.py
输出: 弹出 3D 绘图窗口，展示灰色的分割平面。提示：使用鼠标拖拽可旋转视角。

步骤 4: 三维概率曲面 (Task 3)
运行 task3.py 查看概率随特征变化的趋势。

Bash

python task3.py
输出: 弹出 3D 绘图窗口，展示呈 S 型（Sigmoid）趋势的彩色概率曲面。

步骤 5: 进阶融合可视化 (Task 4 Bonus)
运行 task4.py 查看最终的综合可视化结果。

Bash

python task4.py
输出: 弹出 3D 绘图窗口，展示带有概率颜色映射的决策平面，并用黑色 "x" 标记出了支持向量 (Support Vectors)。





📝 注意事项 (Notes)
交互操作: 所有 3D 图表 (task2.py, task3.py, task4.py) 均支持鼠标左键拖拽旋转，右键拖拽缩放，建议旋转至最佳视角进行观察。

Plotly 图表: data_preview.py 中生成的 Plotly 图表通常会自动调用系统默认浏览器打开。

警告信息: 运行时若出现关于 FutureWarning 或 MKL 的警告，通常不影响代码正常运行和结果输出，可忽略。