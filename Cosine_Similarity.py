import numpy as np
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt

# 定义集合 D
D = [
    np.array([1, 0, 0, 0]),  # e_1
    np.array([0, 1, 0, 0]),  # e_2
    np.array([1, 1, 1, 0]),  # e_1 + e_2 + e_3
    np.array([0, 0, 0, 1]),  # e_4
    np.array([-1, -1, 0, 0]),# -e_1 - e_2
    np.array([0, 0, -1, -1]) # -e_3 - e_4
]
N = 4
# 计算余弦相似度
def cosine_similarity(u, d):
    return np.dot(u, d) / np.linalg.norm(d)

# 计算余弦度量 cm(D)
def cosine_measure(D, u):
    # best_cosine_measure = float('inf')

    max_cosine_similarity = max(cosine_similarity(u, d) for d in D)

        # best_cosine_measure = min(best_cosine_measure, max_cosine_similarity)

    return max_cosine_similarity

arr = []

min_u = None
min_position = None
min_value = 100
for i in range(20):
    u = np.random.uniform(-np.pi, np.pi, N)
    u = u / np.linalg.norm(u)
    cosine_val = cosine_measure(D, u)

    if cosine_val < min_value:
        min_value = cosine_val
        min_u = u

ut = min_u
# 生成单位向量 u
for i in range(1000):  # 这里可以根据需要调整测试的单位向量数量
    gradient = 0
    for j in range(10):
        u = np.random.randn(N)  # 生成一个随机向量
        u = u / np.linalg.norm(u)  # 将其规范化为单位向量
        g1 = cosine_measure(D, ut+0.1*u)
        g2 = cosine_measure(D, ut-0.1*u)
        gradient += ((g1-g2)/0.2)*u
    gradient = gradient/10

    ut = ut - 0.01 * gradient
    ut = ut / np.linalg.norm(ut)
    min = 100
    val = cosine_measure(D, ut)
    arr.append(val)
    if val < min:
        min = val
        min_u = ut

# cm_value = cosine_measure(D)
print(f"Cosine Measure: {min}")
#print(arr)

fig, ax = plt.subplots()

# 绘制loss值
ax.plot(arr, label='Training Loss')

# 添加标题和标签
ax.set_title('Loss During Training')
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')

# 显示图例
ax.legend()

# 显示网格
ax.grid(True)

# 显示图形
plt.show()