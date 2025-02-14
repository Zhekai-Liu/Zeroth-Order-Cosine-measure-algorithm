import numpy as np
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt

# define D
# D = [
#     np.array([1, 0, 0, 0]),  # e_1
#     np.array([0, 1, 0, 0]),  # e_2
#     np.array([1, 1, 1, 0]),  # e_1 + e_2 + e_3
#     np.array([0, 0, 0, 1]),  # e_4
#     np.array([-1, -1, 0, 0]),# -e_1 - e_2
#     np.array([0, 0, -1, -1]) # -e_3 - e_4
# ]
# D = [
#     np.array([1, 0]),  # e_1
#     np.array([0, 1]),  # e_2
#     np.array([-1, -1])
# ]
D = [
    np.array([1, 0, 0, 0, 0, 0, 0, 0]),
    np.array([0, 1, 0, 0, 0, 0, 0, 0]),
    np.array([0, 0, 1, 0, 0, 0, 0 ,0]),
    np.array([0, 0, 0, 1, 0, 0 ,0 ,0]),
    np.array([0, 0, 0, 0, 1, 0 ,0 ,0]),
    np.array([0, 0, 0, 0, 0, 1 ,0 ,0]),
    np.array([0, 0, 0, 0, 0, 0 ,1 ,0]),
    np.array([0, 0, 0, 0, 0, 0 ,0 ,1])
]
N = 8

def cosine_similarity(u, d):
    return np.dot(u, d) / np.linalg.norm(u) * np.linalg.norm(d)


def cosine_measure(D, u):
    # best_cosine_measure = float('inf')

    max_cosine_similarity = max(cosine_similarity(np.cos(u), d) for d in D)

        # best_cosine_measure = min(best_cosine_measure, max_cosine_similarity)

    return max_cosine_similarity

arr = []

min_u = None
min_position = None
min_value = 100
for i in range(1000):
    u = np.random.uniform(-np.pi, np.pi, N)
    cosine_val = cosine_measure(D, u)

    if cosine_val < min_value:
        min_value = cosine_val
        min_u = u

ut = min_u

for i in range(1000):  
    gradient = 0
    for j in range(10):
        u = np.random.randn(N)
        u = u / np.linalg.norm(u)
        # make the spheric position illegal 
        alpha = ut + 0.1 * u  # 先计算原始值
        alpha = np.where(alpha < -np.pi, alpha + 2 * np.pi, alpha)  # smaller than -π add 2π
        alpha = np.where(alpha > np.pi, alpha - 2 * np.pi, alpha)  #  bigger than π minus 2π

        # make the spheric position illegal 
        beta = ut - 0.1 * u  
        beta = np.where(beta < -np.pi, beta + 2 * np.pi, beta)  #  smaller than -π add 2π
        beta = np.where(beta > np.pi, beta - 2 * np.pi, beta)  # bigger than π minus 2π

        g1 = cosine_measure(D, alpha)
        g2 = cosine_measure(D, beta)
        gradient += ((g1-g2)/0.2)*u
    gradient = gradient/10

    ut = ut - 0.1 * gradient
    ut = np.where(ut < -np.pi, ut + 2 * np.pi, ut)  # smaller than -π add 2π
    ut = np.where(ut > np.pi, ut - 2 * np.pi, ut)  # bigger than π minus 2π

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


ax.plot(arr, label='Training Loss')

ax.set_title('Loss During Training')
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')

ax.legend()


ax.grid(True)

plt.show()
