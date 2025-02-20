import numpy as np
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt

# D = [
#      np.array([1, 0]),  # e_1
#      np.array([0, 1])  # e_2
# ]
# D = [
#     np.array([1, 0, 0, 0]),  # e_1
#     np.array([0, 1, 0, 0]),  # e_2
#     np.array([1, 1, 1, 0]),  # e_1 + e_2 + e_3
#     np.array([0, 0, 0, 1]),  # e_4
#     np.array([-1, -1, 0, 0]),# -e_1 - e_2
#     np.array([0, 0, -1, -1]) # -e_3 - e_4
# ]

# D = [
#     np.array([1, 0, 0, 0, 0, 0, 0, 0]),
#     np.array([0, 1, 0, 0, 0, 0, 0, 0]),
#     np.array([0, 0, 1, 0, 0, 0, 0 ,0]),
#     np.array([0, 0, 0, 1, 0, 0 ,0 ,0]),
#     np.array([0, 0, 0, 0, 1, 0 ,0 ,0]),
#     np.array([0, 0, 0, 0, 0, 1 ,0 ,0]),
#     np.array([0, 0, 0, 0, 0, 0 ,1 ,0]),
#     np.array([0, 0, 0, 0, 0, 0 ,0 ,1])
# ]
N = 225

D = [
    np.array([1 if j == i else 0 for j in range(N)])  # e_i
    for i in range(N)
]


def centered_simplex_gradient(Y_plus, f, D):

    y0 = Y_plus[0]
    #n = Y_plus.shape[1]  # dimensions n
    L = (Y_plus[1:] - y0).T  # construct L，(n, n) dimensions

    # calculate delta
    delta = []
    for yi in Y_plus[1:]:
        yd = 2 * y0 - yi  # calculate y^0 - d^i = 2y0 - y^i
        delta_i = (f(D, yi) - f(D, yd)) / 2
        delta.append(delta_i)
    delta = np.array(delta)

    # solve linear system of function L^T @ beta = delta
    beta = np.linalg.solve(L.T, delta)

    return beta

def cosine_similarity(u, d):
    return np.dot(u, d) / (np.linalg.norm(u)*np.linalg.norm(d))

#  cm(D)
def cosine_measure(D, u):
    # best_cosine_measure = float('inf')

    max_cosine_similarity = max(cosine_similarity(u, d) for d in D)

        # best_cosine_measure = min(best_cosine_measure, max_cosine_similarity)

    return max_cosine_similarity

# cm(D) for spheric vision
def cosine_measuress(D, u):
    # best_cosine_measure = float('inf')

    max_cosine_similarity = max(cosine_similarity(np.cos(u)/np.linalg.norm(np.cos(u)), d) for d in D)

        # best_cosine_measure = min(best_cosine_measure, max_cosine_similarity)

    return max_cosine_similarity

arr1 = []
arr2 = []
arrs = []

min_u = None
min_position = None
min_value = 100
for i in range(30):
    u = np.random.uniform(-np.pi, np.pi, N)
    u = u / np.linalg.norm(u)
    cosine_val = cosine_measure(D, u)

    if cosine_val < min_value:
        min_value = cosine_val
        min_u = u

ut = min_u
current_point = min_u
uts = np.arccos(min_u)
uts = np.where(uts < -np.pi, uts + 2 * np.pi, uts)  # smaller than -π plus 2π
uts = np.where(uts > np.pi, uts - 2 * np.pi, uts)  # bigger than π minus 2π

min = 100
min2 = 100
mins = 100

learning_rate1 = 0.01
learning_rate2 = 0.01
for i in range(1000):  # max iteration
    gradient = 0
    gradient_sphere = 0
    Y_plus = [current_point]
    for j in range(N):
        u = np.random.randn(N)
        u = u / np.linalg.norm(u) #### Random vector

        g1 = cosine_measure(D, ut+0.1*u)
        g2 = cosine_measure(D, ut-0.1*u)
        gradient += ((g1-g2)/0.2)*u

        Y_plus.append(current_point+0.1*u)
# ---------------------------------------------------------------------
        alpha = uts + 0.1 * u  # make the spheric position illegal
        alpha = np.where(alpha < -np.pi, alpha + 2 * np.pi, alpha)  # smaller than -π plus 2π
        alpha = np.where(alpha > np.pi, alpha - 2 * np.pi, alpha)  # bigger than π minus 2π

        # beta = ut - 0.1*u make the spheric position illegal
        beta = uts - 0.1 * u
        beta = np.where(beta < -np.pi, beta + 2 * np.pi, beta)  # smaller than -π plus 2π
        beta = np.where(beta > np.pi, beta - 2 * np.pi, beta)  # bigger than π minus 2π

        g11 = cosine_measuress(D, alpha)
        g22 = cosine_measuress(D, beta)
        gradient_sphere += ((g11 - g22) / 0.2) * u


    gradient_sphere = gradient_sphere / N
    gradient_sphere = gradient_sphere / np.linalg.norm(gradient_sphere)
    uts = uts - learning_rate1 * gradient_sphere
    uts = np.where(uts < -np.pi, uts + 2 * np.pi, uts)  # smaller than -π plus 2π
    uts = np.where(uts > np.pi, uts - 2 * np.pi, uts)  # bigger than π minus 2π
    valllll = cosine_measuress(D, uts)
    arrs.append(valllll)
    if valllll < mins:
        mins = valllll

# ---------------------------------------------------------------------
    gradient = gradient/N
    gradient = gradient / np.linalg.norm(gradient)

    ut = ut - learning_rate1 * gradient
    ut = ut / np.linalg.norm(ut)

    val = cosine_measure(D, ut)
    arr1.append(val)
    if val < min:
        min = val

        #min_u = ut
#---------------------------------------------------------------------
    Y_plus = np.array(Y_plus)
    gradient_CS = centered_simplex_gradient(Y_plus, cosine_measure, D)
    gradient_CS = gradient_CS / np.linalg.norm(gradient_CS)
    current_point = current_point - learning_rate2 * gradient_CS
    current_point = current_point / np.linalg.norm(current_point)
    val2 = cosine_measure(D,current_point)
    arr2.append(val2)
    if val2 < min2:
        min2 = val2




# cm_value = cosine_measure(D)
print(f"Cosine Measure ZO-R: {min}")
print(f"Cosine Measure CS: {min2}")
print(f"Cosine Measure ZO-S: {mins}")
#print(arr)

fig, ax = plt.subplots()


ax.plot(arr1, label='Zoerth-Order Rectangular Loss 1')  # 绘制 arr1
ax.plot(arr2, label='Central Simplex Loss 2')  # 绘制 arr2
ax.plot(arrs, label='Zoerth-Order Spheric Loss 2')  # 绘制 arr2


ax.set_title('Loss During Training')
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')


ax.legend()


ax.grid(True)


plt.show()
