import numpy as np
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt

# Positive spanning D
D = [
    np.array([1, 0, 0, 0]),  # e_1
    np.array([0, 1, 0, 0]),  # e_2
    np.array([1, 1, 1, 0]),  # e_1 + e_2 + e_3
    np.array([0, 0, 0, 1]),  # e_4
    np.array([-1, -1, 0, 0]),# -e_1 - e_2
    np.array([0, 0, -1, -1]) # -e_3 - e_4
]
N = 4

def cosine_similarity(u, d):
    return np.dot(u, d) / np.linalg.norm(d)

# cosine measure cm(D)
def cosine_measure(D, u):
    # best_cosine_measure = float('inf')

    max_cosine_similarity = max(cosine_similarity(u, d) for d in D)

        # best_cosine_measure = min(best_cosine_measure, max_cosine_similarity)

    return max_cosine_similarity

# iteration information
arr = []

min_u = None
min_position = None
min_value = 100

# Initialization
for i in range(20):
    u = np.random.uniform(-np.pi, np.pi, N)
    u = u / np.linalg.norm(u)
    cosine_val = cosine_measure(D, u)

    if cosine_val < min_value:
        min_value = cosine_val
        min_u = u
# end initialization
ut = min_u

# Optimization
for i in range(1000):  # Max iterations
    gradient = 0
    for j in range(10):   # reduce the variance
        u = np.random.randn(N)  # Guassion noise \mu
        u = u / np.linalg.norm(u)  # normalize it
        g1 = cosine_measure(D, ut+0.1*u)
        g2 = cosine_measure(D, ut-0.1*u)
        gradient += ((g1-g2)/0.2)*u  # Centered finite difference gradient
        
    gradient = gradient/10

    ut = ut - 0.01 * gradient  # update the position of the vector
    ut = ut / np.linalg.norm(ut)  # make sure it still on the unit sphere
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

#  DRAW FIGURE
ax.plot(arr, label='Training Loss')


ax.set_title('Loss During Training')
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')


ax.legend()

ax.grid(True)

plt.show()
