import matplotlib.pyplot as plt
import numpy as np

def plot_tsp_solution(ax, cities, tour, title="TSP Solution", bridge=None):
    ax.clear()
    x, y = zip(*[cities[i] for i in tour] + [cities[tour[0]]])
    ax.plot(x, y, 'bo-', markersize=8, label="Cities & Path")
    ax.plot(x[0], y[0], 'ro', markersize=10, label="Start")

    # Only draw bridge if it's actually used
    if bridge:
        a, b = bridge
        for i in range(len(tour)):
            x1, x2 = tour[i], tour[(i + 1) % len(tour)]
            if (x1 == a and x2 == b) or (x1 == b and x2 == a):
                ax.plot(
                    [cities[a][0], cities[b][0]],
                    [cities[a][1], cities[b][1]],
                    'r--', linewidth=3, label="Mandatory Bridge"
                )
                break  # stop after drawing

    ax.set_title(title)
    ax.legend()



def apply_mandatory_bridge(cost_matrix, city_a, city_b, bridge_cost=1e-5):
    cost_matrix = cost_matrix.copy()

    # Step 1: Apply very low cost for the bridge
    cost_matrix[city_a, city_b] = bridge_cost
    cost_matrix[city_b, city_a] = bridge_cost

    # Step 2 (Optional): Slightly penalize other connections *if they are valid*
    penalty = np.nanmean(cost_matrix[np.isfinite(cost_matrix)]) * 0.5

    for i in range(len(cost_matrix)):
        if i != city_b and np.isfinite(cost_matrix[city_a, i]):
            cost_matrix[city_a, i] += penalty
            cost_matrix[i, city_a] += penalty
        if i != city_a and np.isfinite(cost_matrix[city_b, i]):
            cost_matrix[city_b, i] += penalty
            cost_matrix[i, city_b] += penalty

    # Step 3: Reset bridge cost just in case penalty was applied to it
    cost_matrix[city_a, city_b] = bridge_cost
    cost_matrix[city_b, city_a] = bridge_cost

    return cost_matrix


def create_cost_matrix(cities):
    num = len(cities)
    matrix = np.zeros((num, num))
    for i in range(num):
        for j in range(num):
            if i == j:
                matrix[i][j] = np.inf
            else:
                matrix[i][j] = np.linalg.norm(np.array(cities[i]) - np.array(cities[j]))
    return matrix


