import matplotlib.pyplot as plt
import numpy as np

def plot_tsp_solution(ax, cities, best_solution, title="TSP Solution", bridge=None):
    ax.clear()  # Clear previous plot

    # Extract x, y coordinates for the cities in tour order
    x, y = zip(*[cities[i] for i in best_solution] + [cities[best_solution[0]]])  # Close the loop
    ax.plot(x, y, 'bo-', markersize=8, label="Cities & Path")
    ax.plot(x[0], y[0], 'ro', markersize=10, label="Start")
    ax.set_title(title)

    # Highlight bridge if provided
    if bridge:
        city_a, city_b = bridge
        bx = [cities[city_a][0], cities[city_b][0]]
        by = [cities[city_a][1], cities[city_b][1]]
        ax.plot(bx, by, 'r--', linewidth=3, label="Mandatory Bridge")

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


