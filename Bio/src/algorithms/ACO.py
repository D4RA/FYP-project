import numpy as np
import random
from PyQt5.QtWidgets import QApplication

from Bio.src.plotting.utils import plot_tsp_solution


# Assuming you're using this utility

def cost_matrix_to_coords(cost_matrix):
    """Reverse-engineer approximate coordinates for plotting from a cost matrix."""
    num_cities = cost_matrix.shape[0]
    angle_step = 2 * np.pi / num_cities
    return [(np.cos(i * angle_step) * 50 + 50, np.sin(i * angle_step) * 50 + 50) for i in range(num_cities)]


def ant_colony_optimization(
    cost_matrix, alpha, beta, initial_pheromone, evap_rate, m, constant, I_max,
    ax=None, canvas=None
):
    num_nodes = len(cost_matrix)

    # Initialize pheromone and heuristic matrices
    pheromones = np.full((num_nodes, num_nodes), initial_pheromone)
    heuristics = 1 / (cost_matrix + 1e-10)

    best_solution = None
    best_cost = float('inf')

    for iteration in range(I_max):
        solutions = []
        costs = []

        # Build solutions for each ant
        for _ in range(m):
            visited = [random.randint(0, num_nodes - 1)]
            while len(visited) < num_nodes:
                probabilities = [
                    (j, (pheromones[visited[-1]][j] ** alpha) * (heuristics[visited[-1]][j] ** beta))
                    for j in range(num_nodes) if j not in visited
                ]
                total = sum(prob for _, prob in probabilities)
                probabilities = [(node, prob / total) for node, prob in probabilities]
                next_node = random.choices(
                    [node for node, _ in probabilities],
                    [prob for _, prob in probabilities]
                )[0]
                visited.append(next_node)

            visited.append(visited[0])  # Close the loop
            cost = sum(cost_matrix[visited[i]][visited[i + 1]] for i in range(len(visited) - 1))

            solutions.append(visited)
            costs.append(cost)

        # Get best solution of current iteration
        min_cost_index = np.argmin(costs)
        iteration_best_solution = solutions[min_cost_index]
        iteration_best_cost = costs[min_cost_index]

        # Update global best
        if iteration_best_cost < best_cost:
            best_solution = iteration_best_solution
            best_cost = iteration_best_cost

        # === Update Pheromones ===
        # Evaporation
        pheromones *= (1 - evap_rate)

        # Deposit pheromone from the best ant of this iteration
        for i in range(len(iteration_best_solution) - 1):
            a = iteration_best_solution[i]
            b = iteration_best_solution[i + 1]
            pheromones[a][b] += constant / iteration_best_cost
            pheromones[b][a] += constant / iteration_best_cost  # Assuming symmetric TSP

        # Optional: Close the loop deposit
        a = iteration_best_solution[-1]
        b = iteration_best_solution[0]
        pheromones[a][b] += constant / iteration_best_cost
        pheromones[b][a] += constant / iteration_best_cost

        # === Live Plotting ===
        if ax and canvas:
            ax.clear()
            plot_tsp_solution(ax, cost_matrix_to_coords(cost_matrix), iteration_best_solution,
                              f"ACO Progress - Iteration {iteration + 1}")
            canvas.draw()
            QApplication.processEvents()

    return best_solution, best_cost
