import numpy as np
import random

def ant_colony_optimization(cost_matrix, alpha, beta, InitialPheremone, evap_rate, m, constant, I_max):
    num_nodes = len(cost_matrix)
    pheromones = np.full((num_nodes, num_nodes), InitialPheremone)
    heuristics = 1 / (cost_matrix + 1e-10)

    best_solution = None
    best_cost = float('inf')

    for _ in range(I_max):
        solutions, costs = [], []
        for _ in range(m):
            visited = [random.randint(0, num_nodes - 1)]
            while len(visited) < num_nodes:
                probabilities = [(j, (pheromones[visited[-1]][j] ** alpha) * (heuristics[visited[-1]][j] ** beta))
                                 for j in range(num_nodes) if j not in visited]
                total = sum(prob for _, prob in probabilities)
                probabilities = [(node, prob / total) for node, prob in probabilities]
                next_node = random.choices([node for node, _ in probabilities], [prob for _, prob in probabilities])[0]
                visited.append(next_node)

            visited.append(visited[0])
            solutions.append(visited)
            costs.append(sum(cost_matrix[visited[i]][visited[i + 1]] for i in range(len(visited) - 1)))

        min_cost_index = np.argmin(costs)
        if costs[min_cost_index] < best_cost:
            best_solution, best_cost = solutions[min_cost_index], costs[min_cost_index]

    return best_solution, best_cost  # No plots, just return data
