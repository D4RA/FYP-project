import random
import numpy as np

def generate_random_nodes(num_nodes):
    return [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(num_nodes)]

def initialise_pheremones(num_nodes, InitialPheremone):
    return np.full((num_nodes, num_nodes), InitialPheremone)

def calculate_probabilities(pheremones, heuristics, alpha, beta, current_node, visited):
    total = 0
    probabilities = []
    for j in range(len(pheremones)):
        if j not in visited:
            value = (pheremones[current_node][j] ** alpha) * (heuristics[current_node][j] ** beta)
            probabilities.append((j, value))
            total += value
    probabilities = [(node, prob / total) for node, prob in probabilities]
    return probabilities

def select_next_node(probabilities):
    rand = random.random()
    cum = 0
    for node, prob in probabilities:
        cum += prob
        if rand <= cum:
            return node
    return probabilities[-1][0]

def evaluate_solution(solution, cost_matrix):
    return sum(cost_matrix[solution[i]][solution[i + 1]] for i in range(len(solution) - 1))

def evaporate_pheremones(pheremones, evap_rate):
    return pheremones * (1 - evap_rate)

def deposit(pheremones, solutions, costs, constant):
    for k, path in enumerate(solutions):
        cost = costs[k]
        for i in range(len(path) - 1):
            pheremones[path[i]][path[i + 1]] += constant / cost
            pheremones[path[i + 1]][path[i]] += constant / cost
    return pheremones

def ant_colony_optimization(cost_matrix, alpha, beta, InitialPheremone, evap_rate, m, constant, I_max):
    num_nodes = len(cost_matrix)
    pheromones = initialise_pheremones(num_nodes, InitialPheremone)
    heuristics = 1 / (cost_matrix + 1e-10)

    best_solution, best_cost = None, float('inf')
    for _ in range(I_max):
        solutions, costs = [], []
        for _ in range(m):
            solution = [random.randint(0, num_nodes - 1)]
            while len(solution) < num_nodes:
                probabilities = calculate_probabilities(pheromones, heuristics, alpha, beta, solution[-1], solution)
                solution.append(select_next_node(probabilities))
            solution.append(solution[0])
            solutions.append(solution)
            costs.append(evaluate_solution(solution, cost_matrix))

        best_idx = np.argmin(costs)
        best_solution, best_cost = solutions[best_idx], costs[best_idx]

        pheromones = evaporate_pheremones(pheromones, evap_rate)
        pheromones = deposit(pheromones, solutions, costs, constant)

    return best_solution, best_cost
