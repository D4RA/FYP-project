import random
import numpy as np
import matplotlib.pyplot as plt


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
    if total == 0:
        probabilities = [(j, 1) for j in range(len(pheremones)) if j not in visited]
        total = len(probabilities)

    probabilities = [(node, prob / total) for node, prob in probabilities]
    return probabilities


def select_next_node(probabilites):
    rand = random.random()
    cum = 0
    for node, prob in probabilites:
        cum += prob
        if rand <= cum:
            return node
    return probabilites[-1][0]


def evaluate_solution(solution, cost_matrix):
    total_cost = 0
    for i in range(len(solution) - 1):
        total_cost += cost_matrix[solution[i]][solution[i + 1]]
    return total_cost


def evaporate_pheremones(pheremones, evap_rate):
    return pheremones * (1 - evap_rate)


def deposit( pheremones,solutions, costs, constant):
    for k, path in enumerate(solutions):
        cost = costs[k]
        for i in range(len(path) - 1):
            nodei = path[i]
            nodej = path[i + 1]
            pheremones[int(nodei)][int(nodej)] += constant / cost
            pheremones[int(nodej)][int(nodei)] += constant / cost
        return pheremones


def ant_colony_optimization(cost_matrix, alpha, beta, InitialPheremone, evap_rate, m, constant, I_max):
    
    num_nodes = len(cost_matrix)
    pheromones = initialise_pheremones(num_nodes, InitialPheremone)
    heuristics = 1 / (cost_matrix + 1e-10)  # Heuristic is inverse of cost

    best_solution = None
    best_cost = float('inf')

    for t in range(I_max):
        solutions = []
        costs = []

        # Step 1: Each ant builds a solution
        for k in range(m):
            visited = []
            current_node = random.randint(0, num_nodes - 1)
            solution = [current_node]
            visited.append(current_node)

            while len(visited) < num_nodes:
                probabilities = calculate_probabilities(pheromones, heuristics, alpha, beta, current_node, visited)
                next_node = select_next_node(probabilities)
                solution.append(next_node)
                visited.append(next_node)
                current_node = next_node

            solution.append(solution[0])  # Return to the starting node
            solutions.append(solution)
            costs.append(evaluate_solution(solution, cost_matrix))

        # Step 2: Update the best solution
        for k in range(m):
            if costs[k] < best_cost:
                best_cost = costs[k]
                best_solution = solutions[k]

        # Step 3: Update pheromones
        pheromones = evaporate_pheremones(pheromones, evap_rate)
        pheromones = deposit(pheromones, solutions, costs, constant)

    return best_solution, best_cost


def plot_tsp_solution(nodes, solution, title="TSP Solution"):
    """
    Plot the TSP solution on a 2D graph.

    Parameters:
    - nodes: A list of (x, y) coordinates for the nodes.
    - solution: A list representing the order of nodes in the solution.
    - title: Title of the plot.
    """
    plt.figure(figsize=(8, 6))

    # Extract the coordinates of the nodes
    x = [nodes[i][0] for i in solution]
    y = [nodes[i][1] for i in solution]

    # Plot the nodes
    plt.scatter(x, y, c="red", s=50, label="Nodes")
    for i, (x_coord, y_coord) in enumerate(nodes):
        plt.text(x_coord, y_coord, f"{i}", fontsize=10, ha="right")

    # Plot the solution path
    plt.plot(x, y, c="blue", linestyle="--", label="Path")

    # Customize the plot
    plt.title(title)
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Example TSP problem with node coordinates
    nodes = [
        (0, 0),  # Node 0
        (10, 0),  # Node 1
        (10, 20),  # Node 2
        (0, 10)  # Node 3
    ]

    # Generate cost matrix based on Euclidean distance
    num_nodes = len(nodes)
    cost_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            cost_matrix[i][j] = np.linalg.norm(np.array(nodes[i]) - np.array(nodes[j]))

    alpha = 1  # Influence of pheromones
    beta = 2  # Influence of heuristics
    InitialPheremone = 1.0  # Initial pheromone levels
    evap_rate = 0.5  # Pheromone evaporation rate
    m = 5  # Number of ants
    constant = 100  # Pheromone deposit constant
    I_max = 100  # Number of iterations

    # Run the ACO algorithm
    best_solution, best_cost = ant_colony_optimization(cost_matrix, alpha, beta, InitialPheremone, evap_rate, m, constant,I_max)

    # Print the results
    print("Best solution:", best_solution)
    print("Best cost:", best_cost)

    # Visualize the solution
    plot_tsp_solution(nodes, best_solution, title=f"TSP Solution with Cost: {best_cost:.2f}")
