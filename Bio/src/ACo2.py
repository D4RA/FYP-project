import random
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk

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

def deposit(pheremones, solutions, costs, constant):
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
    heuristics = 1 / (cost_matrix + 1e-10)

    best_solution = None
    best_cost = float('inf')

    for t in range(I_max):
        solutions = []
        costs = []

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

            solution.append(solution[0])
            solutions.append(solution)
            costs.append(evaluate_solution(solution, cost_matrix))

        for k in range(m):
            if costs[k] < best_cost:
                best_cost = costs[k]
                best_solution = solutions[k]

        pheromones = evaporate_pheremones(pheromones, evap_rate)
        pheromones = deposit(pheromones, solutions, costs, constant)

    return best_solution, best_cost

def plot_tsp_solution(nodes, solution, title="TSP Solution"):
    plt.figure(figsize=(8, 6))
    x = [nodes[i][0] for i in solution]
    y = [nodes[i][1] for i in solution]

    plt.scatter(x, y, c="red", s=50, label="Nodes")
    for i, (x_coord, y_coord) in enumerate(nodes):
        plt.text(x_coord, y_coord, f"{i}", fontsize=10, ha="right")

    plt.plot(x, y, c="blue", linestyle="--", label="Path")

    plt.title(title)
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.legend()
    plt.grid(True)
    plt.show()

def generate_random_nodes(num_nodes):
    return [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(num_nodes)]

def run_aco():
    try:
        num_nodes = int(nodes_var.get())
        alpha = float(alpha_var.get())
        beta = float(beta_var.get())
        InitialPheremone = float(pheromone_var.get())
        evap_rate = float(evap_rate_var.get())
        m = int(ants_var.get())
        constant = float(deposit_var.get())
        I_max = int(iterations_var.get())

        nodes = generate_random_nodes(num_nodes)
        cost_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                cost_matrix[i][j] = np.linalg.norm(np.array(nodes[i]) - np.array(nodes[j]))

        best_solution, best_cost = ant_colony_optimization(cost_matrix, alpha, beta, InitialPheremone, evap_rate, m, constant, I_max)

        plot_tsp_solution(nodes, best_solution, title=f"TSP Solution with Cost: {best_cost:.2f}")

        result_label.config(text=f"Best Solution: {best_solution}\nBest Cost: {best_cost:.2f}")
    except Exception as e:
        result_label.config(text=f"Error: {e}")

root = tk.Tk()
root.title("ACO Parameter Tuner")

param_frame = ttk.LabelFrame(root, text="ACO Parameters", padding=10)
param_frame.grid(row=0, column=0, padx=10, pady=10)

ttk.Label(param_frame, text="Number of Nodes:").grid(row=0, column=0, sticky="w")
nodes_var = tk.StringVar(value="5")
nodes_entry = ttk.Entry(param_frame, textvariable=nodes_var, width=10)
nodes_entry.grid(row=0, column=1, padx=5, pady=5)

ttk.Label(param_frame, text="Alpha:").grid(row=1, column=0, sticky="w")
alpha_var = tk.StringVar(value="1.0")
alpha_entry = ttk.Entry(param_frame, textvariable=alpha_var, width=10)
alpha_entry.grid(row=1, column=1, padx=5, pady=5)

ttk.Label(param_frame, text="Beta:").grid(row=2, column=0, sticky="w")
beta_var = tk.StringVar(value="2.0")
beta_entry = ttk.Entry(param_frame, textvariable=beta_var, width=10)
beta_entry.grid(row=2, column=1, padx=5, pady=5)

ttk.Label(param_frame, text="Initial Pheromone:").grid(row=3, column=0, sticky="w")
pheromone_var = tk.StringVar(value="1.0")
pheromone_entry = ttk.Entry(param_frame, textvariable=pheromone_var, width=10)
pheromone_entry.grid(row=3, column=1, padx=5, pady=5)

ttk.Label(param_frame, text="Evaporation Rate:").grid(row=4, column=0, sticky="w")
evap_rate_var = tk.StringVar(value="0.5")
evap_rate_entry = ttk.Entry(param_frame, textvariable=evap_rate_var, width=10)
evap_rate_entry.grid(row=4, column=1, padx=5, pady=5)

ttk.Label(param_frame, text="Number of Ants:").grid(row=5, column=0, sticky="w")
ants_var = tk.StringVar(value="5")
ants_entry = ttk.Entry(param_frame, textvariable=ants_var, width=10)
ants_entry.grid(row=5, column=1, padx=5, pady=5)

ttk.Label(param_frame, text="Deposit Constant:").grid(row=6, column=0, sticky="w")
deposit_var = tk.StringVar(value="100.0")
deposit_entry = ttk.Entry(param_frame, textvariable=deposit_var, width=10)
deposit_entry.grid(row=6, column=1, padx=5, pady=5)

ttk.Label(param_frame, text="Max Iterations:").grid(row=7, column=0, sticky="w")
iterations_var = tk.StringVar(value="100")
iterations_entry = ttk.Entry(param_frame, textvariable=iterations_var, width=10)
iterations_entry.grid(row=7, column=1, padx=5, pady=5)

run_button = ttk.Button(root, text="Run ACO", command=run_aco)
run_button.grid(row=1, column=0, pady=10)

result_label = ttk.Label(root, text="", padding=10)
result_label.grid(row=2, column=0, pady=10)

root.mainloop()
