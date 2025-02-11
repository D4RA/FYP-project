import numpy as np
import matplotlib.pyplot as plt

# Utility function to calculate the Euclidean distance matrix between nodes
def calculate_distance_matrix(nodes):
    num_nodes = len(nodes)
    distance_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            distance_matrix[i][j] = np.linalg.norm(np.array(nodes[i]) - np.array(nodes[j]))
    return distance_matrix

# Utility function to generate random coordinates for nodes
def generate_random_nodes(num_nodes, x_range=(0, 100), y_range=(0, 100)):
    nodes = [(np.random.uniform(*x_range), np.random.uniform(*y_range)) for _ in range(num_nodes)]
    return nodes

# Utility function to plot a TSP solution
def plot_tsp_solution(nodes, solution, title="TSP Solution"):
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

# Utility function to print the results
def print_results(algorithm_name, best_solution, best_cost):
    print(f"{algorithm_name} Results:")
    print("Best solution:", best_solution)
    print("Best cost:", best_cost)
    print("-" * 40)
