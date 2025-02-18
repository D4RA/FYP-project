import matplotlib.pyplot as plt

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
