def plot_tsp_solution(ax, nodes, tour, title="TSP Solution"):
    ax.clear()
    tour_nodes = nodes[tour + [tour[0]]]  # Complete the loop
    ax.plot(tour_nodes[:, 0], tour_nodes[:, 1], 'bo-', markersize=8, label="Tour")
    ax.set_title(title)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")

    for i, (x, y) in enumerate(nodes):
        ax.text(x, y, str(i), fontsize=10, ha="right")

    ax.legend()
