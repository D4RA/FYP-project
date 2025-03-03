import matplotlib.pyplot as plt

def plot_tsp_solution(ax, cities, best_solution, title="TSP Solution"):
    """Plots the best route found by the algorithm."""
    ax.clear()  # Clear previous plot

    # Extract x, y coordinates for the cities
    x, y = zip(*[cities[i] for i in best_solution] + [cities[best_solution[0]]])  # Close the loop

    # Plot the tour
    ax.plot(x, y, 'bo-', markersize=8, label="Cities & Path")  # Draw path
    ax.plot(x[0], y[0], 'ro', markersize=10, label="Start")  # Mark start city
    ax.set_title(title)
    ax.legend()
