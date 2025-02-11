import random
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk

# PSO Parameters (default values)
p_best_factor = 1.5
g_best_factor = 1.5

# Helper Function: Calculate Distance
def calculate_distance(city1, city2):
    return ((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2) ** 0.5

# Helper Function: Calculate Tour Distance
def calculate_tour_distance(tour, cities):
    distance = 0
    for i in range(len(tour)):
        distance += calculate_distance(cities[tour[i]], cities[tour[(i + 1) % len(tour)]])
    return distance

# Update Velocity for a Particle
def update_velocity(particle, g_best_pos):
    velocity = []
    for i in range(len(particle["position"])):
        if random.random() < p_best_factor:
            if particle["position"][i] != particle["p_best_pos"][i]:
                velocity.append((i, particle["p_best_pos"].index(particle["position"][i])))

        if random.random() < g_best_factor:
            if particle["position"][i] != g_best_pos[i]:
                velocity.append((i, g_best_pos.index(particle["position"][i])))
    return velocity

# Update Position Based on Velocity
def update_position(position, velocity):
    position = position[:]
    for swap in velocity:
        i, j = swap
        position[i], position[j] = position[j], position[i]
    return position

# Real-Time PSO Algorithm with Plot Updates
def run_tsp_pso(particles_num, iterations, num_cities):
    cities = [(random.random() * 100, random.random() * 100) for _ in range(num_cities)]
    particles = []

    for _ in range(particles_num):
        position = random.sample(range(num_cities), num_cities)
        velocity = []
        distance = calculate_tour_distance(position, cities)
        particles.append({
            "position": position,
            "velocity": velocity,
            "p_best_dist": distance,
            "p_best_pos": position[:]
        })

    g_best_pos = particles[0]["p_best_pos"][:]
    g_best_dist = particles[0]["p_best_dist"]

    for particle in particles:
        if particle["p_best_dist"] < g_best_dist:
            g_best_pos = particle["p_best_pos"][:]
            g_best_dist = particle["p_best_dist"]

    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots(figsize=(8, 8))

    for iteration in range(iterations):
        for particle in particles:
            if particle["p_best_dist"] < g_best_dist:
                g_best_pos = particle["p_best_pos"][:]
                g_best_dist = particle["p_best_dist"]

        for particle in particles:
            particle["velocity"] = update_velocity(particle, g_best_pos)
            particle["position"] = update_position(particle["position"], particle["velocity"])
            new_distance = calculate_tour_distance(particle["position"], cities)

            if new_distance < particle["p_best_dist"]:
                particle["p_best_pos"] = particle["position"][:]
                particle["p_best_dist"] = new_distance

        # Update Plot
        ax.clear()
        x = [city[0] for city in cities]
        y = [city[1] for city in cities]
        ax.scatter(x, y, color='red', s=100, label='Cities')
        for i, (xi, yi) in enumerate(cities):
            ax.text(xi + 1, yi + 1, str(i), fontsize=10, color='blue')

        tour_cities = [cities[i] for i in g_best_pos] + [cities[g_best_pos[0]]]
        tour_x = [city[0] for city in tour_cities]
        tour_y = [city[1] for city in tour_cities]
        ax.plot(tour_x, tour_y, color='black', linestyle='-', linewidth=1, label=f'Best Distance: {g_best_dist:.2f}')
        ax.set_title(f'Iteration {iteration + 1}/{iterations}')
        ax.legend()
        ax.grid()
        plt.pause(0.1)  # Pause for visualization

    plt.ioff()  # Disable interactive mode
    plt.show()  # Show final plot

    return cities, g_best_pos, g_best_dist

# Tkinter GUI
def run_pso():
    try:
        particles = int(particles_var.get())
        max_iter = int(iterations_var.get())
        num_cities = int(cities_var.get())

        cities, best_tour, best_distance = run_tsp_pso(particles, max_iter, num_cities)
        result_label.config(text=f"Best Tour: {best_tour}\nBest Distance: {best_distance:.4f}")

    except Exception as e:
        result_label.config(text=f"Error: {e}")

root = tk.Tk()
root.title("PSO Parameter Tuner")

param_frame = ttk.LabelFrame(root, text="PSO Parameters", padding=10)
param_frame.grid(row=0, column=0, padx=10, pady=10)

ttk.Label(param_frame, text="Particles:").grid(row=0, column=0, sticky="w")
particles_var = tk.StringVar(value="30")
particles_entry = ttk.Entry(param_frame, textvariable=particles_var, width=10)
particles_entry.grid(row=0, column=1, padx=5, pady=5)

ttk.Label(param_frame, text="Max Iterations:").grid(row=1, column=0, sticky="w")
iterations_var = tk.StringVar(value="100")
iterations_entry = ttk.Entry(param_frame, textvariable=iterations_var, width=10)
iterations_entry.grid(row=1, column=1, padx=5, pady=5)

ttk.Label(param_frame, text="Cities:").grid(row=2, column=0, sticky="w")
cities_var = tk.StringVar(value="4")
cities_entry = ttk.Entry(param_frame, textvariable=cities_var, width=10)
cities_entry.grid(row=2, column=1, padx=5, pady=5)

run_button = ttk.Button(root, text="Run PSO", command=run_pso)
run_button.grid(row=1, column=0, pady=10)

result_label = ttk.Label(root, text="", padding=10)
result_label.grid(row=2, column=0, pady=10)

root.mainloop()
