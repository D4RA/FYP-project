import random
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk


particles_num = 30
iterations = 100
num_cities = 4
p_best_factor=  1.5
g_best_factor = 1.5

def calculate_distance(city1, city2):
    return ((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2) ** 0.5

# Helper function: Calculate the total distance of a tour
def calculate_tour_distance(tour, cities):
    distance = 0
    for i in range(len(tour)):
        distance += calculate_distance(cities[tour[i]], cities[tour[(i + 1) % len(tour)]])
    return distance
cities = [(random.random() * 100, random.random() * 100) for _ in range(num_cities)]

particles = []
for i in range(particles_num):
    position = random.sample(range(num_cities),num_cities)
    velocity =[]
    distance = calculate_tour_distance(position,cities)
    particles.append({
        "position":position,"velocity":velocity,"p_best_dist":distance,"p_best_pos":position[:]
    })

g_best_pos = particles[0]["p_best_pos"][:]
g_best_dist = particles[0]["p_best_dist"]
for particle in particles:
    if particle["p_best_dist"] < g_best_dist:
        g_best_pos = particle["p_best_pos"][:]
        g_best_dist = particle["p_best_dist"]

def update_velocity(particle,g_best_pos):
    velocity = []

    for i in range(len(particle["position"])):
        if random.random()<p_best_factor:
            if particle["position"][i] != particle["p_best_pos"][i]:
                velocity.append((i,particle["p_best_pos"].index(particle["position"][i])))

        if random.random() < g_best_factor:
            if particle["position"][i] != g_best_pos[i]:
                velocity.append((i,g_best_pos.index(particle["position"][i])))
    return velocity

def update_position(position,velocity):
    position = position[:]
    for swap in velocity:
        i,j = swap
        position[i],position[j] = position[j],position[i]

    return position

for iteration in range(iterations):
    for particle in particles:
        # Update the global best
        if particle["p_best_dist"] < g_best_dist:
            g_best_pos = particle["p_best_pos"][:]
            g_best_dist = particle["p_best_dist"]

    for particle in particles:
        # Update velocity and position
        particle["velocity"] = update_velocity(particle, g_best_pos)
        particle["position"] = update_position(particle["position"], particle["velocity"])

        # Calculate the new distance
        new_distance = calculate_tour_distance(particle["position"], cities)

        # Update personal best if necessary
        if new_distance < particle["p_best_dist"]:
            particle["p_best_pos"] = particle["position"][:]
            particle["p_best_dist"] = new_distance

    print(f"Iteration {iteration + 1}: Global Best Distance = {g_best_dist}")

# Output the final result
print("\nOptimal Tour:", g_best_pos)
print("Optimal Distance:", g_best_dist)


# def plot_tsp(cities, tour):
#     plt.figure(figsize=(8, 8))
#
#     # Plot cities
#     x = [city[0] for city in cities]
#     y = [city[1] for city in cities]
#     plt.scatter(x, y, color='red', s=100, label='Cities')
#
#     # Annotate city indices
#     for i, (xi, yi) in enumerate(cities):
#         plt.text(xi + 1, yi + 1, str(i), fontsize=10, color='blue')
#
#     # Plot the tour
#     tour_cities = [cities[i] for i in tour] + [cities[tour[0]]]  # Complete the loop
#     tour_x = [city[0] for city in tour_cities]
#     tour_y = [city[1] for city in tour_cities]
#     plt.plot(tour_x, tour_y, color='black', linestyle='-', linewidth=1, label='Optimal Tour')
#
#     # Add title and legend
#     plt.title('TSP Optimal Tour Found by PSO', fontsize=16)
#     plt.xlabel('X Coordinate')
#     plt.ylabel('Y Coordinate')
#     plt.legend()
#     plt.grid()
#     plt.show()
#
#
# # Plot the optimal tour
# plot_tsp(cities, g_best_pos)

def run_pso():
    try:
        particles = int(particles_var.get())
        max_iter = int(iterations_var.get())
        w = float(w_var.get())
        c1 = float(c1_var.get())
        c2 = float(c2_var.get())

        matrix = np.array([
            [0, 10, 15, 20],
            [5, 0, 15, 35],
            [15, 20, 0, 30],
            [5, 10, 15, 0]
        ])

        best_tour, best_distance = pso(4, matrix, particles, max_iter, w, c1, c2)
        result_label.config(text=f"Best Tour: {best_tour}\nBest Distance: {best_distance:.4f}")
    except Exception as e:
        result_label.config(text=f"Error: {e}")

root = tk.Tk()
root.title("PSO Parameter Tuner")

# Parameters Frame
param_frame = ttk.LabelFrame(root, text="PSO Parameters", padding=10)
param_frame.grid(row=0, column=0, padx=10, pady=10)

# Particles
ttk.Label(param_frame, text="Particles:").grid(row=0, column=0, sticky="w")
particles_var = tk.StringVar(value="30")
particles_entry = ttk.Entry(param_frame, textvariable=particles_var, width=10)
particles_entry.grid(row=0, column=1, padx=5, pady=5)

# Iterations
ttk.Label(param_frame, text="Max Iterations:").grid(row=1, column=0, sticky="w")
iterations_var = tk.StringVar(value="100")
iterations_entry = ttk.Entry(param_frame, textvariable=iterations_var, width=10)
iterations_entry.grid(row=1, column=1, padx=5, pady=5)

# w
ttk.Label(param_frame, text="Inertia (w):").grid(row=2, column=0, sticky="w")
w_var = tk.StringVar(value="0.5")
w_entry = ttk.Entry(param_frame, textvariable=w_var, width=10)
w_entry.grid(row=2, column=1, padx=5, pady=5)

# c1
ttk.Label(param_frame, text="Cognitive (c1):").grid(row=3, column=0, sticky="w")
c1_var = tk.StringVar(value="1.5")
c1_entry = ttk.Entry(param_frame, textvariable=c1_var, width=10)
c1_entry.grid(row=3, column=1, padx=5, pady=5)

# c2
ttk.Label(param_frame, text="Social (c2):").grid(row=4, column=0, sticky="w")
c2_var = tk.StringVar(value="1.5")
c2_entry = ttk.Entry(param_frame, textvariable=c2_var, width=10)
c2_entry.grid(row=4, column=1, padx=5, pady=5)

# Run Button
run_button = ttk.Button(root, text="Run PSO", command=run_pso)
run_button.grid(row=1, column=0, pady=10)

# Result Label
result_label = ttk.Label(root, text="", padding=10)
result_label.grid(row=2, column=0, pady=10)

# Run the application
print("")
root.mainloop()