import random
import numpy as np

def calculate_distance(city1, city2):
    return np.linalg.norm(np.array(city1) - np.array(city2))

def calculate_tour_distance(tour, cities):
    return sum(calculate_distance(cities[tour[i]], cities[tour[(i + 1) % len(tour)]]) for i in range(len(tour)))

def run_tsp_pso(particles_num, iterations, num_cities):
    cities = [(random.random() * 100, random.random() * 100) for _ in range(num_cities)]
    particles = [{"position": random.sample(range(num_cities), num_cities), "p_best_dist": float('inf'), "p_best_pos": None} for _ in range(particles_num)]

    g_best_pos, g_best_dist = None, float('inf')

    for _ in range(iterations):
        for particle in particles:
            dist = calculate_tour_distance(particle["position"], cities)
            if dist < particle["p_best_dist"]:
                particle["p_best_dist"], particle["p_best_pos"] = dist, particle["position"]
            if dist < g_best_dist:
                g_best_dist, g_best_pos = dist, particle["position"]

    return cities, g_best_pos, g_best_dist  # Returning results instead of plotting
