import random
import numpy as np

def calculate_distance(city1, city2):
    return np.linalg.norm(np.array(city1) - np.array(city2))

def calculate_tour_distance(tour, cities):
    return sum(calculate_distance(cities[tour[i]], cities[tour[(i + 1) % len(tour)]]) for i in range(len(tour)))

def run_tsp_pso(swarm_size, max_iterations, num_nodes):
    # Ensure cities are a NumPy array
    cities = np.random.rand(num_nodes, 2) * 100  # Simulating city coordinates
    best_tour = list(range(num_nodes))  # Simulating best tour (permutation of indices)
    np.random.shuffle(best_tour)
    best_distance = np.random.rand() * 500  # Simulating distance

    return np.array(cities), best_tour, best_distance  # Ensure cities is a NumPy arrayd
