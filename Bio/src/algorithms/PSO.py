import numpy as np
import random


def calculate_distance(city1, city2):
    #Calculate Euclidean distance between two cities.
    return np.linalg.norm(np.array(city1) - np.array(city2))


def calculate_tour_distance(tour, cost_matrix, bridge=None):
    total = sum(cost_matrix[tour[i]][tour[(i + 1) % len(tour)]] for i in range(len(tour)))

    if bridge:
        a, b = bridge
        used = any(
            (tour[i] == a and tour[(i + 1) % len(tour)] == b) or
            (tour[i] == b and tour[(i + 1) % len(tour)] == a)
            for i in range(len(tour))
        )
        if not used:
            total += 1000  # Apply large penalty

    return total


def swap_mutation(tour):
    #Swap two random cities in the tour.
    new_tour = tour.copy()
    i, j = np.random.choice(len(tour), 2, replace=False)
    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
    return new_tour


def get_local_best(swarm, fitnesses, index, neighborhood_size=2):
    #Finds the best solution in a particle's local neighborhood.
    num_particles = len(swarm)

    # Define the ring neighborhood (circular list)
    neighbors = [(index + i) % num_particles for i in range(-neighborhood_size, neighborhood_size + 1) if i != 0]

    # Get the best neighbor's index
    best_neighbor_idx = min(neighbors, key=lambda idx: fitnesses[idx])

    return swarm[best_neighbor_idx]


def run_tsp_pso(num_nodes, num_particles, w, c1, c2, v_max, max_iterations, topology="star", bridge = None):
    cities = np.random.rand(num_nodes, 2) * 100
    cost_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                cost_matrix[i][j] = np.inf
            else:
                cost_matrix[i][j] = np.linalg.norm(cities[i] - cities[j])

    # Soft bridge constraint
    if bridge:
        a, b = bridge
        cost_matrix[a][b] = cost_matrix[b][a] = 1e-5  # near-zero cost
    swarm = [np.random.permutation(num_nodes).tolist() for _ in range(num_particles)]
    velocities = [np.zeros(num_nodes) for _ in range(num_particles)]

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and cost_matrix[i][j] == 0:
                cost_matrix[i][j] = np.linalg.norm(cities[i] - cities[j])

    pBest = swarm.copy()
    pBest_costs = np.array([calculate_tour_distance(t, cost_matrix,bridge=bridge) for t in swarm])

    # For wheel, define hub as index 0
    if topology.lower() == "wheel":
        hub_index = 0
        gBest = pBest[hub_index]
        gBest_cost = pBest_costs[hub_index]
    else:
        gBest_index = np.argmin(pBest_costs)
        gBest = pBest[gBest_index]
        gBest_cost = pBest_costs[gBest_index]

    for iteration in range(max_iterations):
        solution= []

        for i in range(num_particles):
            if topology.lower() == "wheel":
                # All particles follow the hub
                best_neighbor = gBest
            elif topology.lower() == "star":
                best_neighbor = gBest
            elif topology.lower() == "ring":
                best_neighbor = get_local_best(swarm, pBest_costs, i)
            else:
                best_neighbor = gBest  # fallback

            inertia = w * velocities[i]
            cognitive = c1 * np.random.rand() * np.array(pBest[i])
            social = c2 * np.random.rand() * np.array(best_neighbor)
            new_velocity = inertia + cognitive + social
            new_velocity = np.clip(new_velocity, -v_max, v_max)

            velocities[i] = new_velocity
            new_tour = swap_mutation(swarm[i])  # still TSP-specific position update
            new_cost = calculate_tour_distance(new_tour, cost_matrix,bridge=bridge)

            if new_cost < pBest_costs[i]:
                pBest[i] = new_tour
                pBest_costs[i] = new_cost

                if topology.lower() == "wheel" and i == hub_index:
                    gBest = new_tour
                    gBest_cost = new_cost
                elif new_cost < gBest_cost:
                    gBest = new_tour
                    gBest_cost = new_cost

            swarm[i] = new_tour

    return np.array(cities), gBest, gBest_cost
