import numpy as np
import random


def calculate_distance(city1, city2):
    """Calculate Euclidean distance between two cities."""
    return np.linalg.norm(np.array(city1) - np.array(city2))


def calculate_tour_distance(tour, cities):
    """Calculate total distance of a tour."""
    return sum(calculate_distance(cities[tour[i]], cities[tour[(i + 1) % len(tour)]]) for i in range(len(tour)))


def swap_mutation(tour):
    """Swap two random cities in the tour."""
    new_tour = tour.copy()
    i, j = np.random.choice(len(tour), 2, replace=False)
    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
    return new_tour


def get_local_best(swarm, fitnesses, index, neighborhood_size=2):
    """Finds the best solution in a particle's local neighborhood."""
    num_particles = len(swarm)

    # Define the ring neighborhood (circular list)
    neighbors = [(index + i) % num_particles for i in range(-neighborhood_size, neighborhood_size + 1) if i != 0]

    # Get the best neighbor's index
    best_neighbor_idx = min(neighbors, key=lambda idx: fitnesses[idx])

    return swarm[best_neighbor_idx]


def run_tsp_pso(num_nodes, num_particles, w, c1, c2, v_max, max_iterations, topology="star"):
    """Runs PSO for TSP with support for star and ring topology."""

    # Step 1: Generate cities
    cities = np.random.rand(num_nodes, 2) * 100

    # Step 2: Initialize swarm
    swarm = [np.random.permutation(num_nodes).tolist() for _ in range(num_particles)]
    velocities = [np.zeros(num_nodes) for _ in range(num_particles)]  # Initialize velocities

    # Step 3: Compute initial fitness
    pBest = swarm.copy()
    pBest_costs = np.array([calculate_tour_distance(tour, cities) for tour in swarm])

    # Step 4: Initialize global best (gBest)
    gBest_index = np.argmin(pBest_costs)
    gBest = pBest[gBest_index]
    gBest_cost = pBest_costs[gBest_index]

    for iteration in range(max_iterations):
        for i in range(num_particles):
            # Determine the best solution based on topology
            if topology == "star":
                best_neighbor = gBest  # Global best (same as before)
            elif topology == "ring":
                best_neighbor = get_local_best(swarm, pBest_costs, i)  # Local best

            # Compute new velocity based on inertia, cognitive, and social components
            inertia_component = w * velocities[i]
            cognitive_component = c1 * np.random.rand() * np.array(pBest[i])
            social_component = c2 * np.random.rand() * np.array(best_neighbor)

            # Combine components
            new_velocity = inertia_component + cognitive_component + social_component
            new_velocity = np.clip(new_velocity, -v_max, v_max)  # Apply velocity limits
            velocities[i] = new_velocity

            # Update position using velocity (swap mutation for TSP)
            new_tour = swap_mutation(swarm[i])
            new_cost = calculate_tour_distance(new_tour, cities)

            # Update personal best
            if new_cost < pBest_costs[i]:
                pBest[i] = new_tour
                pBest_costs[i] = new_cost

            # Update global best (only for star topology)
            if topology == "star" and new_cost < gBest_cost:
                gBest = new_tour
                gBest_cost = new_cost

            # Update swarm position
            swarm[i] = new_tour

        # Debugging logs
        if iteration % 10 == 0:
            print(f"Iteration {iteration}: Best Distance = {gBest_cost if topology == 'star' else min(pBest_costs)}")

    # Return results based on topology
    if topology == "star":
        return np.array(cities), gBest, gBest_cost
    else:  # Ring topology
        best_ring_idx = np.argmin(pBest_costs)
        return np.array(cities), pBest[best_ring_idx], pBest_costs[best_ring_idx]
