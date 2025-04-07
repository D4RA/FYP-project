import numpy as np
import random


def calculate_distance(city1, city2):
    "Calculate Euclidean distance between two cities"
    return np.linalg.norm(np.array(city1) - np.array(city2))


def calculate_tour_distance(tour, cities):
    "Calculate total distance of a tour"
    return sum(calculate_distance(cities[tour[i]], cities[tour[(i + 1) % len(tour)]]) for i in range(len(tour)))


def ordered_crossover(parent1, parent2):
    "Performs Ordered Crossover (OX) for TSP."
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))

    child = [-1] * size
    child[start:end] = parent1[start:end]

    parent2_remaining = [city for city in parent2 if city not in child]
    index = 0

    for i in range(size):
        if child[i] == -1:
            child[i] = parent2_remaining[index]
            index += 1

    return child


def partially_mapped_crossover(parent1, parent2):
    """Robust PMX implementation for TSP."""
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))

    child = [-1] * size
    # Step 1: Copy the slice from parent1 to child
    child[start:end] = parent1[start:end]

    # Step 2: Create mapping from parent2 slice
    for i in range(start, end):
        if parent2[i] not in child:
            val = parent2[i]
            pos = i
            while True:
                val_in_parent1 = parent1[pos]
                pos = parent2.index(val_in_parent1)
                if child[pos] == -1:
                    child[pos] = val
                    break

    # Step 3: Fill in remaining positions from parent2
    for i in range(size):
        if child[i] == -1:
            child[i] = parent2[i]

    return child


def cycle_crossover(parent1, parent2):
    """Performs robust Cycle Crossover (CX) for TSP."""
    size = len(parent1)
    child = [-1] * size
    index = 0
    visited = [False] * size

    while not all(visited):
        if visited[index]:
            index = visited.index(False)

        start = index
        val = parent1[index]

        # Start forming cycle
        while True:
            child[index] = parent1[index]
            visited[index] = True
            index = parent1.index(parent2[index])
            if index == start:
                break

        # Fill the rest from parent2
        for i in range(size):
            if not visited[i]:
                child[i] = parent2[i]

    return child

def swap_mutation(tour, mutation_rate):
    "Swap two cities in the tour with probability mutation_rate."
    if random.random() < mutation_rate:
        i, j = np.random.choice(len(tour), 2, replace=False)
        tour[i], tour[j] = tour[j], tour[i]
    return tour


def tournament_selection(population, fitnesses, tournament_size=3):
    "Selects a parent using tournament selection."
    selected_indices = np.random.choice(len(population), tournament_size, replace=False)
    best_index = selected_indices[np.argmin([fitnesses[i] for i in selected_indices])]
    return population[best_index]


def run_tsp_ga(num_cities, population_size, generations, mutation_rate, crossover_type="Random Selection"):
    """Runs Genetic Algorithm for TSP with user-selected crossover."""

    cities = np.random.rand(num_cities, 2) * 100
    population = [np.random.permutation(num_cities).tolist() for _ in range(population_size)]
    fitnesses = np.array([calculate_tour_distance(tour, cities) for tour in population])

    best_tour = population[np.argmin(fitnesses)]
    best_distance = min(fitnesses)

    for gen in range(generations):
        new_population = []

        for _ in range(population_size // 2):
            # Selection
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)

            # Choose crossover type based on user selection
            if crossover_type == "Order Crossover (OX1)":
                child1, child2 = ordered_crossover(parent1, parent2), ordered_crossover(parent2, parent1)

            elif crossover_type == "Cycle Crossover (CX)":
                child1, child2 = cycle_crossover(parent1, parent2), cycle_crossover(parent2, parent1)

            elif crossover_type == "Partially Mapped Crossover (PMX)":
                child1,child2 = partially_mapped_crossover(parent1,parent2), partially_mapped_crossover(parent1, parent2)

            # Mutation
            child1, child2 = swap_mutation(child1, mutation_rate), swap_mutation(child2, mutation_rate)

            new_population.extend([child1, child2])

        # Evaluate new population
        new_fitnesses = np.array([calculate_tour_distance(tour, cities) for tour in new_population])

        # Elitism: Keep the best individual from the previous generation
        best_index = np.argmin(fitnesses)
        best_tour = population[best_index]
        best_distance = fitnesses[best_index]

        # Replace the worst individual in new population with best from previous gen
        worst_index = np.argmax(new_fitnesses)
        new_population[worst_index] = best_tour
        new_fitnesses[worst_index] = best_distance

        # Update population
        population = new_population
        fitnesses = new_fitnesses

        # Update best solution
        if min(new_fitnesses) < best_distance:
            best_tour = new_population[np.argmin(new_fitnesses)]
            best_distance = min(new_fitnesses)

    return np.array(cities), best_tour, best_distance
