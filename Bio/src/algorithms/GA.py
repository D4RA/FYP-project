import numpy as np
import random


def calculate_distance(city1, city2):
    """Calculate Euclidean distance between two cities."""
    return np.linalg.norm(np.array(city1) - np.array(city2))


def calculate_tour_distance(tour, cities):
    """Calculate total distance of a tour."""
    return sum(calculate_distance(cities[tour[i]], cities[tour[(i + 1) % len(tour)]]) for i in range(len(tour)))


def ordered_crossover(parent1, parent2):
    """Performs Ordered Crossover (OX) for TSP."""
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


def swap_mutation(tour, mutation_rate):
    """Swap two cities in the tour with probability mutation_rate."""
    if random.random() < mutation_rate:
        i, j = np.random.choice(len(tour), 2, replace=False)
        tour[i], tour[j] = tour[j], tour[i]
    return tour


def tournament_selection(population, fitnesses, tournament_size=3):
    """Selects a parent using tournament selection."""
    selected_indices = np.random.choice(len(population), tournament_size, replace=False)
    best_index = selected_indices[np.argmin([fitnesses[i] for i in selected_indices])]
    return population[best_index]


def run_tsp_ga(num_cities, population_size, generations, mutation_rate):
    """Runs Genetic Algorithm for TSP."""

    # Generate random city coordinates
    cities = np.random.rand(num_cities, 2) * 100

    # Initialize population (random permutations of cities)
    population = [np.random.permutation(num_cities).tolist() for _ in range(population_size)]

    # Compute initial fitness values
    fitnesses = np.array([calculate_tour_distance(tour, cities) for tour in population])

    best_tour = population[np.argmin(fitnesses)]
    best_distance = min(fitnesses)

    for gen in range(generations):
        new_population = []

        for _ in range(population_size // 2):
            # Selection
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)

            # Crossover
            child1, child2 = ordered_crossover(parent1, parent2), ordered_crossover(parent2, parent1)

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
