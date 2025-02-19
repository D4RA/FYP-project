import numpy as np


def dabc_fns(cost_matrix, sn=10, max_cycle=5000, trial_limit=100):
    num_cities = cost_matrix.shape[0]

    # Step 2: Generate SN initial random solutions
    solutions = [np.random.permutation(num_cities) for _ in range(sn)]
    fitness_values = np.array([calculate_fitness(sol, cost_matrix) for sol in solutions])

    # Initialize tracking variables
    trials = np.zeros(sn, dtype=int)
    cycle = 0
    best_solution = solutions[np.argmin(fitness_values)]
    best_cost = np.min(fitness_values)

    while cycle < max_cycle:
        for i in range(sn):
            # Step 5: Generate a new solution Vi for Xi (random neighbor swap)
            new_solution = mutate_solution(solutions[i])
            new_fitness = calculate_fitness(new_solution, cost_matrix)

            # Step 7: Compare and update solution
            if new_fitness < fitness_values[i]:
                solutions[i] = new_solution
                fitness_values[i] = new_fitness
                trials[i] = 0
            else:
                trials[i] += 1

        # Step 9: Calculate selection probability P(Xi)
        probabilities = calculate_selection_probability(fitness_values)

        for i in range(sn):
            if np.random.rand() < probabilities[i]:  # Step 11: Select solution
                new_solution = mutate_solution(solutions[i])
                new_fitness = calculate_fitness(new_solution, cost_matrix)

                # Step 14: Apply neighborhood search with probability
                if np.random.rand() < 0.5:
                    new_solution = local_search(new_solution)
                    new_fitness = calculate_fitness(new_solution, cost_matrix)

                if new_fitness < fitness_values[i]:
                    solutions[i] = new_solution
                    fitness_values[i] = new_fitness
                    trials[i] = 0
                else:
                    trials[i] += 1

        # Step 17: If trial limit exceeded, replace solution
        for i in range(sn):
            if trials[i] > trial_limit:
                solutions[i] = np.random.permutation(num_cities)
                fitness_values[i] = calculate_fitness(solutions[i], cost_matrix)
                trials[i] = 0

        # Update best solution
        current_best_index = np.argmin(fitness_values)
        if fitness_values[current_best_index] < best_cost:
            best_solution = solutions[current_best_index]
            best_cost = fitness_values[current_best_index]

        # Step 19: Perform swap-based local improvement every 1000 cycles
        if cycle % 1000 == 0:
            best_solution = swap_two_nodes(best_solution)
            best_cost = calculate_fitness(best_solution, cost_matrix)

        cycle += 1

    return best_solution, best_cost


def calculate_fitness(route, cost_matrix):
    """Calculate the total cost (distance) of a given route."""
    return sum(cost_matrix[route[i], route[i + 1]] for i in range(len(route) - 1)) + cost_matrix[route[-1], route[0]]


def mutate_solution(solution):
    """Generates a new solution by swapping two random cities."""
    new_solution = solution.copy()
    i, j = np.random.choice(len(solution), 2, replace=False)
    new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
    return new_solution


def calculate_selection_probability(fitness_values):
    """Calculate selection probability based on fitness values."""
    max_fitness = np.max(fitness_values)
    probabilities = max_fitness - fitness_values  # Higher probability for better solutions
    return probabilities / np.sum(probabilities)


def local_search(solution):
    """Performs a simple local search by swapping two adjacent cities."""
    new_solution = solution.copy()
    i = np.random.randint(len(solution) - 1)
    new_solution[i], new_solution[i + 1] = new_solution[i + 1], new_solution[i]
    return new_solution


def swap_two_nodes(solution):
    """Swaps two nodes in the solution to improve the current best path."""
    new_solution = solution.copy()
    i, j = np.random.choice(len(solution), 2, replace=False)
    new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
    return new_solution
