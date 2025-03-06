import numpy as np

def dabc_fns(cost_matrix, sn=10, max_cycle=5000, trial_limit=100):
    num_cities = cost_matrix.shape[0]
    solutions = [np.random.permutation(num_cities) for _ in range(sn)]
    fitness_values = np.array([calculate_fitness(sol, cost_matrix) for sol in solutions])
    trials = np.zeros(sn, dtype=int)

    best_index = np.argmin(fitness_values)
    best_solution, best_cost = solutions[best_index], fitness_values[best_index]

    for cycle in range(max_cycle):
        for i in range(sn):
            new_solution = mutate_solution(solutions[i])
            new_fitness = calculate_fitness(new_solution, cost_matrix)

            if new_fitness < fitness_values[i]:  # Accept better solution
                solutions[i], fitness_values[i], trials[i] = new_solution, new_fitness, 0
            else:
                trials[i] += 1

        # Selection Probability Calculation
        probabilities = calculate_selection_probability(fitness_values)

        # Employed Bees - Probabilistic Selection
        for i in range(sn):
            if np.random.rand() < probabilities[i]:
                new_solution = mutate_solution(solutions[i])
                new_fitness = calculate_fitness(new_solution, cost_matrix)

                if np.random.rand() < 0.5:  # Apply local search with 50% chance
                    new_solution = local_search(new_solution)
                    new_fitness = calculate_fitness(new_solution, cost_matrix)

                if new_fitness < fitness_values[i]:
                    solutions[i], fitness_values[i], trials[i] = new_solution, new_fitness, 0
                else:
                    trials[i] += 1

        # Abandon and replace poor solutions
        for i in range(sn):
            if trials[i] > trial_limit:
                solutions[i] = np.random.permutation(num_cities)
                fitness_values[i] = calculate_fitness(solutions[i], cost_matrix)
                trials[i] = 0

        # Update global best
        best_index = np.argmin(fitness_values)
        if fitness_values[best_index] < best_cost:
            best_solution, best_cost = solutions[best_index], fitness_values[best_index]

    return best_solution, best_cost

def calculate_fitness(route, cost_matrix):
    "Calculate the total cost (distance) of a given route."
    return sum(cost_matrix[route[i], route[i + 1]] for i in range(len(route) - 1)) + cost_matrix[route[-1], route[0]]

def mutate_solution(solution):
    "Generates a new solution by swapping two random cities."
    new_solution = solution.copy()
    i, j = np.random.choice(len(solution), 2, replace=False)
    new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
    return new_solution

def calculate_selection_probability(fitness_values):
    "Calculate selection probability (higher probability for better solutions)."
    adjusted_fitness = np.max(fitness_values) - fitness_values
    adjusted_fitness += 1e-10  # Avoid division by zero
    return adjusted_fitness / np.sum(adjusted_fitness)

def local_search(solution):
    "Performs a simple local search by swapping two adjacent cities."
    new_solution = solution.copy()
    i = np.random.randint(len(solution) - 1)
    new_solution[i], new_solution[i + 1] = new_solution[i + 1], new_solution[i]
    return new_solution
