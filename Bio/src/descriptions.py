descriptions = {
    "Number of Nodes (Cities):": "Specifies how many cities are included in the Traveling Salesman Problem (TSP). More cities increase the problem's complexity.",
    "Max Iterations:": "Sets how many times the algorithm will try to improve the solution. Higher values can lead to better results but increase runtime.",

    "Alpha (ACO):": "Determines how strongly ants follow existing pheromone trails. Higher values make ants prefer previously discovered paths.",
    "Beta (ACO):": "Controls how much ants rely on visible distance (heuristic information). A high beta means ants prefer shorter paths.",
    "Evaporation Rate (ACO):": "Regulates how quickly pheromones fade. Helps balance exploration and exploitation by preventing early convergence on suboptimal paths.",
    "Initial Pheromone (ACO):": "Sets the starting pheromone level on all paths. A higher initial value can lead to faster convergence but may cause premature exploitation of suboptimal routes.",
    "Number of Ants (ACO):": "Determines how many ants (agents) explore paths in each iteration. More ants improve coverage and solution diversity but increase computation time.",
    "Deposit Constant (ACO):": "Controls the amount of pheromone each ant deposits on the path it takes. Larger values intensify the reinforcement of good paths, speeding up convergence.",

    "Number of Particles (PSO):": "Sets how many candidate solutions (particles) explore the search space. More particles increase coverage but require more computation.",
    "Inertia Weight (w - PSO):": "Controls momentum: higher values encourage exploration, lower values focus on refining current paths.",
    "Cognitive Coefficient (c1 - PSO):": "Defines how much each particle is influenced by its own past best solution (self-learning).",
    "Social Coefficient (c2 - PSO):": "Defines how much each particle is influenced by the best solution found by others (swarm intelligence).",
"Max Velocity (v_max - PSO):": "Limits how much a particle's position can change in a single iteration. Lower values promote fine-tuned local search, while higher values allow broader exploration. It helps balance between exploration and exploitation in PSO.",

    "Population Size (GA):": "Specifies the number of potential solutions (individuals) in each generation. Larger populations offer more diversity.",
    "Generations (GA):": "Sets how many evolutionary cycles will occur. More generations allow the algorithm to refine better solutions.",
    "Mutation Rate (GA):": "Probability that random changes will be introduced in offspring. Encourages diversity and helps escape local optima.A higher Mutation encourages more change in the solution, while a lower one preserves optimal solutions",

    "Swarm Size (GBC):": "Specifies the number of individual solutions (bees) in the colony. A larger swarm may explore the solution space more thoroughly, but increases computational time.",
    "Max Cycles (GBC):": "Defines the maximum number of iterations the algorithm will perform. Each cycle involves all bees searching for better solutions and updating the colony’s knowledge.",
    "Trial Limit (GBC):": "The number of unsuccessful attempts a solution (bee) can make before it is abandoned and replaced. Helps the algorithm escape local optima by introducing new random solutions."
}

