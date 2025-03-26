from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
from algorithms.GA import run_tsp_ga  # Import GA function

class GAWorker(QThread):
    """Runs the Genetic Algorithm (GA) in a separate thread."""
    finished = pyqtSignal(np.ndarray, list, float)  # Signal to send results back

    def __init__(self, num_cities, population_size, generations, mutation_rate):
        super().__init__()
        self.num_cities = num_cities
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def run(self):
        """Runs the GA and emits the result when finished."""
        cities, best_tour, best_distance = run_tsp_ga(
            self.num_cities, self.population_size, self.generations, self.mutation_rate
        )
        self.finished.emit(cities, best_tour, best_distance)  # Send results back
