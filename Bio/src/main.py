import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QLabel, QLineEdit, QFrame
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from algorithms.ACo2 import ant_colony_optimization
from algorithms.pso import run_tsp_pso
from algorithms.GBC import dabc_fns

class TSPApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TSP Solver")
        self.setGeometry(100, 100, 800, 500)  # Adjusted window size

        # **Main Layout**
        self.main_layout = QHBoxLayout(self)

        # **Sidebar (Menu) - Left Panel**
        self.sidebar = QVBoxLayout()
        self.sidebar.setAlignment(Qt.AlignTop)

        # Algorithm Selector
        self.algorithm_selector = QComboBox()
        self.algorithm_selector.addItems(["Ant Colony Optimization (ACO)", "Particle Swarm Optimization (PSO)", "DABC_FNS(GBC)"])
        self.algorithm_selector.currentIndexChanged.connect(self.update_ui)
        self.sidebar.addWidget(QLabel("Select Algorithm:"))
        self.sidebar.addWidget(self.algorithm_selector)

        # Common Inputs
        self.node_label = QLabel("Number of Nodes (Cities):")
        self.sidebar.addWidget(self.node_label)
        self.node_input = QLineEdit("10")
        self.sidebar.addWidget(self.node_input)

        self.iter_label = QLabel("Max Iterations:")
        self.sidebar.addWidget(self.iter_label)
        self.iter_input = QLineEdit("50")
        self.sidebar.addWidget(self.iter_input)

        # **ACO-Specific Inputs**
        self.alpha_label = QLabel("Alpha (ACO):")
        self.sidebar.addWidget(self.alpha_label)
        self.alpha_input = QLineEdit("1.0")
        self.sidebar.addWidget(self.alpha_input)

        self.beta_label = QLabel("Beta (ACO):")
        self.sidebar.addWidget(self.beta_label)
        self.beta_input = QLineEdit("2.0")
        self.sidebar.addWidget(self.beta_input)

        self.pheromone_label = QLabel("Initial Pheromone (ACO):")
        self.sidebar.addWidget(self.pheromone_label)
        self.pheromone_input = QLineEdit("1.0")
        self.sidebar.addWidget(self.pheromone_input)

        self.evap_label = QLabel("Evaporation Rate (ACO):")
        self.sidebar.addWidget(self.evap_label)
        self.evap_input = QLineEdit("0.5")
        self.sidebar.addWidget(self.evap_input)

        self.ants_label = QLabel("Number of Ants (ACO):")
        self.sidebar.addWidget(self.ants_label)
        self.ants_input = QLineEdit("5")
        self.sidebar.addWidget(self.ants_input)

        self.deposit_label = QLabel("Deposit Constant (ACO):")
        self.sidebar.addWidget(self.deposit_label)
        self.deposit_input = QLineEdit("100.0")
        self.sidebar.addWidget(self.deposit_input)

        # Run Button
        self.run_button = QPushButton("Run Algorithm")
        self.run_button.clicked.connect(self.run_algorithm)
        self.sidebar.addWidget(self.run_button)

        # Add Sidebar to Main Layout
        self.main_layout.addLayout(self.sidebar)

        # **Result Display Area - Center Panel**
        self.result_area = QVBoxLayout()
        self.result_area.setAlignment(Qt.AlignCenter)

        # **Matplotlib Figure Canvas**
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.result_area.addWidget(self.canvas)

        # **Result Label (Below Graph)**
        self.result_label = QLabel("Results will appear here.")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFrameShape(QFrame.Box)
        self.result_label.setStyleSheet("font-size: 14px; padding: 10px; background-color: #f0f0f0;")
        self.result_area.addWidget(self.result_label)

        # Add Result Display to Main Layout
        self.main_layout.addLayout(self.result_area)

        # Set Layout
        self.setLayout(self.main_layout)

        # Update UI to show/hide parameters correctly
        self.update_ui()

    def run_algorithm(self):
        algorithm = self.algorithm_selector.currentText()
        num_nodes = int(self.node_input.text())  # Get user-defined node count
        max_iterations = int(self.iter_input.text())  # Get user-defined iterations

        cost_matrix = np.random.rand(num_nodes, num_nodes) * 100
        cost_matrix = (cost_matrix + cost_matrix.T) / 2

        if "ACO" in algorithm:
            alpha = float(self.alpha_input.text())
            beta = float(self.beta_input.text())
            initial_pheromone = float(self.pheromone_input.text())
            evaporation_rate = float(self.evap_input.text())
            num_ants = int(self.ants_input.text())
            deposit_constant = float(self.deposit_input.text())

            # Generate random cost matrix
            cost_matrix = np.random.rand(num_nodes, num_nodes) * 100

            # Run ACO
            best_solution, best_cost = ant_colony_optimization(
                cost_matrix, alpha, beta, initial_pheromone, evaporation_rate, num_ants, deposit_constant, max_iterations
            )

            # Generate random node positions
            nodes = np.random.rand(num_nodes, 2) * 100
            self.plot_tsp_solution(nodes, best_solution, f"ACO Solution - Cost: {best_cost:.2f}")
            self.result_label.setText(f"ACO Best Cost: {best_cost:.2f}")

        elif "PSO" in algorithm:
            # Run PSO with user-defined values
            cities, best_tour, best_distance = run_tsp_pso(30, max_iterations, num_nodes)

            cities = np.array(cities)

            self.plot_tsp_solution(cities, best_tour, f"PSO Solution - Distance: {best_distance:.2f}")
            self.result_label.setText(f"PSO Best Distance: {best_distance:.2f}")

        elif "DABC-FNS" in algorithm:
            # Run DABC-FNS Algorithm
            best_solution, best_cost = dabc_fns(cost_matrix, sn=30, max_cycle=max_iterations, trial_limit=100)

            # Plot and display result
            cities = np.random.rand(num_nodes, 2) * 100
            self.plot_tsp_solution(cities, best_solution, title=f"DABC-FNS Solution - Cost: {best_cost:.2f}")
            self.result_label.setText(f"DABC-FNS Best Cost: {best_cost:.2f}")

    def plot_tsp_solution(self, nodes, tour, title):
        """Plot the TSP solution inside the window using Matplotlib."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Get ordered coordinates for the tour
        tour_nodes = nodes[tour + [tour[0]]]  # Return to starting node

        ax.plot(tour_nodes[:, 0], tour_nodes[:, 1], 'bo-', markersize=8, label="Tour")
        ax.set_title(title)
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")

        for i, (x, y) in enumerate(nodes):
            ax.text(x, y, str(i), fontsize=10, ha="right")

        ax.legend()
        self.canvas.draw()

    def update_ui(self):
        """Show ACO-specific parameters when ACO is selected, hide otherwise."""
        algorithm = self.algorithm_selector.currentText()

        aco_fields = [
            self.alpha_label, self.alpha_input,
            self.beta_label, self.beta_input,
            self.pheromone_label, self.pheromone_input,
            self.evap_label, self.evap_input,
            self.ants_label, self.ants_input,
            self.deposit_label, self.deposit_input
        ]

        if "ACO" in algorithm:
            for field in aco_fields:
                field.show()
        else:  # Hide ACO parameters for PSO
            for field in aco_fields:
                field.hide()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TSPApp()
    window.show()
    sys.exit(app.exec())
