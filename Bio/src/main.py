import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QComboBox, QLabel, QLineEdit, QFrame, QSizePolicy
)
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from algorithms.ACO import ant_colony_optimization
from algorithms.PSO import run_tsp_pso
from algorithms.GBC import dabc_fns
from plotting.utils import plot_tsp_solution

class TSPApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TSP Solver")
        self.setGeometry(100, 100, 800, 500)

        # Main Layout
        self.main_layout = QHBoxLayout(self)

        # Sidebar Layout
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

        # ACO-Specific Inputs
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

        # PSO Specific Inputs
        self.pso_particles_label = QLabel("Number of Particles (PSO):")
        self.sidebar.addWidget(self.pso_particles_label)
        self.pso_particles_input = QLineEdit("30")
        self.sidebar.addWidget(self.pso_particles_input)

        self.pso_w_label = QLabel("Inertia Weight (w - PSO):")
        self.sidebar.addWidget(self.pso_w_label)
        self.pso_w_input = QLineEdit("0.7")
        self.sidebar.addWidget(self.pso_w_input)

        self.pso_c1_label = QLabel("Cognitive Coefficient (c1 - PSO):")
        self.sidebar.addWidget(self.pso_c1_label)
        self.pso_c1_input = QLineEdit("1.5")
        self.sidebar.addWidget(self.pso_c1_input)

        self.pso_c2_label = QLabel("Social Coefficient (c2 - PSO):")
        self.sidebar.addWidget(self.pso_c2_label)
        self.pso_c2_input = QLineEdit("2.0")
        self.sidebar.addWidget(self.pso_c2_input)

        self.pso_vmax_label = QLabel("Max Velocity (v_max - PSO):")
        self.sidebar.addWidget(self.pso_vmax_label)
        self.pso_vmax_input = QLineEdit("4.0")
        self.sidebar.addWidget(self.pso_vmax_input)

        # Run Button
        self.run_button = QPushButton("Run Algorithm")
        self.run_button.clicked.connect(self.run_algorithm)
        self.sidebar.addWidget(self.run_button)

        # Set Sidebar Expandable
        for widget in self.sidebar.children():
            if isinstance(widget, (QLineEdit, QComboBox)):
                widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.main_layout.addLayout(self.sidebar, 1)

        # Result Display Area
        self.result_area = QVBoxLayout()
        self.result_area.setAlignment(Qt.AlignCenter)

        # Matplotlib Figure Canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.result_area.addWidget(self.canvas, 1)

        # Result Label
        self.result_label = QLabel("Results will appear here.")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFrameShape(QFrame.Box)
        self.result_label.setStyleSheet("font-size: 14px; padding: 10px; background-color: #f0f0f0;")
        self.result_area.addWidget(self.result_label)

        self.main_layout.addLayout(self.result_area, 3)

        # Set Layout
        self.setLayout(self.main_layout)

        self.update_ui()

    def run_algorithm(self):
        algorithm = self.algorithm_selector.currentText()
        num_nodes = int(self.node_input.text())
        max_iterations = int(self.iter_input.text())

        cost_matrix = np.random.rand(num_nodes, num_nodes) * 100
        cost_matrix = (cost_matrix + cost_matrix.T) / 2

        ax = self.figure.clear()
        ax = self.figure.add_subplot(111)

        if "ACO" in algorithm:
            alpha = float(self.alpha_input.text())
            beta = float(self.beta_input.text())
            initial_pheromone = float(self.pheromone_input.text())
            evaporation_rate = float(self.evap_input.text())
            num_ants = int(self.ants_input.text())
            deposit_constant = float(self.deposit_input.text())

            best_solution, best_cost = ant_colony_optimization(
                cost_matrix, alpha, beta, initial_pheromone, evaporation_rate, num_ants, deposit_constant, max_iterations
            )

            nodes = np.random.rand(num_nodes, 2) * 100
            plot_tsp_solution(ax, nodes, best_solution, f"ACO Solution - Cost: {best_cost:.2f}")

            self.result_label.setText(f"ACO Best Cost: {best_cost:.2f}")

        elif "PSO" in algorithm:
            num_particles = int(self.pso_particles_input.text())
            w = float(self.pso_w_input.text())
            c1 = float(self.pso_c1_input.text())
            c2 = float(self.pso_c2_input.text())
            v_max = float(self.pso_vmax_input.text())

            cities, best_tour, best_distance = run_tsp_pso(num_particles, max_iterations, num_nodes)
            cities = np.array(cities)

            plot_tsp_solution(ax, cities, best_tour, f"PSO Solution - Distance: {best_distance:.2f}")

            self.result_label.setText(f"PSO Best Distance: {best_distance:.2f}")

        elif "DABC-FNS" in algorithm:
            best_solution, best_cost = dabc_fns(cost_matrix, sn=30, max_cycle=max_iterations, trial_limit=100)

            cities = np.random.rand(num_nodes, 2) * 100
            plot_tsp_solution(ax, cities, best_solution, f"DABC-FNS Solution - Cost: {best_cost:.2f}")

            self.result_label.setText(f"DABC-FNS Best Cost: {best_cost:.2f}")

        self.canvas.draw()

    def update_ui(self):
        algorithm = self.algorithm_selector.currentText()

        aco_fields = [self.alpha_label, self.alpha_input, self.beta_label, self.beta_input,
                      self.pheromone_label, self.pheromone_input, self.evap_label, self.evap_input,
                      self.ants_label, self.ants_input, self.deposit_label, self.deposit_input]
        pso_fields = [self.pso_particles_label, self.pso_particles_input, self.pso_w_label, self.pso_w_input,
                      self.pso_c1_label, self.pso_c1_input, self.pso_c2_label, self.pso_c2_input,
                      self.pso_vmax_label, self.pso_vmax_input]

        for field in aco_fields + pso_fields:
            field.setVisible("ACO" in algorithm if field in aco_fields else "PSO" in algorithm)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TSPApp()
    window.show()
    sys.exit(app.exec())
