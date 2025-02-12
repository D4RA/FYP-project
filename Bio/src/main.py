import sys
import random
import numpy as np
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QFormLayout, QLineEdit
from PyQt5.QtCore import QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Import ACO functions from utils.py
from Utils import ant_colony_optimization, generate_random_nodes


class ACOThread(QThread):
    update_signal = pyqtSignal(object, object, float)

    def __init__(self, num_nodes, alpha, beta, InitialPheremone, evap_rate, m, constant, I_max):
        super().__init__()
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.beta = beta
        self.InitialPheremone = InitialPheremone
        self.evap_rate = evap_rate
        self.m = m
        self.constant = constant
        self.I_max = I_max

    def run(self):
        # Generate nodes and cost matrix
        nodes = generate_random_nodes(self.num_nodes)
        cost_matrix = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                cost_matrix[i][j] = np.linalg.norm(np.array(nodes[i]) - np.array(nodes[j]))

        # Run ACO algorithm
        best_solution, best_cost = ant_colony_optimization(
            cost_matrix, self.alpha, self.beta, self.InitialPheremone, self.evap_rate, self.m, self.constant, self.I_max
        )

        # Emit results for UI update
        self.update_signal.emit(nodes, best_solution, best_cost)


class ACOApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Ant Colony Optimization - TSP Solver")
        self.setGeometry(100, 100, 800, 600)

        # Main layout
        self.layout = QVBoxLayout()

        # Input form layout
        self.form_layout = QFormLayout()
        self.num_nodes_input = QLineEdit("5")
        self.alpha_input = QLineEdit("1.0")
        self.beta_input = QLineEdit("2.0")
        self.pheromone_input = QLineEdit("1.0")
        self.evap_rate_input = QLineEdit("0.5")
        self.ants_input = QLineEdit("5")
        self.deposit_input = QLineEdit("100.0")
        self.iterations_input = QLineEdit("100")

        self.form_layout.addRow("Number of Nodes:", self.num_nodes_input)
        self.form_layout.addRow("Alpha:", self.alpha_input)
        self.form_layout.addRow("Beta:", self.beta_input)
        self.form_layout.addRow("Initial Pheromone:", self.pheromone_input)
        self.form_layout.addRow("Evaporation Rate:", self.evap_rate_input)
        self.form_layout.addRow("Number of Ants:", self.ants_input)
        self.form_layout.addRow("Deposit Constant:", self.deposit_input)
        self.form_layout.addRow("Max Iterations:", self.iterations_input)

        self.layout.addLayout(self.form_layout)

        # Run button
        self.run_button = QPushButton("Run ACO")
        self.run_button.clicked.connect(self.run_aco)
        self.layout.addWidget(self.run_button)

        # Matplotlib figure for plotting
        self.figure, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        # Label to show results
        self.result_label = QLabel("")
        self.layout.addWidget(self.result_label)

        # Central widget
        self.central_widget = QWidget()
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)

    def run_aco(self):
        try:
            # Get parameters from UI
            num_nodes = int(self.num_nodes_input.text())
            alpha = float(self.alpha_input.text())
            beta = float(self.beta_input.text())
            InitialPheremone = float(self.pheromone_input.text())
            evap_rate = float(self.evap_rate_input.text())
            m = int(self.ants_input.text())
            constant = float(self.deposit_input.text())
            I_max = int(self.iterations_input.text())

            # Run ACO in a separate thread
            self.aco_thread = ACOThread(num_nodes, alpha, beta, InitialPheremone, evap_rate, m, constant, I_max)
            self.aco_thread.update_signal.connect(self.update_plot)
            self.aco_thread.start()

        except ValueError:
            self.result_label.setText("Error: Please enter valid numeric values.")

    def update_plot(self, nodes, solution, cost):
        self.ax.clear()
        x = [nodes[i][0] for i in solution]
        y = [nodes[i][1] for i in solution]

        self.ax.scatter(x, y, c="red", s=50, label="Nodes")
        for i, (x_coord, y_coord) in enumerate(nodes):
            self.ax.text(x_coord, y_coord, f"{i}", fontsize=10, ha="right")

        self.ax.plot(x, y, c="blue", linestyle="--", label="Path")

        self.ax.set_title(f"TSP Solution - Cost: {cost:.2f}")
        self.ax.set_xlabel("X-coordinate")
        self.ax.set_ylabel("Y-coordinate")
        self.ax.legend()
        self.ax.grid(True)

        self.canvas.draw()

        # Update result label
        self.result_label.setText(f"Best Cost: {cost:.2f}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ACOApp()
    window.show()
    sys.exit(app.exec_())
