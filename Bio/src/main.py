import sys

from PyQt5.QtCore import QTimer
from PyQt5.QtSensors import QCompass
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QLineEdit, \
    QPushButton, QTextEdit, QWidget, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from Utils import generate_random_nodes, calculate_distance_matrix, plot_tsp_solution
from ACo2 import ant_colony_optimization  # Replace with your ACO module
#from psoII import particle_swarm_optimization
import numpy as np

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        top_layout = QHBoxLayout()
        bottom_layout = QVBoxLayout()

        self.algorithmLabel = QLabel("select Algorithm:")
        self.algortihmDrpdown = QComboBox()
        self.algortihmDrpdown.addItems(["Ant Colony OPtimization", "Particle Swarm Optimization"])

        self.paramaterLabel = QLabel("Number of nodes")
        self.paramaterInput = QLineEdit()
        self.paramaterInput.setPlaceholderText("enter number of nodes")

        self.runButton = QPushButton("Run Algorithm")
        self.runButton.clicked.connect(self.runAlgorithm)

        self.generateButton = QPushButton("Generate Random nodes")
        self.generateButton.clicked.connect(self.generateNodes)

        self.resultArea = QTextEdit()
        self.resultArea.setReadOnly(True)

        self.figure, self.ax = plt.subplots(figsize=(10, 10))
        self.canvas = FigureCanvas(self.figure)

        # Add Widgets to Layouts
        top_layout.addWidget(self.algorithmLabel)
        top_layout.addWidget(self.algortihmDrpdown)
        top_layout.addWidget(self.paramaterLabel)
        top_layout.addWidget(self.paramaterInput)
        top_layout.addWidget(self.generateButton)
        top_layout.addWidget(self.runButton)

        bottom_layout.addWidget(self.canvas)
        bottom_layout.addWidget(self.resultArea)

        layout.addLayout(top_layout)
        layout.addLayout(bottom_layout)

        # Set Central Widget
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Data
        self.nodes = []
        self.distance_matrix = None

    def runAlgorithm(self):
        if not self.nodes:
            self.resultArea.append("Please generate nodes first.")
            return

        selected_algo = self.algo_dropdown.currentText()
        if selected_algo == "Ant Colony Optimization (ACO)":
            self.run_aco()
        elif selected_algo == "Particle Swarm Optimization (PSO)":
            self.run_pso()

    def generateNodes(self):
        try:
            num_nodes = int(self.param_input.text())  # Get number of nodes
            if num_nodes <= 0:
                QMessageBox.warning(self, "Input Error", "Number of nodes must be greater than zero.")
                return

            # Generate random (x, y) coordinates
            self.nodes = np.random.rand(num_nodes, 2) * 100

            # Debugging: Print nodes to console
            print("Generated Nodes:", self.nodes)

            # Update the plot
            self.update_plot()
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please enter a valid number.")

    def update_plot(self):
        if not hasattr(self, "nodes") or len(self.nodes) == 0:
            return  # No nodes to plot

        self.ax.clear()  # Clear previous plot
        self.ax.scatter(self.nodes[:, 0], self.nodes[:, 1], c="red", marker="o", label="Nodes")

        self.ax.set_title("Randomly Generated Nodes")
        self.ax.set_xlabel("X Coordinate")
        self.ax.set_ylabel("Y Coordinate")
        self.ax.legend()

        # Ensure UI update happens in the main thread
        QTimer.singleShot(0, self.canvas.draw)




if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
