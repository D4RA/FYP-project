# Bio/src/home.py

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

class HomeScreen(QWidget):
    def __init__(self, on_start_callback=None):
        super().__init__()
        self.setWindowTitle("Welcome to TSP Solver")
        self.setGeometry(200, 100, 600, 400)

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        title = QLabel("Travelling Salesman Problem Visual Solver")
        title.setFont(QFont("Arial", 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)

        description = QLabel(
            "Explore and compare bio-inspired optimization algorithms for solving\n"
            "the Travelling Salesman Problem:\n\n"
            "- Ant Colony Optimization (ACO)\n"
            "- Particle Swarm Optimization (PSO)\n"
            "- Genetic Algorithm (GA)\n"
            "- Genetic Bee Colony (GBC)\n\n"
            "Customize parameters, visualize results, and learn how each algorithm works!"
        )
        description.setFont(QFont("Arial", 12))
        description.setAlignment(Qt.AlignCenter)
        description.setWordWrap(True)

        start_button = QPushButton("Start Application")
        start_button.setFont(QFont("Arial", 14))

        if on_start_callback is not None:
            start_button.clicked.connect(on_start_callback)

        layout.addWidget(title)
        layout.addWidget(description)
        layout.addWidget(start_button)

        self.setLayout(layout)
