# Bio/src/about.py

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QPushButton
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QRect

class AboutPage(QWidget):
    def __init__(self, return_callback, prev_geometry=None, was_maximized=False):
        super().__init__()
        self.setWindowTitle("About This Application")
        self.setGeometry(prev_geometry if prev_geometry else QRect(200, 100, 800, 600))

        if was_maximized:
            self.showMaximized()

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)

        # 📝 Rich text HTML description
        info = QTextEdit()
        info.setReadOnly(True)
        info.setFont(QFont("Segoe UI", 11))
        info.setHtml(self.get_about_html())
        layout.addWidget(info)

        # 🔙 Back button
        back_btn = QPushButton("Back to Main App")
        back_btn.setFont(QFont("Segoe UI", 11))
        back_btn.clicked.connect(return_callback)
        layout.addWidget(back_btn)

        self.setLayout(layout)

    def get_about_html(self):
        return """
        <h2>🧭 Travelling Salesman Problem (TSP)</h2>
        <p>
            The <strong>Travelling Salesman Problem</strong> (TSP) involves finding the shortest path
            that visits each city exactly once and returns to the starting point.
            It is known for its computational complexity and is considered <em>NP-hard</em>.
        </p>

        <hr>
        <h2>🐜 Ant Colony Optimization (ACO)</h2>
        <p>Simulates the pheromone-laying and path-following behavior of real ants.</p>
        <ul>
            <li><strong>Alpha (α):</strong> Weight of pheromone trails.</li>
            <li><strong>Beta (β):</strong> Weight of heuristic information (inverse of distance).</li>
            <li><strong>Evaporation Rate:</strong> Rate at which pheromones evaporate over time.</li>
            <li><strong>Number of Ants:</strong> Agents that construct solutions each iteration.</li>
            <li><strong>Deposit Constant:</strong> Amount of pheromone laid by each ant.</li>
        </ul>

        <hr>
        <h2>🕊️ Particle Swarm Optimization (PSO)</h2>
        <p>Inspired by social behaviors of birds and fish. Each solution is a "particle" moving through space.</p>
        <ul>
            <li><strong>Inertia Weight (w):</strong> Momentum from previous movement.</li>
            <li><strong>c1:</strong> Cognitive attraction to particle’s best known position.</li>
            <li><strong>c2:</strong> Social attraction to neighborhood/global best.</li>
            <li><strong>Topologies:</strong> Star (global), Ring (local), Wheel (hub-based).</li>
        </ul>

        <hr>
        <h2>🧬 Genetic Algorithm (GA)</h2>
        <p>Inspired by natural selection. Uses survival of the fittest through crossover and mutation.</p>
        <ul>
            <li><strong>Population Size:</strong> Number of candidate solutions per generation.</li>
            <li><strong>Generations:</strong> How many times the population evolves.</li>
            <li><strong>Mutation Rate:</strong> Likelihood of small random changes in individuals.</li>
            <li><strong>Crossover:</strong> Techniques like OX1, PMX, and CX to mix parent solutions.</li>
        </ul>

        <hr>
        <h2>🐝 Genetic Bee Colony (GBC)</h2>
        <p>Mimics the way bees explore and share food source locations.</p>
        <ul>
            <li><strong>Swarm Size:</strong> Number of bees (solutions).</li>
            <li><strong>Max Cycles:</strong> Iterations of exploration.</li>
            <li><strong>Trial Limit:</strong> When a bee gives up on a poor solution and tries a new one.</li>
        </ul>

        <hr>
        <p style='font-size: 12px; color: #555; font-style: italic;'>
            Visualize, experiment, and compare how nature-inspired algorithms solve complex problems.
        </p>
        """
