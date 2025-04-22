from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QScrollArea, QLabel, QFrame
from PyQt5.QtCore import Qt, QPropertyAnimation
from PyQt5.QtGui import QFont


class CollapsibleSection(QWidget):
    def __init__(self, title: str, content: str, parent=None):
        super().__init__(parent)

        self.toggle_button = QPushButton(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)
        self.toggle_button.setStyleSheet("""
            QPushButton {
                text-align: left;
                font-weight: bold;
                font-size: 14px;
                background-color: #e6e6e6;
                padding: 6px;
            }
            QPushButton:checked {
                background-color: #d0d0d0;
            }
        """)

        self.content_area = QFrame()
        self.content_area.setMinimumHeight(0)
        self.content_area.setMaximumHeight(0)
        self.content_area.setStyleSheet("background-color: #fafafa; border-left: 2px solid #ccc; padding: 10px;")
        self.content_area.setFrameShape(QFrame.StyledPanel)

        self.content_layout = QVBoxLayout()
        self.content_label = QLabel(content)
        self.content_label.setWordWrap(True)
        self.content_label.setAlignment(Qt.AlignTop)
        self.content_label.setFont(QFont("Segoe UI", 11))
        self.content_layout.addWidget(self.content_label)
        self.content_area.setLayout(self.content_layout)

        self.toggle_animation = QPropertyAnimation(self.content_area, b"maximumHeight")
        self.toggle_animation.setDuration(200)
        self.toggle_animation.setStartValue(0)
        self.toggle_animation.setEndValue(0)

        self.toggle_button.toggled.connect(self.on_toggle)

        layout = QVBoxLayout(self)
        layout.addWidget(self.toggle_button)
        layout.addWidget(self.content_area)
        layout.setContentsMargins(0, 0, 0, 0)

    def on_toggle(self, checked):
        content_height = self.content_area.layout().sizeHint().height()
        self.toggle_animation.setStartValue(0 if checked else content_height)
        self.toggle_animation.setEndValue(content_height if checked else 0)
        self.toggle_animation.start()


class AboutPage(QWidget):
    def __init__(self, return_callback, prev_geometry=None, was_maximized=False):
        super().__init__()
        self.setWindowTitle("About This Application")

        if was_maximized:
            self.showMaximized()
        elif prev_geometry:
            self.setGeometry(prev_geometry)

        layout = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        container = QWidget()
        container_layout = QVBoxLayout(container)

        # In-depth descriptions
        sections = [
            (
                " Travelling Salesman Problem (TSP)",
                "The TSP is one of the most famous and widely studied combinatorial optimization problems in computer science and operations research. The problem is simple to state but computationally complex:\n\n"
                "Given a set of cities and the distances between each pair of cities, find the shortest possible route that visits every city exactly once and returns to the starting city.\n\n"
                "TSP is classified as an NP-hard problem, meaning there is no known algorithm that can solve all instances of TSP efficiently (i.e., in polynomial time).\n\n"
                "TSP has practical applications in:\n"
                "- Logistics (routing delivery trucks, mail carriers)\n"
                "- Manufacturing (minimizing tool path length)\n"
                "- Circuit design (optimizing wire lengths in circuit boards)\n"
                "- DNA sequencing and genome assembly\n\n"
                "Challenges:\n"
                "- The number of possible solutions grows factorially with the number of cities (n!), making brute-force search infeasible for large n.\n"
                "- It requires algorithms that balance exploration (trying new paths) and exploitation (refining promising paths).\n\n"
                "Because of its complexity and real-world relevance, TSP is a benchmark problem for testing heuristic and bio-inspired algorithms such as ACO, PSO, GA, and GBC."

            ),
            (
                " Ant Colony Optimization (ACO)",
                "ACO is inspired by the foraging behavior of real ants that lay down pheromones to mark favorable paths to food sources. In the algorithm, each artificial 'ant' constructs a solution by moving from city to city, probabilistically influenced by artificial pheromone levels and heuristic desirability (like distance).\n\n"
                "The algorithm iteratively improves paths based on collective learning, where better solutions receive stronger pheromone reinforcement. Over time, the pheromone trails guide the population toward optimal or near-optimal paths.\n\n"
                "Key parameters:\n"
                "- Alpha (α): Determines the relative importance of the pheromone trail. Higher values make ants follow existing paths more closely.\n"
                "- Beta (β): Determines the importance of heuristic information (typically the inverse of distance). Higher values lead to greedier behavior toward shorter paths.\n"
                "- Evaporation Rate: Controls the rate at which pheromone trails decay. Higher values lead to faster decay, encouraging exploration and preventing early convergence.\n"
                "- Number of Ants: Defines the number of agents constructing solutions per iteration. More ants can improve diversity but increase computational cost.\n"
                "- Deposit Constant: Governs how much pheromone is deposited based on solution quality. Larger values amplify good paths and accelerate convergence.\n\n"
                "ACO is particularly effective for discrete optimization problems like the TSP, where solution quality improves through iterative reinforcement of favorable decision patterns. It balances exploration (via randomness and evaporation) with exploitation (via pheromone amplification)."

            ),
            (
                " Particle Swarm Optimization (PSO)",
                "PSO is inspired by the social behavior of birds flocking or fish schooling. In PSO, each 'particle' represents a candidate solution (e.g., a tour in TSP), and the entire population of particles collectively explores the solution space.\n\n"
                "Each particle adjusts its position based on:\n"
                "- Its own personal best solution (cognitive memory).\n"
                "- The best known solution found by the swarm (social influence).\n"
                "- Its current velocity (momentum).\n\n"
                "Key parameters:\n"
                "- Inertia Weight (w): Controls the particle's momentum. High values encourage exploration; low values promote local search.\n"
                "- c1 (Cognitive Coefficient): Strength of attraction to a particle's own best solution.\n"
                "- c2 (Social Coefficient): Strength of attraction to the global best or neighborhood best solution.\n"
                "- Max Velocity (v_max): Restricts how much a particle can move in a single iteration. Helps avoid chaotic behavior and fine-tunes convergence.\n"
                "- Topology: Defines how particles influence each other:\n"
                "   - Star: All particles are connected to a central global best.\n"
                "   - Ring: Particles only communicate with neighbors.\n"
                "   - Wheel: One designated hub guides the rest.\n\n"
                "PSO balances individual exploration with collective learning, making it highly adaptable for continuous and combinatorial optimization problems like TSP when modified appropriately."

            ),
            (
                " Genetic Algorithm (GA)",
                "GA mimics the principles of biological evolution and natural selection. A population of candidate solutions (chromosomes) evolves over generations, using operators inspired by genetic processes:\n\n"
                "Key components:\n"
                "- Selection: Determines which individuals are chosen to breed based on their fitness. Tournament selection is a common method.\n"
                "- Crossover: Combines parts of two parents to create offspring. Techniques include:\n"
                "   - OX1 (Ordered Crossover): Preserves the relative order of elements.\n"
                "   - PMX (Partially Mapped Crossover): Preserves value mapping from parents using position mapping.\n"
                "   - CX (Cycle Crossover): Preserves index positions by cycling through the parents.\n"
                "- Mutation: Introduces random small changes (e.g., swapping two cities) to promote diversity.\n\n"
                "Key parameters:\n"
                "- Population Size: The number of individuals in each generation. Larger populations increase diversity but require more computation.\n"
                "- Generations: Number of iterations the population evolves.\n"
                "- Mutation Rate: Probability of mutation occurring per offspring. Prevents premature convergence and helps escape local optima.\n\n"
                "GA is robust and flexible, capable of finding high-quality solutions even in complex landscapes. It’s especially effective when paired with diverse crossover and mutation strategies tailored to the problem domain."

            ),
            (
                " Genetic Bee Colony (GBC)",
                "GBC is inspired by the foraging behavior of honey bees, combining exploration and exploitation through the interaction of employed bees, onlooker bees, and scout bees.\n\n"
                "- Employed bees exploit known food sources (solutions) and share information with onlookers.\n"
                "- Onlooker bees probabilistically choose promising sources based on shared quality (fitness).\n"
                "- Scout bees explore new solutions when current ones are exhausted (i.e., trial limit is reached).\n\n"
                "Key parameters:\n"
                "- Swarm Size: Total number of bees, including employed and onlooker bees. Affects the balance between exploration and exploitation.\n"
                "- Max Cycles: The number of iterations (cycles) over which bees refine solutions.\n"
                "- Trial Limit: If a bee does not improve a solution within this number of attempts, it abandons the solution and becomes a scout.\n\n"
                "GBC is highly adaptive and self-regulating. It avoids stagnation by introducing new solutions when progress stalls and leverages a combination of collective decision-making and individual innovation to converge on high-quality solutions."

            ),
        ]

        # Add collapsible trays to the layout
        for title, content in sections:
            container_layout.addWidget(CollapsibleSection(title, content))

        scroll.setWidget(container)
        layout.addWidget(scroll)

        # Back button
        back_btn = QPushButton("Back to Main App")
        back_btn.setFont(QFont("Segoe UI", 11))
        back_btn.clicked.connect(return_callback)
        layout.addWidget(back_btn)
