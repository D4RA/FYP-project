import sys
import numpy as np
from PyQt5.QtGui import QCursor, QFont
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QComboBox, QLabel, QLineEdit, QFrame, QSizePolicy, QTextEdit, QCheckBox
)
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from Bio.src.about import AboutPage
from algorithms.ACO import ant_colony_optimization, cost_matrix_to_coords
from algorithms.PSO import run_tsp_pso
from algorithms.GBC import dabc_fns
from plotting.utils import plot_tsp_solution, apply_mandatory_bridge,create_cost_matrix
from algorithms.GA import run_tsp_ga
from clicks import ClickableLabel
from descriptions import descriptions



class TSPApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TSP Solver")
        self.setGeometry(100, 100, 1200, 700)
        self.setMinimumSize(1000, 600)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

        # Main Layout
        self.main_layout = QHBoxLayout(self)
        self.main_layout.setContentsMargins(10, 10, 10, 10)

        # Sidebar Layout
        self.sidebar = QVBoxLayout()
        self.sidebar.setAlignment(Qt.AlignTop)

        # Info Panel (must be defined before being added to layout)
        self.info_panel = QTextEdit()
        self.info_panel.setReadOnly(True)
        self.info_panel.setFrameShape(QFrame.Box)
        self.info_panel.setStyleSheet("font-size: 14px; padding: 10px; background-color: #f9f9f9;")
        self.info_panel.setFixedWidth(250)

        # Result Area Layout
        self.result_area = QVBoxLayout()
        self.result_area.setAlignment(Qt.AlignCenter)

        # About Button
        self.about_button = QPushButton("About")
        self.about_button.clicked.connect(self.show_about_page)
        self.sidebar.addWidget(self.about_button)

        # Algorithm Selector
        self.algorithm_selector = QComboBox()
        self.algorithm_selector.addItems([
            "Ant Colony Optimization (ACO)", "Particle Swarm Optimization (PSO)",
            "DABC_FNS(GBC)", "Genetic Algorithm (GA)"
        ])
        self.algorithm_selector.currentIndexChanged.connect(self.update_ui)
        self.sidebar.addWidget(QLabel("Select Algorithm:"))
        self.sidebar.addWidget(self.algorithm_selector)

        # Topology for PSO
        self.topology_label = QLabel("PSO Topology:")
        self.sidebar.addWidget(self.topology_label)
        self.topology_selector = QComboBox()
        self.topology_selector.addItems(["Star", "Ring", "Wheel"])
        self.sidebar.addWidget(self.topology_selector)

        self.info_panel = QTextEdit()
        self.info_panel.setReadOnly(True)
        self.info_panel.setFrameShape(QFrame.Box)
        self.info_panel.setStyleSheet("font-size: 14px; padding: 10px; background-color: #f9f9f9;")
        self.info_panel.setFixedWidth(250)  # Adjust width of the text panel
        self.result_area.addWidget(self.info_panel)  # Add to UI

        # Common Inputs
        self.node_label = ClickableLabel("Number of Nodes (Cities):", self.show_description)
        self.sidebar.addWidget(self.node_label)
        self.node_input = QLineEdit("10")
        self.sidebar.addWidget(self.node_input)

        self.iter_label = ClickableLabel("Max Iterations:", self.show_description)
        self.sidebar.addWidget(self.iter_label)
        self.iter_input = QLineEdit("50")
        self.sidebar.addWidget(self.iter_input)

        # ACO-Specific Inputs
        self.alpha_label = ClickableLabel("Alpha (ACO):", self.show_description)
        self.sidebar.addWidget(self.alpha_label)
        self.alpha_input = QLineEdit("1.0")
        self.sidebar.addWidget(self.alpha_input)

        self.beta_label = ClickableLabel("Beta (ACO):", self.show_description)
        self.sidebar.addWidget(self.beta_label)
        self.beta_input = QLineEdit("2.0")
        self.sidebar.addWidget(self.beta_input)

        self.pheromone_label = ClickableLabel("Initial Pheromone (ACO):", self.show_description)
        self.sidebar.addWidget(self.pheromone_label)
        self.pheromone_input = QLineEdit("1.0")
        self.sidebar.addWidget(self.pheromone_input)

        self.evap_label = ClickableLabel("Evaporation Rate (ACO):", self.show_description)
        self.sidebar.addWidget(self.evap_label)
        self.evap_input = QLineEdit("0.5")
        self.sidebar.addWidget(self.evap_input)

        self.ants_label = ClickableLabel("Number of Ants (ACO):", self.show_description)
        self.sidebar.addWidget(self.ants_label)
        self.ants_input = QLineEdit("5")
        self.sidebar.addWidget(self.ants_input)

        self.deposit_label = ClickableLabel("Deposit Constant (ACO):", self.show_description)
        self.sidebar.addWidget(self.deposit_label)
        self.deposit_input = QLineEdit("100.0")
        self.sidebar.addWidget(self.deposit_input)

        # PSO Specific Inputs
        self.pso_particles_label = ClickableLabel("Number of Particles (PSO):", self.show_description)
        self.sidebar.addWidget(self.pso_particles_label)
        self.pso_particles_input = QLineEdit("30")
        self.sidebar.addWidget(self.pso_particles_input)

        self.pso_w_label = ClickableLabel("Inertia Weight (w - PSO):", self.show_description)
        self.sidebar.addWidget(self.pso_w_label)
        self.pso_w_input = QLineEdit("0.7")
        self.sidebar.addWidget(self.pso_w_input)

        self.pso_c1_label = ClickableLabel("Cognitive Coefficient (c1 - PSO):", self.show_description)
        self.sidebar.addWidget(self.pso_c1_label)
        self.pso_c1_input = QLineEdit("1.5")
        self.sidebar.addWidget(self.pso_c1_input)

        self.pso_c2_label = ClickableLabel("Social Coefficient (c2 - PSO):", self.show_description)
        self.sidebar.addWidget(self.pso_c2_label)
        self.pso_c2_input = QLineEdit("2.0")
        self.sidebar.addWidget(self.pso_c2_input)

        self.pso_vmax_label = ClickableLabel("Max Velocity (v_max - PSO):", self.show_description)
        self.sidebar.addWidget(self.pso_vmax_label)
        self.pso_vmax_input = QLineEdit("4.0")
        self.sidebar.addWidget(self.pso_vmax_input)

        # GBC-Specific Inputs
        self.gbc_sn_label = ClickableLabel("Swarm Size (GBC):", self.show_description)
        self.sidebar.addWidget(self.gbc_sn_label)
        self.gbc_sn_input = QLineEdit("10")  # Default value
        self.sidebar.addWidget(self.gbc_sn_input)

        self.gbc_max_cycle_label = ClickableLabel("Max Cycles (GBC):", self.show_description)
        self.sidebar.addWidget(self.gbc_max_cycle_label)
        self.gbc_max_cycle_input = QLineEdit("5000")
        self.sidebar.addWidget(self.gbc_max_cycle_input)

        self.gbc_trial_limit_label = ClickableLabel("Trial Limit (GBC):", self.show_description)
        self.sidebar.addWidget(self.gbc_trial_limit_label)
        self.gbc_trial_limit_input = QLineEdit("100")
        self.sidebar.addWidget(self.gbc_trial_limit_input)

        # GA Specific Inputs
        self.ga_population_label = ClickableLabel("Population Size (GA):", self.show_description)
        self.sidebar.addWidget(self.ga_population_label)
        self.ga_population_input = QLineEdit("50")
        self.sidebar.addWidget(self.ga_population_input)

        self.ga_generations_label = ClickableLabel("Generations (GA):", self.show_description)
        self.sidebar.addWidget(self.ga_generations_label)
        self.ga_generations_input = QLineEdit("100")
        self.sidebar.addWidget(self.ga_generations_input)

        self.ga_mutation_label = ClickableLabel("Mutation Rate (GA):", self.show_description)
        self.sidebar.addWidget(self.ga_mutation_label)
        self.ga_mutation_input = QLineEdit("0.1")
        self.sidebar.addWidget(self.ga_mutation_input)

        # Crossover Type Selection
        self.ga_crossover_label = QLabel("Crossover Type (GA):")
        self.sidebar.addWidget(self.ga_crossover_label)

        self.ga_crossover_selector = QComboBox()
        self.ga_crossover_selector.addItems(
            ["Order Crossover (OX1)", "Cycle Crossover (CX)", "Partially Mapped Crossover (PMX)"])
        self.sidebar.addWidget(self.ga_crossover_selector)

        # -- Bridge Controls --
        self.bridge_checkbox = QCheckBox("Enable Bridge Between Cities")
        self.sidebar.addWidget(self.bridge_checkbox)

        self.bridge_city_a_input = QLineEdit("0")
        self.bridge_city_b_input = QLineEdit("1")

        self.sidebar.addWidget(QLabel("Bridge: City A Index"))
        self.sidebar.addWidget(self.bridge_city_a_input)

        self.sidebar.addWidget(QLabel("Bridge: City B Index"))
        self.sidebar.addWidget(self.bridge_city_b_input)
        self.bridge_checkbox.stateChanged.connect(self.toggle_bridge_inputs)

        # Run Button
        self.run_button = QPushButton("Run Algorithm")
        self.run_button.clicked.connect(self.run_algorithm)
        self.sidebar.addWidget(self.run_button)

        # Set Sidebar Expandable
        for widget in self.sidebar.children():
            if isinstance(widget, (QLineEdit, QComboBox)):
                widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.main_layout.addLayout(self.sidebar, stretch=1)
        self.main_layout.addLayout(self.result_area, stretch=5)

        # Result Display Area
        #self.result_area = QVBoxLayout()
        self.result_area.setAlignment(Qt.AlignCenter)

        # Matplotlib Figure Canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.result_area.addWidget(self.canvas, 1)

        self.ga_figure = Figure(figsize=(3, 2))
        self.ga_canvas = FigureCanvas(self.ga_figure)
        self.result_area.addWidget(self.ga_canvas)
        self.ga_canvas.hide()

        # self.main_layout.addLayout(self.sidebar)
        # self.main_layout.addLayout(self.result_area)
        self.main_layout.addWidget(self.info_panel, stretch=1)
        self.main_layout.addLayout(self.result_area, stretch=4)

        self.result_label = QLabel("Results will appear here.")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFrameShape(QFrame.Box)
        self.result_label.setStyleSheet("font-size: 14px; padding: 10px; background-color: #f0f0f0;")
        self.result_area.addWidget(self.result_label)

        # Set Layout
        self.setLayout(self.main_layout)

        self.update_ui()

    def create_clickable_label(self, text):
        """Creates a clickable label that updates the info panel with a description."""
        label = QLabel(text)
        label.setCursor(QCursor(Qt.PointingHandCursor))
        label.setStyleSheet("color: blue; text-decoration: underline;")
        label.setFont(QFont("Arial", 10, QFont.Bold))
        label.mousePressEvent = lambda event: self.show_description(text)
        return label

        # Assign the event
        label.mousePressEvent = lambda event, lbl_text=text: self.show_description(lbl_text)

        self.sidebar.addWidget(label)
        return label  # Returning the label in case further modifications are needed

    def show_description(self, label_text):
        print(f"Clicked label: '{label_text}'")  # Debug line
        description = descriptions.get(label_text, "No description available.")
        self.info_panel.setPlainText(description)


    def toggle_bridge_inputs(self, state):
        is_checked = state == Qt.Checked
        self.bridge_city_a_input.setVisible(is_checked)
        self.bridge_city_b_input.setVisible(is_checked)

    def run_algorithm(self):
        try:
            algorithm = self.algorithm_selector.currentText()
            num_nodes = int(self.node_input.text())
            max_iterations = int(self.iter_input.text())

            # Step 1: Generate cities and create cost matrix
            cities = np.random.rand(num_nodes, 2) * 100
            cost_matrix = create_cost_matrix(cities)
            np.fill_diagonal(cost_matrix, np.inf)

            print(f"Running {algorithm} with {num_nodes} nodes and {max_iterations} iterations")

            # Apply mandatory bridge only if the user enables it
            bridge = None  # ← Define it first

            if self.bridge_checkbox.isChecked():
                try:
                    city_a_text = self.bridge_city_a_input.text().strip()
                    city_b_text = self.bridge_city_b_input.text().strip()

                    if not city_a_text or not city_b_text:
                        raise ValueError("Bridge city fields are empty")

                    city_a = int(city_a_text)
                    city_b = int(city_b_text)

                    if 0 <= city_a < num_nodes and 0 <= city_b < num_nodes and city_a != city_b:
                        cost_matrix = apply_mandatory_bridge(cost_matrix, city_a, city_b)
                        bridge = (city_a, city_b)  # ← Store the bridge tuple
                        print(f"Applied bridge between City {city_a} and City {city_b}")
                    else:
                        print(
                            f" Invalid city indices (must be between 0 and {num_nodes - 1} and not equal). Skipping bridge.")
                except Exception as e:
                    print(f" Failed to apply bridge: {e}")
            else:
                print(" Bridge not enabled.")

            # Step 3: Set up canvas
            self.figure.clear()
            ax = self.figure.add_subplot(111)

            # Store for later plotting
            self.current_cities = cities
            self.current_cost_matrix = cost_matrix
            self.current_ax = ax

        except Exception as e:
            print(f"Exception during run_algorithm: {e}")

        if "ACO" in algorithm:
            alpha = float(self.alpha_input.text())
            beta = float(self.beta_input.text())
            initial_pheromone = float(self.pheromone_input.text())
            evaporation_rate = float(self.evap_input.text())
            num_ants = int(self.ants_input.text())
            deposit_constant = float(self.deposit_input.text())
            max_iterations = int(self.iter_input.text())

            best_solution, best_cost = ant_colony_optimization(
                cost_matrix, alpha, beta, initial_pheromone, evaporation_rate,
                num_ants, deposit_constant, max_iterations
            )

            nodes = np.random.rand(num_nodes, 2) * 100
            plot_tsp_solution(self.current_ax, self.current_cities, best_solution, f"ACO (with bridge) - Cost: {best_cost:.2f}", bridge=bridge)
            self.result_label.setText(f"ACO Best Cost: {best_cost:.2f}")


        elif "PSO" in algorithm:
            num_particles = int(self.pso_particles_input.text())
            w = float(self.pso_w_input.text())
            c1 = float(self.pso_c1_input.text())
            c2 = float(self.pso_c2_input.text())
            v_max = float(self.pso_vmax_input.text())
            topology = self.topology_selector.currentText().lower()  # Get user-selected topology
            cities, best_tour, best_distance = run_tsp_pso(
                num_nodes=num_nodes,
                num_particles=num_particles,
                w=w,
                c1=c1,
                c2=c2,
                v_max=v_max,
                max_iterations=max_iterations,
                topology=topology,
                bridge=(0,1) if self.bridge_checkbox.isChecked() else None

            )
            plot_tsp_solution(ax, cities, best_tour, f"PSO ({topology.capitalize()}) - Distance: {best_distance:.2f}",bridge=bridge)
            self.result_label.setText(f"PSO Best Distance: {best_distance:.2f}")


        elif "DABC_FNS" in algorithm:
            sn = int(self.gbc_sn_input.text())  # Swarm size
            max_cycle = int(self.gbc_max_cycle_input.text())  # Maximum number of cycles
            trial_limit = int(self.gbc_trial_limit_input.text())  # Trial limit before replacement
            # Generate a random cost matrix
            cost_matrix = np.full((num_nodes, num_nodes), np.inf)  # Start with no connections

            # Example default or user-defined connections:
            # Format: (city1, city2, distance)
            connections = [(0, 1, 50), (1, 2, 60), (2, 3, 40), (3, 0, 70)]

            for a, b, d in connections:
                cost_matrix[a][b] = d
                cost_matrix[b][a] = d  # Assuming undirected graph

            # Run the GBC Algorithm with user inputs
            best_solution, best_cost = dabc_fns(cost_matrix, sn=sn, max_cycle=max_cycle, trial_limit=trial_limit)

            # Generate random city positions
            cities = np.random.rand(num_nodes, 2) * 100

            # Plot the solution
            plot_tsp_solution(ax, cities, best_solution, f"GBC Solution - Cost: {best_cost:.2f}")
            self.result_label.setText(f"GBC Best Cost: {best_cost:.2f}")

        elif "GA" in algorithm:
            population_size = int(self.ga_population_input.text())
            generations = int(self.ga_generations_input.text())
            mutation_rate = float(self.ga_mutation_input.text())
            crossover_type = self.ga_crossover_selector.currentText()
            cities, best_tour, best_distance,history = run_tsp_ga(
                num_cities=num_nodes,
                population_size=population_size,
                generations=generations,
                mutation_rate=mutation_rate,
                crossover_type=crossover_type,


            )
            plot_tsp_solution(ax, cities, best_tour, f"GA Solution - Distance: {best_distance:.2f}")
            self.ga_figure.clear()
            ga_ax = self.ga_figure.add_subplot(111)
            ga_ax.plot(history, label="Fitness over Generations", color='purple')
            ga_ax.set_title("GA Convergence")
            ga_ax.set_xlabel("Generation")
            ga_ax.set_ylabel("Best Distance")
            ga_ax.legend()
            self.ga_canvas.draw()
            self.result_label.setText(f"GA Best Distance: {best_distance:.2f}")
        self.canvas.draw()
        QApplication.processEvents()

    def update_ui(self):
        algorithm = self.algorithm_selector.currentText()

        aco_fields = [self.alpha_label, self.alpha_input, self.beta_label, self.beta_input,
                      self.pheromone_label, self.pheromone_input, self.evap_label, self.evap_input,
                      self.ants_label, self.ants_input, self.deposit_label, self.deposit_input]

        pso_fields = [self.pso_particles_label, self.pso_particles_input, self.pso_w_label, self.pso_w_input,
                      self.pso_c1_label, self.pso_c1_input, self.pso_c2_label, self.pso_c2_input,
                      self.pso_vmax_label, self.pso_vmax_input, self.topology_label, self.topology_selector]

        ga_fields = [self.ga_population_label, self.ga_population_input,
                     self.ga_generations_label, self.ga_generations_input,
                     self.ga_mutation_label, self.ga_mutation_input, self.ga_crossover_selector,
                     self.ga_crossover_label]

        gbc_fields = [self.gbc_sn_label, self.gbc_sn_input,
                      self.gbc_max_cycle_label, self.gbc_max_cycle_input,
                      self.gbc_trial_limit_label, self.gbc_trial_limit_input]
        # Common field to hide for GA and GBC
        max_iter_fields = [self.iter_label, self.iter_input]

        bridge_fields = [self.bridge_checkbox,self.bridge_city_b_input,self.bridge_city_a_input]

        if "ACO" in algorithm:
            for field in aco_fields:
                field.show()
            for field in pso_fields + ga_fields + gbc_fields:
                field.hide()
            for field in max_iter_fields:
                field.show()
            for field in bridge_fields:
                field.show()
        elif "PSO" in algorithm:
            for field in pso_fields:
                field.show()
            for field in aco_fields + ga_fields + gbc_fields:
                field.hide()
            for field in max_iter_fields:
                field.show()
            for field in bridge_fields:
                field.show()
        elif "GA" in algorithm:
            for field in ga_fields:
                field.show()
            for field in aco_fields + pso_fields + gbc_fields:
                field.hide()
            for field in max_iter_fields:
                field.hide()
            for field in bridge_fields:
                field.hide()
        elif "DABC_FNS" in algorithm:
            for field in gbc_fields:
                field.show()
            for field in aco_fields + pso_fields + ga_fields:
                field.hide()
            for field in max_iter_fields:
                field.hide()
            for field in bridge_fields:
                field.hide()
        self.ga_canvas.setVisible("GA" in algorithm)

    def show_about_page(self):
        self.about_window = AboutPage(
            return_callback=self.return_from_about,
            prev_geometry=self.geometry(),
            was_maximized=self.isMaximized()
        )
        self.about_window.show()
        self.hide()

    def return_from_about(self):
        self.about_window.close()
        self.show()
        if self.isMaximized():  # Re-apply fullscreen if it was
            self.showMaximized()

