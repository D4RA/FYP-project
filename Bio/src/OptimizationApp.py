# main.py
import tkinter as tk
from tkinter import ttk
from pso import run_pso
from ACo2 import run_aco
from Utils import generate_random_nodes,  plot_tsp_solution
import threading

class OptimizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Optimization Algorithms")

        # Algorithm selection
        self.algorithm_var = tk.StringVar(value="PSO")
        ttk.Label(root, text="Select Algorithm:").pack(pady=5)
        algorithm_menu = ttk.Combobox(root, textvariable=self.algorithm_var, values=["PSO", "ACO"])
        algorithm_menu.pack()

        # Parameters
        self.node_count_var = tk.IntVar(value=10)
        self.iterations_var = tk.IntVar(value=100)

        ttk.Label(root, text="Number of Nodes:").pack(pady=5)
        tk.Entry(root, textvariable=self.node_count_var).pack()

        ttk.Label(root, text="Number of Iterations:").pack(pady=5)
        tk.Entry(root, textvariable=self.iterations_var).pack()

        # Start button
        self.start_button = ttk.Button(root, text="Start", command=self.run_algorithm)
        self.start_button.pack(pady=10)

        # Output
        self.output_label = ttk.Label(root, text="Output: ", font=("Arial", 12))
        self.output_label.pack(pady=10)

    def run_algorithm(self):
        self.start_button.config(state=tk.DISABLED)
        algorithm = self.algorithm_var.get()
        num_nodes = self.node_count_var.get()
        iterations = self.iterations_var.get()

        # Generate random nodes
        nodes = generate_random_nodes(num_nodes)

        # Run the selected algorithm in a separate thread
        threading.Thread(target=self.execute_algorithm, args=(algorithm, nodes, iterations)).start()

    def execute_algorithm(self, algorithm, nodes, iterations):
        best_solution = None
        best_cost = float('inf')

        if algorithm == "PSO":
            best_solution, best_cost = run_pso(nodes, iterations)
        elif algorithm == "ACO":
            best_solution, best_cost = run_aco(nodes, iterations)

        # Update UI with results
        self.output_label.config(text=f"Output: Best Cost = {best_cost}")

        # Plot the solution
        plot_tsp_solution(nodes, best_solution, title=f"{algorithm} Solution (Cost: {best_cost:.2f})")

        self.start_button.config(state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    app = OptimizationApp(root)
    root.mainloop()
