"""
algorithms/simulated_annealing.py
Stochastic optimization governed by a temperature schedule.
"""
import random
import math

from typing import Tuple
from algorithms.base import LocalSearchSolver
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors


class SimulatedAnnealingSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.initial_temp: float = self.params.get("initial_temp", 100.0)
        self.cooling_rate: float = self.params.get("cooling_rate", 0.995)
        self.min_temp: float = self.params.get("min_temp", 1e-3)
        self.max_iterations: int = self.params.get("max_iterations", 2000)

    def solve(self) -> Tuple[State, float]:
        """
        Executes Simulated Annealing optimization.

        Guidelines:
        - Start from a high temperature where worse moves are accepted with significant probability.
        - Lower temperature over iterations according to a cooling schedule.
        - Downhill moves (improvements) must always be accepted.
        - Uphill moves (worse costs) should be accepted probabilistically according to temperature.
        - Record objective values throughout the run in self.history.

        Prohibited:
        - Ignoring the temperature parameter or turning this into pure hill climbing.
        """
        current_state = self.problem.generate_initial_state()
        current_cost = self.problem.evaluate(current_state)

        best_state = current_state
        best_cost = current_cost

        temperature = self.initial_temp

        self.history = [best_cost]

        iteration = 0

        while temperature > self.min_temp and iteration < self.max_iterations:

            neighbors = get_neighbors(current_state, self.problem)

            if not neighbors:
                break

            next_state = random.choice(neighbors)

            next_cost = self.problem.evaluate(next_state)

            difference = next_cost - current_cost

            if difference < 0:
                current_state = next_state
                current_cost = next_cost

            else:
                probability = math.exp(-difference / temperature)

                if random.random() < probability:
                    current_state = next_state
                    current_cost = next_cost

            if current_cost < best_cost:
                best_state = current_state
                best_cost = current_cost

            self.history.append(best_cost)

            temperature = temperature * self.cooling_rate

            iteration += 1

        return best_state, best_cost
        
