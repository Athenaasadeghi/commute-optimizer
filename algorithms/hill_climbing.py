"""
algorithms/hill_climbing.py
Hill Climbing search with sideways move limits and random restarts.
"""

from typing import Tuple
from algorithms.base import LocalSearchSolver
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors


class HillClimbingSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.max_sideways: int = self.params.get("max_sideways", 100)
        self.max_restarts: int = self.params.get("max_restarts", 5)


    def solve(self) -> Tuple[State, float]:
        """
        Executes Hill Climbing optimization.

        Guidelines:
        - Search through the neighborhood greedily for candidates that minimize total travel time.
        - Handle flat local maxima / shoulders by allowing up to `max_sideways` non-improving steps.
        - Apply up to `max_restarts` to explore different starting basins.
        - Track the best objective cost seen at each major step in self.history.

        Prohibited:
        - Looping indefinitely on flat plateaus.
        """
        current_state = self.problem.generate_initial_state()

        best_state = current_state
        best_cost = self.problem.evaluate(current_state)

        for restart in range(self.max_restarts + 1):

            if restart > 0:
                current_state = self.problem.generate_initial_state()

            sideways_count = 0

            while True:
                neighbors = get_neighbors(current_state, self.problem)

                if not neighbors:
                    break

                best_neighbor = min(
                    neighbors,
                    key=lambda state: self.problem.evaluate(state)
                )

                current_cost = self.problem.evaluate(current_state)
                neighbor_cost = self.problem.evaluate(best_neighbor)

                if neighbor_cost < current_cost:
                    current_state = best_neighbor
                    sideways_count = 0

                elif neighbor_cost == current_cost:
                    if sideways_count >= self.max_sideways:
                        break

                    current_state = best_neighbor
                    sideways_count += 1

                else:
                    break

                cost = self.problem.evaluate(current_state)
                if cost < best_cost:
                    best_state = current_state
                    best_cost = cost
            return best_state, best_cost