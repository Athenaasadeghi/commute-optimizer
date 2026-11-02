"""
algorithms/beam_search.py
Local Beam Search maintaining K parallel candidate states.
"""

from typing import Tuple
from algorithms.base import LocalSearchSolver
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors


class LocalBeamSearchSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.beam_width: int = self.params.get("beam_width", 5)
        self.max_iterations: int = self.params.get("max_iterations", 150)

    def solve(self) -> Tuple[State, float]:
        """
        Executes Local Beam Search.

        Guidelines:
        - Maintain a set of `beam_width` parallel configurations.
        - In each iteration, generate neighbors from all active states in the beam.
        - Pool and rank all generated successors collectively, then retain the top `beam_width` states.
        - Halt when the beam ceases to improve or reaches max_iterations.
        - Record the best cost of the population at each step in self.history.

        Encouraged:
        - Ensuring diversity across states in the beam to prevent collapse into a single redundant path.

        Prohibited:
        - Treating this as K independent, non-communicating searches.
        """
        beam = []

        for i in range(self.beam_width):
            state = self.problem.generate_initial_state()

            if state not in beam:
                beam.append(state)
        best_state = min(
            beam,
            key=lambda state: self.problem.evaluate(state)
        )
        best_cost = self.problem.evaluate(best_state)
        self.history = [best_cost]
        for iteration in range(self.max_iterations):
            all_neighbors = []
            for state in beam:
                neighbors = get_neighbors(state, self.problem)

                for neighbor in neighbors:
                    if neighbor not in all_neighbors:
                        all_neighbors.append(neighbor)
            if not all_neighbors:
                break
            all_neighbors.sort(
                key=lambda state: self.problem.evaluate(state)
            )
            new_beam = all_neighbors[:self.beam_width]
            new_best_state = new_beam[0]
            new_best_cost = self.problem.evaluate(new_best_state)
            if new_best_cost >= best_cost:
                self.history.append(best_cost)
                break
            best_state = new_best_state
            best_cost = new_best_cost
            beam = new_beam
            self.history.append(best_cost)
        return best_state, best_cost