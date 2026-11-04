
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

        - Maintains K parallel candidate states.
        - Generates neighbors from all beam states.
        - Selects the best K successors collectively.
        - Removes duplicate states to preserve diversity.
        - Tracks the best solution and convergence history.
        """
        beam = []
        attempts = 0
        max_attempts = self.beam_width * 10

        while len(beam) < self.beam_width and attempts < max_attempts:
            state = self.problem.generate_initial_state()
            if state not in beam:
                beam.append(state)
            attempts += 1
        while len(beam) < self.beam_width:

            candidates = []

            for state in beam:
                neighbors = get_neighbors(state, self.problem)

                for neighbor in neighbors:
                    if neighbor not in beam and neighbor not in candidates:
                        candidates.append(neighbor)

            if not candidates:
                break
            candidates.sort(
                key=lambda state: self.problem.evaluate(state)
            )
            remaining = self.beam_width - len(beam)
            beam.extend(candidates[:remaining])
        beam.sort(
            key=lambda state: self.problem.evaluate(state)
        )

        if not beam:
            raise ValueError("Could not generate a valid initial state.")

        best_state = beam[0]
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

            if not new_beam:
                break
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
