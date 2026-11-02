"""
algorithms/genetic_algorithm.py
Population-based optimization using selection, recombination, and mutation.
"""
import random
import math
from typing import Tuple
from algorithms.base import LocalSearchSolver
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors, crossover


class GeneticAlgorithmSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.population_size: int = self.params.get("population_size", 30)
        self.generations: int = self.params.get("generations", 100)
        self.crossover_rate: float = self.params.get("crossover_rate", 0.8)
        self.mutation_rate: float = self.params.get("mutation_rate", 0.2)
        self.elitism: int = self.params.get("elitism", 2)

    def solve(self) -> Tuple[State, float]:
        """
        Executes Genetic Algorithm optimization.

        Guidelines:
        - Initialize a population of valid candidate solutions.
        - Design a fitness metric suitable for minimizing commute travel time (lower cost -> higher fitness).
        - Use selection operators (such as tournament or roulette selection) to pick parents.
        - Pair parents to produce offspring via `crossover()`, and apply stochastic mutation.
        - Preserve the top candidates across generations if using elitism.
        - Track best population performance per generation in self.history.

        Encouraged:
        - Experimenting with population size, selection pressure, and operator rates to balance exploration vs. exploitation.
        """
        population = []
        initial_state = self.problem.generate_initial_state()
        population.append(initial_state)
        attempts = 0
        max_attempts = self.population_size * 50
        while len(population) < self.population_size and attempts < max_attempts:
            parent = random.choice(population)
            neighbors = get_neighbors(parent, self.problem)

            if neighbors:
                new_state = random.choice(neighbors)

                if self.problem.is_valid_state(new_state) and new_state not in population:
                    population.append(new_state)
            attempts += 1
        while len(population) < self.population_size:
            population.append(random.choice(population))
        def fitness(state):
            cost = self.problem.evaluate(state)
            return 1 / (1 + cost)
        def select_parent():
            tournament_size = min(3, len(population))
            candidates = random.sample(population, tournament_size)

            return max(candidates, key=fitness)
        best_state = min(
            population,
            key=lambda state: self.problem.evaluate(state)
        )
        best_cost = self.problem.evaluate(best_state)

        self.history = [best_cost]
        for generation in range(self.generations):
            population.sort(
                key=lambda state: self.problem.evaluate(state)
            )
            elite_count = min(
                max(0, self.elitism),
                self.population_size
            )

            new_population = population[:elite_count]
            while len(new_population) < self.population_size:

                parent1 = select_parent()
                parent2 = select_parent()
                if random.random() < self.crossover_rate:
                    child = crossover(parent1, parent2, self.problem)
                else:
                    child = parent1
                if child is None or not self.problem.is_valid_state(child):
                    child = parent1
                if random.random() < self.mutation_rate:
                    neighbors = get_neighbors(child, self.problem)

                    if neighbors:
                        child = random.choice(neighbors)
                new_population.append(child)

            population = new_population
            generation_best = min(
                population,
                key=lambda state: self.problem.evaluate(state)
            )
            generation_cost = self.problem.evaluate(generation_best)
            if generation_cost < best_cost:
                best_state = generation_best
                best_cost = generation_cost
            self.history.append(generation_cost)

        return best_state, best_cost
