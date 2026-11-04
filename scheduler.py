"""Independent-task scheduling on unrelated parallel machines.

Each assignment[j] is a machine ID in range(num_machines).
Objective: minimize makespan, max over machines of summed processing times.
The metaheuristics are implemented directly, not imported from solver libraries.
All stochastic algorithms take an explicit seed for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Sequence


@dataclass(frozen=True)
class Instance:
    """durations[j][m]: minutes to process task j on machine m."""

    name: str
    durations: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if not self.durations or not self.durations[0]:
            raise ValueError("Instance must include at least one task and machine")
        machine_count = len(self.durations[0])
        if any(len(row) != machine_count for row in self.durations):
            raise ValueError("All tasks must have times for the same machines")
        if any(not math.isfinite(t) or t <= 0 for row in self.durations for t in row):
            raise ValueError("Processing times must be positive finite numbers")

    @property
    def num_jobs(self) -> int:
        return len(self.durations)

    @property
    def num_machines(self) -> int:
        return len(self.durations[0])


def generate_instance(name: str, jobs: int, machines: int, seed: int) -> Instance:
    """Deterministic synthetic workshop with different speeds/affinities.

    Duration_ij = round(work_j / speed_i * affinity_ji + setup_ji, 2).
    Different task-machine pairs have distinct processing times.
    """
    if jobs < 1 or machines < 1:
        raise ValueError("jobs and machines must both be positive")
    rng = random.Random(seed)
    speeds = [rng.uniform(0.72, 1.38) for _ in range(machines)]
    work = [rng.randint(8, 30) for _ in range(jobs)]
    matrix = []
    for w in work:
        matrix.append(tuple(
            round(w / speed * rng.uniform(0.78, 1.28) + rng.uniform(0, 3), 2)
            for speed in speeds
        ))
    return Instance(name, tuple(matrix))


def machine_loads(instance: Instance, assignment: Sequence[int]) -> list[float]:
    if len(assignment) != instance.num_jobs:
        raise ValueError("Exactly one assignment is required per task")
    loads = [0.0] * instance.num_machines
    for job, machine in enumerate(assignment):
        if not isinstance(machine, int) or not (0 <= machine < instance.num_machines):
            raise ValueError(f"Invalid machine {machine} for task {job}")
        loads[machine] += instance.durations[job][machine]
    return loads


def fitness(instance: Instance, assignment: Sequence[int]) -> float:
    return max(machine_loads(instance, assignment))


def theoretical_lower_bound(instance: Instance) -> float:
    """Valid, sometimes loose bound; NOT a known optimum."""
    minima = [min(row) for row in instance.durations]
    return max(sum(minima) / instance.num_machines, max(minima))


def greedy_assignment(instance: Instance) -> list[int]:
    """LPT-like baseline: hardest jobs first, choose earliest finish."""
    n, m = instance.num_jobs, instance.num_machines
    order = sorted(range(n), key=lambda j: (-min(instance.durations[j]), j))
    loads = [0.0] * m
    assignment = [0] * n
    for j in order:
        machine = min(
            range(m), key=lambda k: (loads[k] + instance.durations[j][k], instance.durations[j][k], k)
        )
        assignment[j] = machine
        loads[machine] += instance.durations[j][machine]
    return assignment


@dataclass
class SearchResult:
    algorithm: str
    assignment: list[int]
    best_fitness: float
    history: list[float]  # initial population + one best-so-far value per iteration
    evaluations: int

    @property
    def convergence_iteration(self) -> int:
        """First recorded iteration reaching the final best fitness."""
        return next(i for i, v in enumerate(self.history) if abs(v - self.best_fitness) <= 1e-8)


def _mutate(chrom: list[int], machine_count: int, rng: random.Random,
            gene_rate: float, swap_rate: float) -> list[int]:
    mutated = chrom.copy()
    for j in range(len(mutated)):
        if rng.random() < gene_rate:
            old = mutated[j]
            # Always select a different machine, where possible.
            mutated[j] = (old + rng.randrange(1, machine_count)) % machine_count if machine_count > 1 else old
    if len(mutated) > 1 and rng.random() < swap_rate:
        a, b = rng.sample(range(len(mutated)), 2)
        mutated[a], mutated[b] = mutated[b], mutated[a]
    return mutated


def genetic_algorithm(instance: Instance, seed: int, iterations: int = 80,
                      population_size: int = 32, mutation_rate: float | None = None,
                      crossover_rate: float = 0.9, elite_count: int = 2,
                      tournament_size: int = 3) -> SearchResult:
    """Integer GA: tournament selection, uniform crossover, mutation, elitism."""
    if population_size < 3 or not (0 <= elite_count < population_size):
        raise ValueError("Invalid population/elitism settings")
    if iterations < 0:
        raise ValueError("Iterations cannot be negative")
    rng = random.Random(seed)
    n, m = instance.num_jobs, instance.num_machines
    mutation_rate = 1 / n if mutation_rate is None else mutation_rate
    if not 0 <= mutation_rate <= 1:
        raise ValueError("mutation_rate must be between 0 and 1")
    baseline = greedy_assignment(instance)
    pop = [baseline]
    # Start near a feasible greedy solution, with diverse global candidates.
    while len(pop) < population_size:
        if len(pop) % 4 != 0:
            candidate = baseline.copy()
            changes = rng.randint(1, max(2, n // 4))
            for j in rng.sample(range(n), min(changes, n)):
                candidate[j] = rng.randrange(m)
        else:
            candidate = [rng.randrange(m) for _ in range(n)]
        pop.append(candidate)
    scored = sorted(((fitness(instance, a), a) for a in pop), key=lambda x: x[0])
    evaluations = population_size
    history = [scored[0][0]]

    def tournament(items: list[tuple[float, list[int]]]) -> list[int]:
        return min((rng.choice(items) for _ in range(tournament_size)), key=lambda t: t[0])[1]

    for _ in range(iterations):
        new_pop = [a.copy() for _, a in scored[:elite_count]]
        while len(new_pop) < population_size:
            parent_a, parent_b = tournament(scored), tournament(scored)
            if rng.random() < crossover_rate:
                child = [parent_a[j] if rng.random() < 0.5 else parent_b[j] for j in range(n)]
            else:
                child = parent_a.copy()
            child = _mutate(child, m, rng, mutation_rate, 0.25)
            new_pop.append(child)
        scored = sorted(((fitness(instance, a), a) for a in new_pop), key=lambda x: x[0])
        evaluations += population_size
        history.append(scored[0][0])
    best_value, best_assignment = scored[0]
    return SearchResult("GA", best_assignment.copy(), best_value, history, evaluations)


def particle_swarm_optimization(instance: Instance, seed: int, iterations: int = 80,
                                swarm_size: int = 32, inertia: float = 0.68,
                                cognitive: float = 1.6, social: float = 1.6,
                                velocity_limit: float = 2.0,
                                exploration_rate: float = 0.012) -> SearchResult:
    """Discrete PSO via continuous positions rounded to machine IDs.

    A small random reset of coordinates preserves exploration in discrete space.
    """
    if swarm_size < 2 or iterations < 0:
        raise ValueError("Invalid PSO settings")
    rng = random.Random(seed)
    n, m = instance.num_jobs, instance.num_machines
    seed_solution = greedy_assignment(instance)

    def decode(pos: list[float]) -> list[int]:
        return [max(0, min(m - 1, int(math.floor(x + 0.5)))) for x in pos]

    positions: list[list[float]] = []
    velocities: list[list[float]] = []
    for p in range(swarm_size):
        if p == 0:
            pos = [float(x) for x in seed_solution]
        elif p < swarm_size * 0.70:
            sol = seed_solution.copy()
            for j in rng.sample(range(n), rng.randint(1, max(2, n // 4))):
                sol[j] = rng.randrange(m)
            pos = [float(x) + rng.uniform(-0.48, 0.48) for x in sol]
        else:
            pos = [rng.uniform(0, m - 1) for _ in range(n)]
        positions.append(pos)
        velocities.append([rng.uniform(-0.5, 0.5) for _ in range(n)])

    pbest_positions = [p.copy() for p in positions]
    pbest_fitness = [fitness(instance, decode(p)) for p in positions]
    gbest_index = min(range(swarm_size), key=pbest_fitness.__getitem__)
    gbest_position = pbest_positions[gbest_index].copy()
    gbest_fitness = pbest_fitness[gbest_index]
    gbest_assignment = decode(gbest_position)
    history = [gbest_fitness]
    evaluations = swarm_size

    for _ in range(iterations):
        for particle in range(swarm_size):
            for j in range(n):
                velocity = (
                    inertia * velocities[particle][j]
                    + cognitive * rng.random() * (pbest_positions[particle][j] - positions[particle][j])
                    + social * rng.random() * (gbest_position[j] - positions[particle][j])
                )
                velocity = max(-velocity_limit, min(velocity_limit, velocity))
                velocities[particle][j] = velocity
                positions[particle][j] = max(0.0, min(m - 1.0, positions[particle][j] + velocity))
                if rng.random() < exploration_rate:
                    positions[particle][j] = float(rng.randrange(m))
            assignment = decode(positions[particle])
            value = fitness(instance, assignment)
            evaluations += 1
            if value < pbest_fitness[particle] - 1e-9:
                pbest_fitness[particle] = value
                pbest_positions[particle] = positions[particle].copy()
                if value < gbest_fitness - 1e-9:
                    gbest_position = positions[particle].copy()
                    gbest_fitness = value
                    gbest_assignment = assignment.copy()
        history.append(gbest_fitness)
    return SearchResult("PSO", gbest_assignment, gbest_fitness, history, evaluations)


def _roulette(weights: list[float], rng: random.Random) -> int:
    s = sum(weights)
    if s <= 0 or not math.isfinite(s):
        return rng.randrange(len(weights))
    pick = rng.random() * s
    for idx, weight in enumerate(weights):
        pick -= weight
        if pick <= 0:
            return idx
    return len(weights) - 1


def ant_colony_optimization(instance: Instance, seed: int, iterations: int = 80,
                            ant_count: int = 32, alpha: float = 1.0,
                            beta: float = 3.0, evaporation: float = 0.2,
                            global_elite_weight: float = 1.5) -> SearchResult:
    """ACO with task-machine pheromone, load-aware heuristic and elitist deposit."""
    if ant_count < 1 or iterations < 0 or not (0 < evaporation < 1):
        raise ValueError("Invalid ACO settings")
    rng = random.Random(seed)
    n, m = instance.num_jobs, instance.num_machines
    best_assignment = greedy_assignment(instance)
    best_value = fitness(instance, best_assignment)
    # Warm-start pheromone from same greedy baseline used by GA and PSO.
    pheromone = [[1.3 if best_assignment[j] == k else 1.0 for k in range(m)] for j in range(n)]
    history = [best_value]
    evaluations = 1  # The greedy baseline used to initialize best-so-far.
    durations = instance.durations

    for _ in range(iterations):
        iteration_best: list[int] | None = None
        iteration_value = math.inf
        for ant in range(ant_count):
            # Task difficulty ordering with randomness for solution diversity.
            order = sorted(range(n), key=lambda j: min(durations[j]) * rng.uniform(0.78, 1.22), reverse=True)
            assignment = [0] * n
            loads = [0.0] * m
            for j in order:
                desirability = [
                    pheromone[j][k] ** alpha * (1.0 / (loads[k] + durations[j][k])) ** beta
                    for k in range(m)
                ]
                machine = _roulette(desirability, rng)
                assignment[j] = machine
                loads[machine] += durations[j][machine]
            value = max(loads)
            evaluations += 1
            if value < iteration_value:
                iteration_best, iteration_value = assignment, value
            if value < best_value - 1e-9:
                best_value, best_assignment = value, assignment.copy()

        keep = 1.0 - evaporation
        for row in pheromone:
            for k in range(m):
                row[k] = max(0.04, keep * row[k])
        assert iteration_best is not None
        # Relative scaling keeps deposit magnitudes interpretable across instance sizes.
        for j, machine in enumerate(iteration_best):
            pheromone[j][machine] = min(10.0, pheromone[j][machine] + best_value / iteration_value)
        for j, machine in enumerate(best_assignment):
            pheromone[j][machine] = min(10.0, pheromone[j][machine] + global_elite_weight)
        history.append(best_value)
    return SearchResult("ACO", best_assignment.copy(), best_value, history, evaluations)


ALGORITHMS = {
    "GA": genetic_algorithm,
    "PSO": particle_swarm_optimization,
    "ACO": ant_colony_optimization,
}
