"""Run with: python -m unittest discover -s tests -v"""
import itertools
import unittest

from scheduler import (
    Instance, ALGORITHMS, ant_colony_optimization, fitness,
    generate_instance, greedy_assignment, machine_loads, theoretical_lower_bound,
)


class SchedulingTests(unittest.TestCase):
    def setUp(self):
        self.instance = Instance("toy", (
            (5.0, 10.0, 7.0),
            (7.0, 4.0, 8.0),
            (9.0, 8.0, 5.0),
            (4.0, 6.0, 7.0),
            (8.0, 5.0, 9.0),
            (6.0, 8.0, 4.0),
        ))

    def test_objective_and_validation(self):
        a = [0, 1, 2, 0, 1, 2]
        self.assertEqual(machine_loads(self.instance, a), [9.0, 9.0, 9.0])
        self.assertEqual(fitness(self.instance, a), 9.0)
        with self.assertRaises(ValueError):
            fitness(self.instance, [0, 1])
        with self.assertRaises(ValueError):
            fitness(self.instance, [0, 1, 2, 0, 1, 99])

    def test_lower_bound(self):
        optimum = min(fitness(self.instance, a) for a in itertools.product(range(3), repeat=6))
        self.assertLessEqual(theoretical_lower_bound(self.instance), optimum + 1e-9)
        self.assertEqual(optimum, 9.0)

    def test_all_algorithms_feasible_reproducible_and_monotone(self):
        greedy = fitness(self.instance, greedy_assignment(self.instance))
        for name, algorithm in ALGORITHMS.items():
            with self.subTest(algorithm=name):
                res = algorithm(self.instance, seed=42, iterations=12, **{
                    "GA": {"population_size": 12},
                    "PSO": {"swarm_size": 12},
                    "ACO": {"ant_count": 12},
                }[name])
                second = algorithm(self.instance, seed=42, iterations=12, **{
                    "GA": {"population_size": 12},
                    "PSO": {"swarm_size": 12},
                    "ACO": {"ant_count": 12},
                }[name])
                self.assertEqual(res.history, second.history)
                self.assertEqual(res.assignment, second.assignment)
                self.assertEqual(len(res.history), 13)
                self.assertEqual(len(res.assignment), self.instance.num_jobs)
                self.assertAlmostEqual(res.best_fitness, fitness(self.instance, res.assignment))
                self.assertLessEqual(res.best_fitness, greedy + 1e-8)
                self.assertTrue(all(x >= y - 1e-8 for x, y in zip(res.history, res.history[1:])))

    def test_generation_is_reproducible(self):
        a = generate_instance("same", 20, 4, 123)
        b = generate_instance("same", 20, 4, 123)
        self.assertEqual(a, b)
        self.assertEqual(len(a.durations), 20)
        self.assertEqual(len(a.durations[0]), 4)

    def test_zero_iterations(self):
        result = ant_colony_optimization(self.instance, 5, iterations=0)
        self.assertEqual(len(result.history), 1)
        self.assertAlmostEqual(result.best_fitness, fitness(self.instance, result.assignment))


if __name__ == "__main__":
    unittest.main()
