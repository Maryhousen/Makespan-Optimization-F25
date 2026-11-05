# Makespan Optimization for Task Scheduling on Heterogeneous Machines

**Course:** Intelligent Algorithms (BIA601)  
**Semester:** Fall 2026

## 1. Project Overview

This project implements and compares three intelligent optimization algorithms for scheduling independent tasks on heterogeneous machines:

- Genetic Algorithm (GA)
- Particle Swarm Optimization (PSO)
- Ant Colony Optimization (ACO)

A Greedy scheduling heuristic is used as a baseline for performance comparison.

All three optimization algorithms are implemented in Python without relying on ready-made optimization algorithm libraries.

The experiments use reproducible synthetic processing-time data representing a manufacturing workshop. The dataset does not contain actual company measurements.

## 2. Problem Definition

Each task must be assigned to exactly one machine. Processing times vary depending on the selected machine.

Machines operate in parallel, while each machine processes its assigned tasks sequentially.

**Objective:** Minimize the makespan, defined as the maximum completion time across all machines.

The solution is represented by an integer array, where each element identifies the machine assigned to a particular task.

## 3. Implemented Algorithms

| Algorithm | Main Mechanism | Stopping Criterion |
|---|---|---|
| GA | Tournament selection, uniform crossover, mutation, elitism | 80 generations |
| PSO | Particle velocity and position updates, discrete assignment decoding | 80 iterations |
| ACO | Pheromone-based probabilistic task assignments | 80 iterations |
| Greedy | Assignments based on earliest expected completion time | Deterministic |

Default parameters include 32 search elements per algorithm.

## 4. Experimental Setup

The algorithms are evaluated on three problem sizes:

| Instance | Tasks | Machines | Random Seed |
|---|---:|---:|---:|
| Small | 20 | 3 | 501 |
| Medium | 50 | 5 | 601 |
| Large | 100 | 8 | 701 |

Each algorithm is independently executed six times per problem size, using controlled random seeds.

**Total stochastic runs: 3 × 3 × 6 = 54**

The experiments evaluate:

- Solution quality (makespan)
- Execution time
- Convergence behavior
- Mean performance and standard deviation
- Improvement over the Greedy baseline

## 5. Requirements

- Python 3.10 or newer
- Dependencies listed in `requirements.txt`

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

## 6. Running the Project

Execute the following commands from the repository root.

**Run the automated tests:**

```bash
python -m unittest discover -s tests -v
```

**Reproduce the benchmark experiments:**

```bash
python run_experiments.py
```

This command runs the 54 stochastic experiments and generates the results, statistical summaries, and visualizations.

**Launch the optional interactive dashboard:**

```bash
streamlit run app.py
```

## 7. Project Structure

| File / Directory | Description |
|---|---|
| `scheduler.py` | Problem definition, fitness evaluation, Greedy, GA, PSO, and ACO |
| `run_experiments.py` | Benchmark execution, result aggregation, and plot generation |
| `app.py` | Interactive Streamlit dashboard |
| `requirements.txt` | Required Python dependencies |
| `tests/` | Automated correctness and reproducibility tests |
| `results/` | Benchmark data, CSV results, assignments, and figures |
| `README.md` | Project documentation and execution instructions |

## 8. Experimental Results

The following algorithms achieved the lowest mean makespan for each problem size:

| Instance | Best Algorithm | Mean Makespan | Improvement vs. Greedy |
|---|---|---:|---:|
| Small | GA | 118.65 min | 7.67% |
| Medium | ACO | 200.57 min | 3.03% |
| Large | GA | 220.75 min | 0.87% |

GA achieved the lowest mean makespan in the Small and Large instances, while ACO performed best in the Medium instance.

PSO did not improve upon the Greedy baseline in the Large instance under the tested configuration.

These findings apply to the evaluated instances and parameters and do not establish global optimality.

## 9. Generated Results

The `results/` directory contains:

- `runs.csv`: Individual benchmark runs and recorded performance.
- `summary.csv`: Mean makespan, standard deviation, and runtime statistics.
- `convergence.csv`: Convergence histories across iterations.
- `instances/`: Generated processing-time matrices.
- `assignments/`: Best recorded task-to-machine assignments.
- `plots/`: Convergence, solution-quality, and runtime charts.
- `metadata.json`: Experiment configuration and reproducibility metadata.

The makespan results are reproducible using the same random seeds and implementation. Execution times may differ depending on the computing environment.

## 10. Limitations

The experiments use synthetic datasets and do not model task dependencies, individual deadlines, machine downtime, or sequence-dependent setup times.

Results are limited to the three benchmark instances and the specified experimental settings.

## 11. Project Report and Team

The final English project report contains the mathematical formulation, algorithm descriptions, experimental results, visualizations, conclusions, and individual student contributions.

**Team size:** 8 students.
