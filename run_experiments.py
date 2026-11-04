"""Run reproducible GA / PSO / ACO comparisons and export tables and figures.

Usage:
    python run_experiments.py
    python run_experiments.py --runs 3 --iterations 40 --population 24
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import platform
import statistics
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scheduler import ALGORITHMS, fitness, generate_instance, greedy_assignment, machine_loads, theoretical_lower_bound

SIZE_CONFIG = {
    'Small': (20, 3, 501),
    'Medium': (50, 5, 601),
    'Large': (100, 8, 701),
}
ALGO_ORDER = ('GA', 'PSO', 'ACO')
ALGO_COLORS = {'GA': '#2563eb', 'PSO': '#0891b2', 'ACO': '#d97706'}


def csv_write(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_stdev(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def plot_convergence(data_rows, size, baseline, path):
    plt.figure(figsize=(8.4, 4.6))
    for algorithm in ALGO_ORDER:
        points = sorted((x for x in data_rows if x['size'] == size and x['algorithm'] == algorithm),
                        key=lambda r: r['iteration'])
        plt.plot([v['iteration'] for v in points],
                 [v['mean_best_fitness'] for v in points],
                 label=algorithm, color=ALGO_COLORS[algorithm], linewidth=2.0)
    plt.axhline(baseline, color='#64748b', linestyle='--', linewidth=1.2, label='Greedy baseline')
    plt.title(f'Convergence | {size} instance (lower is better)')
    plt.xlabel('Iteration')
    plt.ylabel('Mean best-so-far makespan (minutes)')
    plt.grid(alpha=.20)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_summary(summary_rows, baselines, metric, ylabel, title, path):
    sizes = list(SIZE_CONFIG)
    width = .23
    base_x = list(range(len(sizes)))
    fig, ax = plt.subplots(figsize=(8.4, 4.65))
    for ai, algorithm in enumerate(ALGO_ORDER):
        subset = [next(r for r in summary_rows if r['size'] == size and r['algorithm'] == algorithm) for size in sizes]
        x = [v + (ai - 1) * width for v in base_x]
        bars = ax.bar(x, [r[metric] for r in subset], width=width * .92, label=algorithm,
                      color=ALGO_COLORS[algorithm], alpha=.9)
        if metric == 'mean_best_fitness':
            ax.errorbar(x, [r[metric] for r in subset],
                        yerr=[r['std_best_fitness'] for r in subset],
                        fmt='none', ecolor='#111827', capsize=3, linewidth=1)
    if metric == 'mean_best_fitness':
        for xi, size in enumerate(sizes):
            ax.plot([xi - .39, xi + .39], [baselines[size], baselines[size]],
                    linestyle=':', linewidth=1.4, color='#64748b')
    ax.set_xticks(base_x, sizes)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=int, default=6, help='Repetitions for each algorithm/size (default: 6)')
    parser.add_argument('--iterations', type=int, default=80, help='Iterations per algorithm (default: 80)')
    parser.add_argument('--population', type=int, default=32, help='Population, swarm size and ant count (default: 32)')
    parser.add_argument('--output', type=Path, default=Path('results'), help='Output directory')
    args = parser.parse_args()
    if args.runs < 1 or args.iterations < 1 or args.population < 4:
        parser.error('runs>=1, iterations>=1, population>=4 are required')
    out = args.output
    (out / 'plots').mkdir(parents=True, exist_ok=True)
    (out / 'assignments').mkdir(parents=True, exist_ok=True)
    (out / 'instances').mkdir(parents=True, exist_ok=True)
    rows = []
    convergence_rows = []
    summary_rows = []
    best_assignments = {}
    baselines = {}
    lower_bounds = {}
    all_histories = {}

    for size_idx, (size, (jobs, machines, instance_seed)) in enumerate(SIZE_CONFIG.items()):
        instance = generate_instance(size, jobs, machines, instance_seed)
        csv_write(out / 'instances' / f'{size.lower()}_durations.csv',
                  [{'task_id': i+1, **{f'machine_{m+1}_min': duration for m, duration in enumerate(row)}}
                   for i, row in enumerate(instance.durations)],
                  ['task_id'] + [f'machine_{m+1}_min' for m in range(machines)])
        baseline_assignment = greedy_assignment(instance)
        baseline = fitness(instance, baseline_assignment)
        lower_bound = theoretical_lower_bound(instance)
        baselines[size] = baseline
        lower_bounds[size] = lower_bound
        all_histories[size] = {}
        best_assignments[size] = {}
        for algorithm in ALGO_ORDER:
            algorithm_results = []
            histories = []
            best_sol = None
            for run_idx in range(args.runs):
                run_seed = 12000 + size_idx * 1000 + run_idx
                extras = {'GA': {'population_size': args.population},
                          'PSO': {'swarm_size': args.population},
                          'ACO': {'ant_count': args.population}}[algorithm]
                start = time.perf_counter()
                result = ALGORITHMS[algorithm](instance, seed=run_seed, iterations=args.iterations, **extras)
                elapsed = time.perf_counter() - start
                assert abs(result.best_fitness - fitness(instance, result.assignment)) < 1e-6
                assert result.best_fitness <= baseline + 1e-6
                if best_sol is None or result.best_fitness < best_sol.best_fitness:
                    best_sol = result
                record = {
                    'size': size, 'jobs': jobs, 'machines': machines, 'instance_seed': instance_seed,
                    'algorithm': algorithm, 'run': run_idx + 1, 'run_seed': run_seed,
                    'baseline_fitness': baseline, 'lower_bound': round(lower_bound, 4),
                    'best_fitness': round(result.best_fitness, 5),
                    'improvement_pct': round(100 * (baseline - result.best_fitness) / baseline, 4),
                    'runtime_sec': round(elapsed, 6), 'evaluations': result.evaluations,
                    'convergence_iteration': result.convergence_iteration,
                    'machine_loads_std': round(statistics.pstdev(machine_loads(instance, result.assignment)), 4),
                }
                rows.append(record)
                algorithm_results.append(record)
                histories.append(result.history)
            assert best_sol is not None
            best_assignments[size][algorithm] = best_sol.assignment
            all_histories[size][algorithm] = histories
            for iteration in range(args.iterations + 1):
                current = [h[iteration] for h in histories]
                convergence_rows.append({
                    'size': size, 'algorithm': algorithm, 'iteration': iteration,
                    'mean_best_fitness': round(statistics.mean(current), 5),
                    'std_best_fitness': round(safe_stdev(current), 5),
                })
            values = [r['best_fitness'] for r in algorithm_results]
            times = [r['runtime_sec'] for r in algorithm_results]
            summary_rows.append({
                'size': size, 'jobs': jobs, 'machines': machines, 'algorithm': algorithm,
                'runs': args.runs, 'iterations': args.iterations, 'population': args.population,
                'baseline_fitness': baseline, 'lower_bound': round(lower_bound, 4),
                'mean_best_fitness': round(statistics.mean(values), 4),
                'std_best_fitness': round(safe_stdev(values), 4),
                'min_best_fitness': round(min(values), 4),
                'mean_runtime_sec': round(statistics.mean(times), 5),
                'std_runtime_sec': round(safe_stdev(times), 5),
                'mean_improvement_pct': round(statistics.mean(r['improvement_pct'] for r in algorithm_results), 4),
                'mean_first_best_iteration': round(statistics.mean(r['convergence_iteration'] for r in algorithm_results), 2),
                'mean_evaluations': round(statistics.mean(r['evaluations'] for r in algorithm_results), 2),
            })
            print(f'{size:6}  {algorithm:3}  best(mean±SD)={statistics.mean(values):8.2f}±{safe_stdev(values):5.2f}'
                  f'  runtime(mean)={statistics.mean(times):6.3f}s  baseline={baseline:7.2f}', flush=True)
        for algorithm, assignment in best_assignments[size].items():
            loads = machine_loads(instance, assignment)
            csv_write(out / 'assignments' / f'{size.lower()}_{algorithm.lower()}_best.csv',
                      [{'task_id': j+1, 'machine_id': assignment[j]+1,
                        'processing_minutes': instance.durations[j][assignment[j]],
                        'total_load_of_assigned_machine_minutes': round(loads[assignment[j]], 3)}
                       for j in range(jobs)],
                      ['task_id', 'machine_id', 'processing_minutes', 'total_load_of_assigned_machine_minutes'])

    csv_write(out / 'runs.csv', rows, list(rows[0]))
    csv_write(out / 'summary.csv', summary_rows, list(summary_rows[0]))
    csv_write(out / 'convergence.csv', convergence_rows, list(convergence_rows[0]))
    for size in SIZE_CONFIG:
        plot_convergence(convergence_rows, size, baselines[size], out / 'plots' / f'convergence_{size.lower()}.png')
    plot_summary(summary_rows, baselines, 'mean_best_fitness', 'Mean makespan (minutes)',
                 'Solution quality by problem size (error bars = ±1 SD)', out / 'plots' / 'solution_quality.png')
    plot_summary(summary_rows, baselines, 'mean_runtime_sec', 'Mean runtime (seconds)',
                 'Execution time by problem size', out / 'plots' / 'runtime.png')

    settings = {
        'problem': 'Unrelated parallel machine scheduling; minimize makespan',
        'data_origin': 'Synthetic, seeded; not real company data',
        'matrix_generation': 'round(work_j / machine_speed_m * affinity_jm + setup_jm, 2)',
        'size_config': {k: {'jobs': v[0], 'machines': v[1], 'instance_seed': v[2]} for k,v in SIZE_CONFIG.items()},
        'runs_per_algorithm_size': args.runs,
        'iterations': args.iterations,
        'population_or_swarm_or_ants': args.population,
        'stochastic_seeds': '12000 + size_index * 1000 + run_index, 0-based indices',
        'algorithms': {
            'GA': {'crossover': 'uniform', 'crossover_rate': 0.9, 'mutation_rate': '1 / jobs per gene',
                   'swap_rate': 0.25, 'elitism': 2, 'tournament_size': 3},
            'PSO': {'inertia': 0.68, 'cognitive': 1.6, 'social': 1.6,
                    'velocity_limit': 2.0, 'exploration_rate': 0.012, 'rounding': 'nearest int and clip'},
            'ACO': {'alpha': 1.0, 'beta': 3.0, 'evaporation': 0.2,
                    'global_elite_weight': 1.5, 'pheromone_floor': 0.04, 'pheromone_ceiling': 10.0},
        },
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'time_metric': 'wall time; machine/environment dependent',
    }
    (out / 'metadata.json').write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\nSaved {len(rows)} runs, {len(summary_rows)} summaries and five plots to {out}')


if __name__ == '__main__':
    main()
