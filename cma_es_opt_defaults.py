import json
import os
import pathlib
import subprocess
import time
import warnings
from typing import Callable, Optional, List, Union

import numpy as np
from sklearn.covariance import MinCovDet

np.random.seed(42)

warnings.filterwarnings('ignore')

# Indices of parameters that should be log10-transformed internally
LOG_PARAMS = [0, 1, 2, 3, 4]

def to_internal(params: np.ndarray) -> np.ndarray:
    """Convert raw params to internal CMA-ES space: log10 for w0-w4."""
    p = np.array(params, dtype=float).copy()
    for i in LOG_PARAMS:
        p[i] = np.log10(p[i])
    return p

def to_external(params: np.ndarray) -> np.ndarray:
    """Convert internal CMA-ES space back to raw params: 10^ for w0-w4."""
    p = np.array(params, dtype=float).copy()
    for i in LOG_PARAMS:
        p[i] = 10.0 ** p[i]
    return p

def replace_init_w(filename, new_values):
    """Replace the init_w list in fsrs_v7.py with new values.
    new_values must be in EXTERNAL (raw) space.
    """
    with open(filename, 'r') as f:
        content = f.read()

    with open(filename + '.backup', 'w') as f:
        f.write(content)

    new_list_str = (
        f"init_w = [{new_values[0]}, {new_values[1]}, {new_values[2]}, "
        f"{new_values[3]}, {new_values[4]}, {new_values[5]}, {new_values[6]}, "
        f"{new_values[7]}, {new_values[8]}, {new_values[9]}, {new_values[10]}, "
        f"{new_values[11]}, {new_values[12]}, {new_values[13]}, {new_values[14]}, "
        f"{new_values[15]}, {new_values[16]}, {new_values[17]}, {new_values[18]}, "
        f"{new_values[19]}, {new_values[20]}, {new_values[21]}, {new_values[22]}, "
        f"{new_values[23]}, {new_values[24]}, {new_values[25]}, {new_values[26]}, "
        f"{new_values[27]}, {new_values[28]}, {new_values[29]}, {new_values[30]}, "
        f"{new_values[31]}, {new_values[32]}, {new_values[33]}, {new_values[34]}, "
        f"{new_values[35]}, {new_values[36]}]"
    )

    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.strip().startswith('init_w'):
            indent = len(line) - len(line.lstrip())
            lines[i] = ' ' * indent + new_list_str + '\n'
            break

    new_content = '\n'.join(lines)
    if content == new_content:
        raise Exception('No changes made. Pattern might not have matched.')

    with open(filename, 'w') as f:
        f.write(new_content)

    return True

def evaluate(params):
    model_name = 'FSRS-7-default-short-secs.jsonl'
    dir_name = r'C:\Users\Andrew\srs-benchmark-gru'

    if not os.path.exists(dir_name):
        print('Error: srs-benchmark directory not found!')
        exit()

    replace_init_w(os.path.join(dir_name, 'models', 'fsrs_v7.py'), params)

    start = time.perf_counter()
    # default parameters are optimized on users 1001-2000
    result = subprocess.run(
        # Use cwd= instead of os.chdir() so we don't mutate the process's
        # working directory globally, and so the path is always correct on
        # repeated calls.
        r'python script.py --algo FSRS-7 --short --secs --default --processes 10'
        r' --data F:\FSRS\FSRS-Anki-10k\anki-revlogs-30',
        shell=True,
        capture_output=True,
        text=True,
        cwd=dir_name,
    )
    end = time.perf_counter()
    total = end - start
    print(f'Benchmark took {total / 3600:.2f} hours, {total / 60:.1f} minutes')

    if result.returncode != 0:
        raise Exception(f'Error={result.stderr}')

    n_users = 30  # dataset is anki-revlogs-e0 (mini-dataset with 30 users)
    result_file = pathlib.Path(dir_name) / 'result' / f'{model_name}'
    if result_file.exists():
        with open(result_file, 'r') as f:
            data = [json.loads(x) for x in f.readlines()]
            assert len(data) == n_users, f"Expected {n_users} users, got {len(data)}"
            losses = [entry['metrics']['LogLoss'] for entry in data]

        os.remove(result_file)
        return np.mean(losses)
    else:
        raise Exception('No result file')

def clip_params(params, bounds):
    """Clip parameters to their bounds"""
    clipped = np.copy(params)
    for i, (lower, upper) in enumerate(bounds):
        clipped[i] = np.clip(clipped[i], lower, upper)
    return clipped

def check_constraints(params):
    """
    Check ordering constraints that mirror FSRS7ParameterClipper:
      w0 <= w1 <= w2 <= w3  (initial stabilities must be non-decreasing)
      w29 <= w30            (decay1 <= decay2)
      w31 <= w32            (base1 <= base2)
    """
    return (
        params[0] <= params[1] <= params[2] <= params[3]
        and params[29] <= params[30]
        and params[31] <= params[32]
    )

def initial_covariance_from_bounds(bounds: list) -> tuple[np.ndarray, float]:
    """
    Diagonal C where each entry is proportional to the square of
    the bound width for that parameter.
    """
    widths = np.array([upper - lower for lower, upper in bounds])
    # Use (width / 4)^2 as variance: places ~95% of initial mass within bounds
    variances = (widths / 4.0) ** 2
    max_var = float(np.max(variances))
    C_normalized = np.diag(variances / max_var)
    sigma0 = float(np.sqrt(max_var)) * 0.1  # reduce sigma since we are likely near an optimum already
    return C_normalized, sigma0

class CMAES:
    """
    Simple CMA-ES implementation for expensive optimization.
    Based on Hansen & Ostermeier.
    """

    def __init__(
        self,
        mean: Union[List[float], np.ndarray],
        sigma: float,
        bounds: List[tuple],
        constraints_fn: Callable[[np.ndarray], bool],
        population_size: Optional[int] = None
    ):
        self.n = len(mean)
        self.mean = np.array(mean, dtype=float)
        self.sigma = float(sigma)
        self.bounds = bounds
        self.constraints_fn = constraints_fn

        if population_size is None:
            self.lambda_ = 4 + int(np.floor(3 * np.log(self.n)))
        else:
            self.lambda_ = population_size

        # Mirrored sampling works best with an even population.
        if self.lambda_ % 2 != 0:
            self.lambda_ += 1

        self.mu = self.lambda_ // 2

        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mu_eff = 1 / np.sum(self.weights ** 2)

        self.cc = (4 + self.mu_eff / self.n) / (self.n + 4 + 2 * self.mu_eff / self.n)
        self.cs = (self.mu_eff + 2) / (self.n + self.mu_eff + 5)
        self.c1 = 2 / ((self.n + 1.3) ** 2 + self.mu_eff)
        self.cmu = min(
            1 - self.c1,
            2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((self.n + 2) ** 2 + self.mu_eff)
        )
        self.damps = (
            1
            + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.n + 1)) - 1)
            + self.cs
        )

        self.pc = np.zeros(self.n)
        self.ps = np.zeros(self.n)
        self.C = np.eye(self.n)
        self.B = np.eye(self.n)
        self.D = np.ones(self.n)

        self.eigeneval = 0
        self.counteval = 0

        self.chi_n = self.n ** 0.5 * (
            1 - 1 / (4 * self.n) + 1 / (21 * self.n ** 2)
        )

    def ask(self) -> list:
        """
        Generate lambda_ candidate solutions using mirrored sampling.

        Important: if clipping changes x, then y must be recomputed from
        the clipped x. Otherwise tell() updates using a mutation vector that
        did not actually produce the evaluated candidate.
        """
        if self.counteval >= self.eigeneval + self.lambda_ / (self.c1 + self.cmu) / self.n / 10:
            self.eigeneval = self.counteval
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            eigenvalues, self.B = np.linalg.eigh(self.C)
            self.D = np.sqrt(np.maximum(eigenvalues, 1e-20))

        candidates = []
        max_attempts = 2000

        def sample_from_z(z: np.ndarray):
            x_unclipped = self.mean + self.sigma * (self.B @ (self.D * z))
            x = clip_params(x_unclipped, self.bounds)

            if not self.constraints_fn(x):
                return None

            # Recompute y from the actual candidate after clipping.
            y_actual = (x - self.mean) / self.sigma
            return x, y_actual, z

        while len(candidates) < self.lambda_:
            z = np.random.randn(self.n)

            for z_use in (z, -z):
                candidate = sample_from_z(z_use)
                if candidate is not None:
                    candidates.append(candidate)
                    if len(candidates) >= self.lambda_:
                        break

            max_attempts -= 1
            if max_attempts <= 0:
                raise RuntimeError(
                    "Could not sample enough valid candidates. "
                    "Check bounds, constraints, and current mean."
                )

        return candidates

    def tell(self, candidates, fitness_values):
        """Update distribution based on fitness values."""
        sorted_indices = np.argsort(fitness_values)
        best_candidates = [candidates[i] for i in sorted_indices[:self.mu]]

        xs = np.array([c[0] for c in best_candidates])
        ys = np.array([c[1] for c in best_candidates])

        old_mean = self.mean.copy()
        self.mean = np.sum(self.weights[:, np.newaxis] * xs, axis=0)

        # Keep mean inside bounds and valid after weighted recombination.
        self.mean = clip_params(self.mean, self.bounds)

        if not self.constraints_fn(self.mean):
            # Repair only the monotone constraints.
            self.mean[0:4] = np.maximum.accumulate(self.mean[0:4])
            if self.mean[29] > self.mean[30]:
                avg = 0.5 * (self.mean[29] + self.mean[30])
                self.mean[29] = avg
                self.mean[30] = avg
            if self.mean[31] > self.mean[32]:
                avg = 0.5 * (self.mean[31] + self.mean[32])
                self.mean[31] = avg
                self.mean[32] = avg
            self.mean = clip_params(self.mean, self.bounds)

        mean_shift = (self.mean - old_mean) / self.sigma

        C_inv_sqrt = self.B @ np.diag(1 / self.D) @ self.B.T
        self.ps = (
                (1 - self.cs) * self.ps
                + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * (C_inv_sqrt @ mean_shift)
        )

        eval_count_for_hsig = max(self.counteval + self.lambda_, 1)
        hsig = (
                np.linalg.norm(self.ps)
                / np.sqrt(1 - (1 - self.cs) ** (2 * eval_count_for_hsig / self.lambda_))
                / self.chi_n
                < 1.4 + 2 / (self.n + 1)
        )

        self.pc = (
                (1 - self.cc) * self.pc
                + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * mean_shift
        )

        # BUGFIX:
        # ys are already normalized steps where x = mean + sigma * y.
        # Do NOT divide by sigma again.
        artmp = ys

        self.C = (
                (1 - self.c1 - self.cmu) * self.C
                + self.c1 * (
                        np.outer(self.pc, self.pc)
                        + (1 - hsig) * self.cc * (2 - self.cc) * self.C
                )
                + self.cmu * sum(
            self.weights[i] * np.outer(artmp[i], artmp[i])
            for i in range(self.mu)
        )
        )

        self.C = np.triu(self.C) + np.triu(self.C, 1).T

        self.sigma *= np.exp(
            (self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chi_n - 1)
        )

        # Avoid runaway step-size.
        self.sigma = float(np.clip(self.sigma, 1e-4, 0.25))

        self.counteval += self.lambda_

        return fitness_values[sorted_indices[0]]


# ============================================================================
# CONFIGURATION
# ============================================================================

checkpoint_filename = os.path.abspath('FSRS_7_cmaes_results.json')
max_generations = 50

# Parameter bounds — matched exactly to FSRS7ParameterClipper
# check_constraints enforces the ordering
bounds = [
    # w0-w4: log10 of initial stabilities and one of the D params
    (np.log10(0.005), np.log10(1.)),    # w0: log10 space
    (np.log10(0.05),  np.log10(5.)),    # w1: log10 space
    (np.log10(0.1),   np.log10(20.)),   # w2: log10 space
    (np.log10(1.),    np.log10(40.)),   # w3: log10 space
    # w4-w8: difficulty
    (np.log10(2.),    np.log10(8.)),    # w4: log10 space
    (0.01, 2),
    (2, 4),
    (0.01, 1),
    (0.01, 1),
    # w9-w17: stability long-term
    (1.5, 3),
    (0, 1.2),
    (0.3, 2),
    (0.01, 1.2),
    (0.001, 0.9),
    (0.1, 1),
    (0, 1),
    (0, 1),
    (1.3, 3.5),
    # w18-w26: stability short-term
    (0, 1.5),
    (0, 1.5),
    (2.5, 4.5),
    (0.001, 1.2),
    (0.001, 1.),
    (0.001, 1.),
    (2, 4),
    (0, 1),
    (1.3, 3.5),
    # w27-w28: long-short transition
    (2.5, 4),
    (0, 1),
    # w29-w36: forgetting curve
    (0.01, 0.25),
    (0.01, 0.7),
    (0.5, 0.75),
    (0.5, 0.99),
    (0.01, 1),
    (0.1, 1),
    (0, 0.7),
    (0.1, 1.0),
]

starting_params = np.array([
    0.0503, 2.7722, 4.47, 11.5811,
    5.7857, 0.3835, 3.3252, 0.9123, 0.0417,
    2.3013, 0.1463, 1.3259, 0.4403, 0.0117, 0.6841, 0.0779, 0.6574, 1.3,
    0.7681, 0.3408, 3.5886, 0.308, 0.013, 0.2236, 2.6014, 0.609, 1.3,
    2.5, 0.9994,
    0.0677, 0.17, 0.5, 0.9564, 0.2477, 0.6244, 0.1539, 0.4051,
])

# sanity check
for n in range(len(starting_params)):
    if n in LOG_PARAMS:
        param = np.log10(starting_params[n])
    else:
        param = starting_params[n]
    assert bounds[n][0] <= param <= bounds[n][1], f'{n}, {param}, ({bounds[n][0]}, {bounds[n][1]})'


# ================================================================================

print("=" * 80)
print("CMA-ES optimisation of default FSRS-7 parameters")
print("=" * 80)
print(f"Starting parameters: {starting_params.tolist()}")
print(f"Max generations: {max_generations}")

# None = use 4+3*ln(n_parameters) rule-of-thumb
population_size = None
if os.path.isfile(checkpoint_filename):
    try:
        print(f'\nLoading checkpoint from {checkpoint_filename}')
        with open(checkpoint_filename, 'r') as f:
            checkpoint = json.load(f)

        cma = CMAES(
            to_internal(starting_params),
            0.05,
            bounds,
            check_constraints,
            population_size=population_size
        )

        cma.mean = np.array(checkpoint['mean'], dtype=float)
        cma.sigma = float(checkpoint['sigma'])
        cma.C = np.array(checkpoint['C'], dtype=float)
        cma.pc = np.array(checkpoint['pc'], dtype=float)
        cma.ps = np.array(checkpoint['ps'], dtype=float)
        cma.counteval = int(checkpoint['counteval'])

        # Force eigendecomposition refresh after restoring C.
        cma.eigeneval = -999999

        history = checkpoint['history']
        best_params = np.array(checkpoint['best_params'])
        best_loss = checkpoint['best_loss']
        completed_generations = checkpoint['completed_generations']

        print(f'Loaded checkpoint: generation {completed_generations}, best loss {best_loss:.4f}')
    except Exception as e:
        print(f'Error loading checkpoint: {e}')
        print('Starting fresh')
        # Mean = starting params (good warm-start for location)
        mean = to_internal(starting_params)
        # C comes from bounds
        C_init, sigma0 = initial_covariance_from_bounds(bounds)
        cma = CMAES(mean, sigma0, bounds, check_constraints, population_size=population_size)
        cma.C = C_init
        cma.eigeneval = -999999

        history = []
        best_params = starting_params.copy()
        best_loss = evaluate(starting_params)
        completed_generations = 0
else:
    print('No checkpoint found, starting fresh')
    # Mean = starting params (good warm-start for location)
    mean = to_internal(starting_params)
    # C comes from bounds
    C_init, sigma0 = initial_covariance_from_bounds(bounds)
    cma = CMAES(mean, sigma0, bounds, check_constraints, population_size=population_size)
    cma.C = C_init
    cma.eigeneval = -999999

    history = []
    best_params = starting_params.copy()
    best_loss = evaluate(starting_params)
    completed_generations = 0

for generation in range(completed_generations, max_generations):
    print(f"\n{'=' * 80}")
    print(f"Generation {generation + 1}/{max_generations}")
    print(f"Current mean (internal): {[round(x, 4) for x in cma.mean.tolist()]}")
    print(f"Current sigma: {cma.sigma:.4f}")
    print(f"Best loss so far: {best_loss:.4f}")

    candidates = cma.ask()
    print(f"Generated {len(candidates)} candidates")

    fitness_values = []
    generation_history = []

    for i, (x_internal, y, z) in enumerate(candidates):
        print(f"\n  Candidate {i + 1}/{len(candidates)}")

        # Convert to external space for evaluate() and display
        x_external = to_external(x_internal)
        x_external = np.array([round(val, 4) for val in x_external.tolist()])

        print(f"  Parameters (external): {x_external.tolist()}")

        loss = evaluate(x_external.tolist())  # evaluate always gets raw params
        fitness_values.append(loss)

        print(f"  Loss={loss:.4f}")

        # Store external params in history and best_params
        generation_history.append({'params': x_external.tolist(), 'loss': loss})

        if loss < best_loss:
            best_loss = loss
            best_params = x_external.copy()  # best_params always in external space
            print(f"  New best parameters: {[round(v, 4) for v in best_params.tolist()]}")
            print(f"  New best loss: {best_loss:.4f}")

    best_gen_loss = cma.tell(candidates, fitness_values)

    print(f"\nGeneration {generation + 1} complete:")
    print(f"  Best in generation: {best_gen_loss:.4f}")
    print(f"  Best overall: {best_loss:.4f}")
    print(f"  Sigma: {cma.sigma:.4f}")

    history.append({
        'generation': generation + 1,
        'candidates': generation_history,
        'best_loss': best_gen_loss,
        'mean_loss': float(np.mean(fitness_values)),
        'sigma': float(cma.sigma),
    })

    checkpoint = {
        'mean': cma.mean.tolist(),
        'sigma': float(cma.sigma),
        'C': cma.C.tolist(),
        'pc': cma.pc.tolist(),
        'ps': cma.ps.tolist(),
        'counteval': int(cma.counteval),
        'best_params': best_params.tolist(),
        'best_loss': float(best_loss),
        'history': history,
        'completed_generations': generation + 1,
    }

    with open(checkpoint_filename, 'w') as f:
        json.dump(checkpoint, f, indent=2)

    print(f"Checkpoint saved to {checkpoint_filename}")

print(f"\n{'=' * 80}")
print("Optimisation complete!")
print(f"Best loss: {best_loss:.4f}")
print(f"Best parameters: {[round(v, 4) for v in best_params.tolist()]}")
print(f"Total evaluations: {cma.counteval}")