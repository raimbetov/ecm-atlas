import json
from pathlib import Path
import numpy as np
import pandas as pd
import ruptures as rpt
from matplotlib import pyplot as plt

VELOCITY_PATH = Path('../../../iteration_01/hypothesis_03_tissue_aging_clocks/codex/tissue_aging_velocity_codex.csv').resolve()
OUTPUT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = OUTPUT_DIR / 'changepoint_results_codex.csv'
POSTERIOR_PATH = OUTPUT_DIR / 'changepoint_posterior_codex.csv'
PLOT_PATH = OUTPUT_DIR / 'visualizations_codex/changepoint_plot_codex.png'
SUMMARY_JSON = OUTPUT_DIR / 'changepoint_summary_codex.json'
THRESHOLD = 1.65


def load_velocities():
    df = pd.read_csv(VELOCITY_PATH)
    df = df.sort_values('Velocity').reset_index(drop=True)
    return df


def binary_segmentation(signal):
    algo = rpt.Binseg(model='l2').fit(signal)
    bkps = algo.predict(n_bkps=1)
    return bkps[0] - 1  # convert to zero-based index


def pelt_method(signal):
    algo = rpt.Pelt(model='l2').fit(signal)
    bkps = algo.predict(pen=1.0)
    return bkps[0] - 1


def grid_search(signal):
    # Evaluate sum of squared errors for every split
    best_idx = None
    best_score = np.inf
    scores = []
    for idx in range(1, len(signal) - 1):
        left = signal[:idx]
        right = signal[idx:]
        left_mu = np.mean(left)
        right_mu = np.mean(right)
        sse = np.sum((left - left_mu) ** 2) + np.sum((right - right_mu) ** 2)
        scores.append({'Index': idx, 'Velocity': signal[idx], 'SSE': sse})
        if sse < best_score:
            best_score = sse
            best_idx = idx
    return best_idx, pd.DataFrame(scores)


def bayesian_posterior(signal):
    # Simple Bayesian odds using normal-inverse-gamma conjugacy with uniform prior on split
    n = len(signal)
    k_candidates = np.arange(1, n - 1)
    posteriors = []
    for k in k_candidates:
        left = signal[:k]
        right = signal[k:]
        left_var = np.var(left, ddof=1) if len(left) > 1 else 1.0
        right_var = np.var(right, ddof=1) if len(right) > 1 else 1.0
        pooled = np.var(signal, ddof=1)
        likelihood_ratio = pooled / ((left_var + right_var) / 2.0 + 1e-8)
        posteriors.append(likelihood_ratio)
    posteriors = np.array(posteriors)
    posteriors = posteriors / posteriors.sum()
    return k_candidates, posteriors


def main():
    df = load_velocities()
    signal = df['Velocity'].values

    binseg_idx = binary_segmentation(signal)
    pelt_idx = pelt_method(signal)
    grid_idx, grid_scores = grid_search(signal)
    bayes_idx_candidates, posterior = bayesian_posterior(signal)
    posterior_df = pd.DataFrame({
        'Candidate_Index': bayes_idx_candidates,
        'Velocity': df.loc[bayes_idx_candidates, 'Velocity'].values,
        'Posterior_Weight': posterior
    })
    posterior_df.to_csv(POSTERIOR_PATH, index=False)

    best_idx = int(np.mean([binseg_idx, pelt_idx, grid_idx]))
    best_velocity = df.loc[best_idx, 'Velocity']

    results = pd.DataFrame([
        {'Method': 'BinarySegmentation', 'Index': binseg_idx, 'Velocity': df.loc[binseg_idx, 'Velocity']},
        {'Method': 'PELT', 'Index': pelt_idx, 'Velocity': df.loc[pelt_idx, 'Velocity']},
        {'Method': 'GridSearch', 'Index': grid_idx, 'Velocity': df.loc[grid_idx, 'Velocity']},
        {'Method': 'Consensus', 'Index': best_idx, 'Velocity': best_velocity},
        {'Method': 'H09_Threshold', 'Index': np.argmin(np.abs(df['Velocity'] - THRESHOLD)), 'Velocity': THRESHOLD}
    ])
    results.to_csv(RESULTS_PATH, index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(df['Velocity'].values, marker='o')
    plt.axhline(THRESHOLD, color='red', linestyle='--', label='v=1.65 reference')
    plt.axvline(best_idx, color='black', linestyle='-', label='Consensus changepoint')
    plt.xlabel('Ordered tissues')
    plt.ylabel('Velocity')
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=300)
    plt.close()

    summary = {
        'binary_segmentation_velocity': float(df.loc[binseg_idx, 'Velocity']),
        'pelt_velocity': float(df.loc[pelt_idx, 'Velocity']),
        'grid_velocity': float(df.loc[grid_idx, 'Velocity']),
        'posterior_max_velocity': float(df.loc[bayes_idx_candidates[np.argmax(posterior)], 'Velocity']),
        'consensus_velocity': float(best_velocity)
    }
    with open(SUMMARY_JSON, 'w') as f:
        json.dump(summary, f, indent=2)

    grid_scores.to_csv(OUTPUT_DIR / 'changepoint_grid_scores_codex.csv', index=False)


if __name__ == '__main__':
    main()
