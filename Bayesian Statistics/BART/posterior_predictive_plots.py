
import numpy as np
import matplotlib.pyplot as plt 
def plot_posterior_predictive_distribution(idata, n_rows=4, random_seed=None,figsize=(12, 8), bins=50):
    """
    Plot posterior predictive distributions for randomly selected rows of data.
    Shows the distribution of predicted y values for each selected row, with 
    the observed value as a vertical line.
    
    Args:
    ----
    idata : arviz.InferenceData
        The inference data object returned by pm.sample()
    n_rows : int, default 4
        Number of rows to randomly select and plot
    random_seed : int, optional
        Random seed for reproducibility
    figsize : tuple, default (12, 8)
        Figure size
    bins : int, default 50
        Number of bins for histograms
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    selected_indices : list
        Indices of the selected rows
    """
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n_total_rows = idata['observed_data'].y.shape[0]
    selected_indices = np.random.choice(n_total_rows, size=n_rows, replace=False)
    y_observed = idata['observed_data'].y.values

    # Extract posterior samples
    posteriors = [idata.posterior_predictive['y'].stack(
    sample=("chain", "draw")).sel(obs_id=i) for i in selected_indices]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    for i, row_idx in enumerate(selected_indices):
        ax = axes[i]
    
        ax.hist(posteriors[i], bins=bins, density=True, alpha=0.7, 
                color= "#2E86AB", edgecolor='black', linewidth=0.5)
        
        ax.axvline(y_observed[row_idx], color='red', linestyle='-', linewidth=3, 
                  label=f'Observed: {y_observed[row_idx]:.2f}')

        mean_pred = np.mean(posteriors[i])
        ax.axvline(mean_pred, color='black', linestyle='--', linewidth=2, 
                  label=f'Mean pred: {mean_pred:.2f}')

        ci_lower = np.percentile(posteriors[i], 2.5)
        ci_upper = np.percentile(posteriors[i], 97.5)
        ax.axvspan(ci_lower, ci_upper, alpha=0.4, color='#457B9D', 
                  label=f'95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]')
        
        # Formatting
        ax.set_xlabel('Predicted y')
        ax.set_ylabel('Density')
        ax.set_title(f'Row {row_idx}: Posterior Predictive Distribution')
        ax.legend(framealpha=0, loc='upper left')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, selected_indices


def plot_posterior_predictive_comparison(
    idata,
    row_indices,
    random_seed=None,
    figsize=(15, 4),
    bins=50
):
    """
    Plot posterior predictive distributions for specified rows side-by-side.

    Parameters
    ----------
    idata : arviz.InferenceData
        The inference data object returned by pm.sample() and pm.sample_posterior_predictive().
    row_indices : list of int
        List of observation indices to plot.
    random_seed : int, optional
        Seed for reproducible selection (not strictly needed here, but kept for symmetry).
    figsize : tuple, default (15, 4)
        Figure size.
    bins : int, default 50
        Number of bins for each histogram.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the subplots.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Pull out observed y and posterior_predictive samples
    y_obs = idata.observed_data["y"].values
    ppc = (
        idata
        .posterior_predictive["y"]
        .stack(sample=("chain", "draw"))
    )

    n = len(row_indices)
    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
    axes = axes[0]

    for ax, idx in zip(axes, row_indices):
        # get predictive draws for this observation
        draws = ppc.sel(obs_id=idx).values

        # histogram
        ax.hist(draws, bins=bins, density=True,
                alpha=0.7, color="#2E86AB", edgecolor="black", lw=0.5)

        # observed value
        obs = y_obs[idx]
        ax.axvline(obs, color="red", lw=3, label=f"Observed: {obs:.2f}")

        # posterior predictive mean
        mean_pred = draws.mean()
        ax.axvline(mean_pred, color="black", ls="--", lw=2,
                   label=f"Mean pred: {mean_pred:.2f}")

        # 95% CI
        lo, hi = np.percentile(draws, [2.5, 97.5])
        ax.axvspan(lo, hi, color="#457B9D", alpha=0.4,
                   label=f"95% CI: [{lo:.2f}, {hi:.2f}]")

        ax.set_title(f"Row {idx}")
        ax.set_xlabel("Predicted y")
        ax.set_ylabel("Density")
        ax.legend(framealpha=0, loc="upper left")
        ax.grid(alpha=0.3)

    plt.tight_layout()