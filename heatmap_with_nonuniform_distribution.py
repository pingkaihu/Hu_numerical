import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def plot_2D_heatmap(csv_path=None, df=None, grid_resolution=500, method='cubic', cmap='viridis'):
    """
    Plot a 2D heatmap from a CSV with columns (x, y, value).

    Non-uniformly distributed (x, y) points are interpolated onto a fine
    regular grid using scipy.interpolate.griddata, enabling subpixel rendering.

    Parameters
    ----------
    csv_path : str, optional
        Path to CSV file with columns x, y, value.
    df : pd.DataFrame, optional
        DataFrame with columns x, y, value (used instead of csv_path).
    grid_resolution : int
        Number of grid pixels along each axis (higher = more subpixel detail).
    method : str
        Interpolation method: 'linear', 'nearest', or 'cubic'.
    cmap : str
        Matplotlib colormap name.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    if df is None:
        df = pd.read_csv(csv_path)

    x = df['x'].values
    y = df['y'].values
    v = df['value'].values

    # Build a fine regular grid for subpixel interpolation
    xi = np.linspace(x.min(), x.max(), grid_resolution)
    yi = np.linspace(y.min(), y.max(), grid_resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    # Interpolate scattered data onto the regular grid
    zi = griddata((x, y), v, (xi_grid, yi_grid), method=method)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(
        zi,
        extent=[x.min(), x.max(), y.min(), y.max()],
        origin='lower',
        aspect='auto',
        cmap=cmap,
        interpolation='bicubic',
        #interpolation='bilinear',   # subpixel anti-aliasing from matplotlib
    )

    # Overlay original sample locations
    ax.scatter(x, y, c='white', s=8, alpha=0.6, linewidths=0.4,
               edgecolors='gray', label=f'Data points (n={len(x)})')

    plt.colorbar(im, ax=ax, label='Value')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'2D Heatmap  |  interpolation={method!r}  |  grid={grid_resolution}×{grid_resolution}')
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.show()

    return fig, ax


# ---------------------------------------------------------------------------
# Test case
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    rng = np.random.default_rng(42)
    n = 144

    x = rng.uniform(0, 10, n)
    y = rng.uniform(0, 10, n)
    value = rng.normal(loc=1.0, scale=0.1, size=n)

    df_test = pd.DataFrame({'x': x, 'y': y, 'value': value})
    df_test.to_csv('test_data.csv', index=False)

    plot_2D_heatmap(df=df_test, grid_resolution=1000, method='cubic', cmap='jet')
