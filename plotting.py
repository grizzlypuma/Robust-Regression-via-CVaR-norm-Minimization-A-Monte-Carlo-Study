# plotting.py

import matplotlib.pyplot as plt

def create_academic_plot(X, y, X_test, ols_preds, quant_preds, cvar_preds, ALPHA_LEVEL, filename="academic_regression_plot.pdf"):
    """
    Creates and saves a plot in an academic style.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Computer Modern', 'Times New Roman'], 'font.size': 12,
        'axes.labelsize': 14, 'axes.titlesize': 16, 'xtick.labelsize': 12,
        'ytick.labelsize': 12, 'legend.fontsize': 12,
    })
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, alpha=0.3, s=20, label='Simulated Data', color='gray')
    ax.plot(X_test, ols_preds, color='#d62728', linewidth=2, linestyle='-', label='OLS Regression (L2)')
    ax.plot(X_test, quant_preds, color='#2ca02c', linewidth=2, linestyle='--', label=f'Quantile Regression (q={1-ALPHA_LEVEL:.2f})')
    ax.plot(X_test, cvar_preds, color='#9467bd', linewidth=2, linestyle=':', label=f'CVaR-norm Regression (α={ALPHA_LEVEL:.2f})')
    ax.set_title('Comparison of Regression Models on Data with Outliers')
    ax.set_xlabel('Asset 1 Return (Factor)')
    ax.set_ylabel('Asset 2 Return (Target)')
    ax.legend(loc='best', frameon=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as {filename}")
    plt.show()
