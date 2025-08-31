# main_experiment.py

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize

# Import our custom functions from other files
from utils import generate_stock_data, calculate_cvar
from plotting import create_academic_plot

# --------------------------------------------------------------------------
# MODEL TRAINING FUNCTIONS
# --------------------------------------------------------------------------

def train_ols_model(X_train, y_train):
    X_train_const = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_train_const).fit()
    return model

def train_quantile_model(X_train, y_train, quantile=0.01):
    X_train_const = sm.add_constant(X_train)
    model = sm.QuantReg(y_train, X_train_const).fit(q=quantile)
    return model

def train_cvar_norm_model(X_train, y_train, alpha=0.99):
    X_train_const = np.c_[np.ones(X_train.shape[0]), X_train]
    
    def cvar_norm_objective(beta, X, y, alpha):
        residuals = y - X @ beta
        abs_residuals = np.abs(residuals)
        
        def cvar_formula(C):
            return C + (1 / (1 - alpha)) * np.mean(np.maximum(0, abs_residuals - C))
            
        result = minimize(cvar_formula, x0=np.mean(abs_residuals), method='Nelder-Mead')
        return result.fun

    initial_beta = np.linalg.lstsq(X_train_const, y_train, rcond=None)[0]
    result = minimize(cvar_norm_objective, x0=initial_beta, args=(X_train_const, y_train, alpha), method='Nelder-Mead')
    
    return result.x

# --------------------------------------------------------------------------
# MONTE CARLO SIMULATION
# --------------------------------------------------------------------------

def run_simulation(num_simulations=500, alpha_level=0.99):
    """Runs the full Monte Carlo simulation and returns the averaged results."""
    results_list = []
    
    for i in range(num_simulations):
        if (i + 1) % 50 == 0:
            print(f"Running simulation {i+1}/{num_simulations}...")
        
        data = generate_stock_data()
        X = data[['Asset1']]
        y = data['Asset2']

        train_size = int(len(data) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        ols_model = train_ols_model(X_train, y_train)
        quant_model = train_quantile_model(X_train, y_train, quantile=(1-alpha_level))
        cvar_beta = train_cvar_norm_model(X_train, y_train, alpha=alpha_level)

        X_test_const = sm.add_constant(X_test)
        ols_predictions = ols_model.predict(X_test_const)
        quant_predictions = quant_model.predict(X_test_const)
        cvar_predictions = X_test_const @ cvar_beta

        ols_errors = y_test - ols_predictions
        quant_errors = y_test - quant_predictions
        cvar_errors = y_test - cvar_predictions

        run_results = {
            'mae_ols': np.mean(np.abs(ols_errors)), 'rmse_ols': np.sqrt(np.mean(ols_errors**2)),
            'cvar_errors_ols': calculate_cvar(np.abs(ols_errors), alpha=alpha_level),
            'mae_quant': np.mean(np.abs(quant_errors)), 'rmse_quant': np.sqrt(np.mean(quant_errors**2)),
            'cvar_errors_quant': calculate_cvar(np.abs(quant_errors), alpha=alpha_level),
            'mae_cvar': np.mean(np.abs(cvar_errors)), 'rmse_cvar': np.sqrt(np.mean(cvar_errors**2)),
            'cvar_errors_cvar': calculate_cvar(np.abs(cvar_errors), alpha=alpha_level),
        }
        results_list.append(run_results)
        
    # Return all necessary data for plotting and printing results
    return pd.DataFrame(results_list).mean(), (X, y, X_test, ols_predictions, quant_predictions, cvar_predictions)

def print_results(average_results, alpha_level):
    """Prints the final formatted table of results."""
    print("\n" + "="*60)
    print("      AVERAGED MODEL PERFORMANCE (OUT-OF-SAMPLE)")
    print("="*60)
    print(f"{'Metric':<25} | {'OLS (L2)':>10} | {'Quantile':>10} | {'CVaR-norm':>10}")
    print("-"*60)
    print(f"{'MAE':<25} | {average_results['mae_ols']:10.4f} | {average_results['mae_quant']:10.4f} | {average_results['mae_cvar']:10.4f}")
    print(f"{'RMSE':<25} | {average_results['rmse_ols']:10.4f} | {average_results['rmse_quant']:10.4f} | {average_results['rmse_cvar']:10.4f}")
    print(f"{f'CVaR of Errors (alpha={alpha_level})':<25} | {average_results['cvar_errors_ols']:10.4f} | {average_results['cvar_errors_quant']:10.4f} | {average_results['cvar_errors_cvar']:10.4f}")
    print("="*60)

if __name__ == '__main__':
    ALPHA_LEVEL = 0.99
    
    # Run the entire simulation
    avg_results, plot_data = run_simulation(num_simulations=500, alpha_level=ALPHA_LEVEL)
    
    # Print the final results table
    print_results(avg_results, ALPHA_LEVEL)
    
    # Create and save the plot from the last simulation run
    X, y, X_test, ols_preds, quant_preds, cvar_preds = plot_data
    create_academic_plot(X, y, X_test, ols_preds, quant_preds, cvar_preds, ALPHA_LEVEL)
