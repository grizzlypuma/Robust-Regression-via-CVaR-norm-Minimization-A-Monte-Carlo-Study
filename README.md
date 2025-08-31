Robust Regression via CVaR-norm Minimization: A Monte Carlo Study
A Python-based numerical experiment comparing the out-of-sample performance of OLS, Quantile, and CVaR-norm regression models on simulated financial data with outliers. This project is the basis for an academic pre-print and demonstrates the practical implications of the risk-coherence principle from the Fundamental Risk Quadrangle framework.

Core Idea
Standard Ordinary Least Squares (OLS) regression is highly sensitive to outliers, a common feature of financial data. While robust methods like Quantile Regression exist, they are designed to model a specific quantile of the distribution, not necessarily to control the magnitude of extreme errors.

This project investigates an alternative approach: a regression model that directly minimizes the Conditional Value-at-Risk (CVaR) of the absolute errors (CVaR-norm). The central hypothesis is that this model, while potentially less accurate on average than OLS, will provide superior performance in controlling the tail risk of its own prediction errors, making it a more robust tool for risk-centric applications.

This study implements and tests this hypothesis through a Monte Carlo simulation.

Technologies Used
Python 3.10+

NumPy

Pandas

Matplotlib

SciPy

Statsmodels

How to Run the Experiment
To replicate the results, please follow these steps:

1. Clone the repository: git clone https://github.com/your-username/cvar-norm-regression-study.git cd cvar-norm-regression-study

2. Create and activate a virtual environment (recommended): python -m venv venv
# On macOS/Linux
source venv/bin/activate
# On Windows
.\venv\Scripts\activate

3. Install the required packages: pip install -r requirements.txt

4. python main_experiment.py

The script will run a 500-iteration Monte Carlo simulation, print the final averaged results to the console, and save the academic-style plot as academic_regression_plot.pdf.

Конечно. Вот готовый README.md файл для вашего GitHub репозитория.

Вам нужно просто скопировать весь текст ниже и вставить его в файл README.md в вашем проекте.

Robust Regression via CVaR-norm Minimization: A Monte Carlo Study
A Python-based numerical experiment comparing the out-of-sample performance of OLS, Quantile, and CVaR-norm regression models on simulated financial data with outliers. This project is the basis for an academic pre-print and demonstrates the practical implications of the risk-coherence principle from the Fundamental Risk Quadrangle framework.

Core Idea
Standard Ordinary Least Squares (OLS) regression is highly sensitive to outliers, a common feature of financial data. While robust methods like Quantile Regression exist, they are designed to model a specific quantile of the distribution, not necessarily to control the magnitude of extreme errors.

This project investigates an alternative approach: a regression model that directly minimizes the Conditional Value-at-Risk (CVaR) of the absolute errors (CVaR-norm). The central hypothesis is that this model, while potentially less accurate on average than OLS, will provide superior performance in controlling the tail risk of its own prediction errors, making it a more robust tool for risk-centric applications.

This study implements and tests this hypothesis through a Monte Carlo simulation.

Technologies Used
Python 3.10+

NumPy

Pandas

Matplotlib

SciPy

Statsmodels

How to Run the Experiment
To replicate the results, please follow these steps:

Clone the repository:

Bash

git clone https://github.com/your-username/cvar-norm-regression-study.git
cd cvar-norm-regression-study
Create and activate a virtual environment (recommended):

Bash

python -m venv venv
# On macOS/Linux
source venv/bin/activate
# On Windows
.\venv\Scripts\activate
Install the required packages:

Bash

pip install -r requirements.txt
Run the main experiment script:

Bash

python main_experiment.py
The script will run a 500-iteration Monte Carlo simulation, print the final averaged results to the console, and save the academic-style plot as academic_regression_plot.pdf.

Averaged Results (500 Simulations)
The out-of-sample performance metrics, averaged over 500 independent runs, are as follows:
Metric	                            OLS (L2)	Quantile (q=0.01)	CVaR-norm (α=0.99)
MAE	                                0.0436	      0.1048	             0.0460
RMSE	                              0.0531	      0.1171	             0.0561
CVaR of Errors (α=0.99)	            0.1028	      0.1934	             0.1001

The results confirm our hypothesis. While OLS is superior on average-case metrics (MAE, RMSE), the CVaR-norm regression model demonstrates the best performance in controlling tail risk, achieving the lowest CVaR of its own errors.
Full Paper
For a detailed mathematical description and a full discussion of the results, please see our pre-print on arXiv:

[Link to be added once submitted]

