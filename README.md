# AI Portfolio Management System

An end-to-end portfolio analytics and decision-support system for a diversified portfolio of Indian equities. The project combines classical portfolio optimization, statistical risk analysis, machine-learning-based market regime detection, out-of-sample backtesting, benchmark comparison, transaction-cost analysis, Monte Carlo simulation, and an interpretable recommendation engine. The system is exposed through an interactive Streamlit dashboard.

## Overview

The objective is to compare portfolio construction strategies under a common out-of-sample framework and evaluate them using multiple performance and risk perspectives rather than a single metric.

```text
Historical Stock Data
        ↓
Data Cleaning & Return Construction
        ↓
Exploratory Data Analysis
        ↓
Risk Analysis
        ↓
Portfolio Optimization
 ┌──────┼───────────────┐
 ↓      ↓               ↓
Max    Min          Risk Parity
Sharpe  Variance
 └──────┬───────────────┘
        ↓
Out-of-Sample Backtesting
        ↓
K-Means Market Regime Detection
        ↓
Regime-Based Strategy Selection
        ↓
Walk-Forward Evaluation
        ↓
NIFTY 50 Benchmark Comparison
        ↓
Transaction-Cost Analysis
        ↓
Monte Carlo Simulation
        ↓
Integrated Recommendation Engine
        ↓
Streamlit Dashboard
```

## Key Features

- Historical price ingestion using Yahoo Finance data
- Daily return construction and exploratory analysis
- Risk metrics including volatility, Sharpe ratio, Value at Risk (VaR), and maximum drawdown
- Markowitz Maximum Sharpe portfolio
- Markowitz Minimum Variance portfolio
- Risk Parity portfolio optimization
- Efficient frontier analysis
- Chronological 80/20 train-test evaluation
- K-Means market regime detection using rolling return, volatility, and correlation features
- Regime-based adaptive strategy selection
- Walk-forward evaluation
- NIFTY 50 benchmark comparison
- Transaction-cost analysis and sensitivity analysis
- Bootstrap Monte Carlo simulation with 10,000 paths
- Monte Carlo analysis of the regime-switching strategy
- Interpretable multi-criteria portfolio recommendation
- Interactive Streamlit dashboard

## Assets

The portfolio contains eight Indian equities:

| Ticker | Company |
|---|---|
| RELIANCE | Reliance Industries |
| TCS | Tata Consultancy Services |
| INFY | Infosys |
| HDFCBANK | HDFC Bank |
| ICICIBANK | ICICI Bank |
| SBIN | State Bank of India |
| LT | Larsen & Toubro |
| ITC | ITC |

## Data

The main backtesting sample spans:

- Training period: **2024-01-02 to 2026-01-21**
- Test period: **2026-01-22 to 2026-07-28**
- Total observations: **637**
- Training observations: **509**
- Test observations: **128**

Portfolio parameters are estimated using training data and evaluated on the chronologically subsequent test period.

## Methodology

### 1. Return Construction

Portfolio analysis is performed using daily returns rather than raw price levels. The return matrix is used for estimating expected returns, covariance, portfolio risk, and performance metrics.

### 2. Risk Analysis

The risk engine calculates:

- Total return
- Annualized return / CAGR
- Annualized volatility
- Sharpe ratio
- Maximum drawdown
- VaR at the 95% level

The project uses an annual risk-free rate of **6.5%** and **252 trading days** for annualization.

### 3. Markowitz Optimization

Two mean-variance portfolios are implemented.

#### Maximum Sharpe

The objective is to maximize risk-adjusted expected return:

\[
\max_w \frac{w^\top\mu-r_f}{\sqrt{w^\top\Sigma w}}
\]

subject to portfolio constraints such as full investment and long-only weights.

#### Minimum Variance

The objective is:

\[
\min_w w^\top\Sigma w
\]

under the corresponding portfolio constraints.

### 4. Risk Parity

Risk Parity attempts to balance **risk contributions**, rather than assigning equal capital to every asset.

For asset \(i\), component risk contribution is based on:

\[
RC_i =
\frac{w_i(\Sigma w)_i}
{\sqrt{w^\top\Sigma w}}
\]

The optimizer seeks approximately equal risk contributions across assets.

### 5. Out-of-Sample Backtesting

The project uses a chronological **80/20 split**. Portfolio weights are estimated from the training period and then applied to the test period.

This helps reduce look-ahead bias compared with randomly shuffling financial time-series observations.

### 6. Market Regime Detection

K-Means clustering is used as the machine-learning component of the system.

Regime features are based on:

- Rolling return
- Rolling volatility
- Rolling correlation

A 20-observation rolling window is used. For each day, rolling features are based on prior observations so that the current day's return is excluded.

The scaler and K-Means model are fitted on training-period features only. Test observations are assigned to the learned clusters.

The resulting regimes are interpreted as:

- **Calm**
- **Defensive**
- **High Volatility**

### 7. Regime-Based Strategy Selection

The regime layer selects among portfolio strategies using strategy performance observed in the training period for each corresponding regime.

An important empirical result is that regime switching did **not** consistently outperform the fixed strategies in the evaluated sample. This is treated as a research finding rather than assuming adaptive allocation must improve performance.

### 8. Walk-Forward Evaluation

An expanding-window walk-forward framework provides an additional robustness check beyond the single 80/20 holdout.

### 9. Benchmark Comparison

Portfolio performance is compared with the **NIFTY 50** over common test dates.

The benchmark provides a broad Indian equity market reference for evaluating return and risk.

### 10. Transaction Costs

A configurable proportional transaction-cost assumption is incorporated.

Baseline:

\[
c = 0.10\%
\]

per unit turnover.

Sensitivity analysis is performed at:

- 0.00%
- 0.05%
- 0.10%
- 0.15%
- 0.20%

Fixed portfolios are treated as initially established portfolios, while the regime-switching strategy incurs additional turnover when the selected underlying strategy changes.

### 11. Monte Carlo Simulation

Bootstrap Monte Carlo simulation is used to characterize possible one-year portfolio outcomes.

For the fixed strategies:

- 10,000 simulated paths
- 252 trading days per path
- Starting wealth = 100
- Historical training-period strategy returns sampled with replacement

Reported outputs include:

- Expected terminal wealth
- Median terminal wealth
- 5th and 95th percentile terminal wealth
- Probability of loss
- Probability of loss greater than 10%
- Maximum drawdown statistics

A separate bootstrap analysis is also performed for the risk-aware regime-switching strategy using its realized test-period return distribution.

### 12. Integrated Recommendation Engine

The final decision layer combines evidence from five categories:

| Evidence Category | Weight |
|---|---:|
| Out-of-Sample Backtest | 25% |
| Benchmark Comparison | 15% |
| Transaction Costs | 15% |
| Monte Carlo Simulation | 25% |
| Downside Risk | 20% |

Each component is normalized to a common scoring scale and combined into an interpretable integrated score.

The recommendation engine is a **multi-criteria decision-support model**, not a black-box machine-learning predictor.

## Key Results from the Evaluated Sample

### Out-of-Sample Test Period

Approximate gross total returns over the main holdout period:

| Strategy | Total Return | Annualized Volatility | Sharpe Ratio | Maximum Drawdown |
|---|---:|---:|---:|---:|
| Max Sharpe | +2.22% | 24.34% | -0.086 | -18.90% |
| Min Variance | -10.90% | 16.90% | -1.588 | -17.13% |
| Risk Parity | -10.24% | 17.01% | -1.508 | -15.39% |
| Risk-Aware Regime Switching | -2.01% | 23.48% | -0.444 | -21.07% |

These results describe the specific test sample and are not forecasts of future returns.

### NIFTY 50 Comparison

Over the common test dates, the NIFTY 50 produced approximately:

- Total return: **-4.66%**
- Annualized volatility: **16.72%**
- Sharpe ratio: **-0.937**
- Maximum drawdown: **-13.96%**

### Transaction Costs

At the 0.10% baseline assumption:

| Strategy | Gross Total Return | Net Total Return | Total Turnover |
|---|---:|---:|---:|
| Max Sharpe | +2.217% | +2.116% | 1.00 |
| Min Variance | -10.901% | -10.990% | 1.00 |
| Risk Parity | -10.236% | -10.326% | 1.00 |
| Regime Switching | -2.008% | -3.086% | 11.00 |

The adaptive regime strategy is more sensitive to transaction costs because strategy changes introduce additional turnover.

### Monte Carlo Results

Bootstrap simulation results for the fixed strategies:

| Strategy | Expected Terminal Wealth | 5th Percentile | 95th Percentile | Probability of Loss | Expected Max Drawdown |
|---|---:|---:|---:|---:|---:|
| Max Sharpe | 128.61 | 90.10 | 174.06 | 12.72% | -14.79% |
| Min Variance | 102.04 | 82.44 | 124.30 | 46.16% | -12.77% |
| Risk Parity | 106.44 | 84.92 | 130.90 | 33.67% | -12.09% |

Regime-switching Monte Carlo result:

- Expected terminal wealth: **98.94**
- Median terminal wealth: **96.28**
- 5th percentile: **65.37**
- 95th percentile: **141.59**
- Probability of loss: **56.46%**
- Probability of loss greater than 10%: **38.99%**
- Expected maximum drawdown: **-24.72%**

These are bootstrap-based scenario distributions, not deterministic forecasts.

### Integrated Recommendation

Using the integrated decision framework, the current evaluated sample produces:

| Strategy | Final Integrated Score |
|---|---:|
| Max Sharpe | 0.8319 |
| Risk Parity | 0.4988 |
| Min Variance | 0.3938 |
| Risk-Aware Regime Switching | 0.2741 |

The current decision-support output selects **Max Sharpe** under the specified scoring weights and evidence set.

This is the output of the stated decision framework, not a guarantee of future performance.

## Streamlit Dashboard

The project includes an interactive Streamlit dashboard with:

```text
Overview
Market & Risk
Portfolio Optimization
Backtesting
Benchmark
Regime Analysis
Transaction Costs
Monte Carlo
AI Recommendation
```

Run the dashboard with:

```powershell
python -m streamlit run app.py
```

## Installation

Clone the repository:

```powershell
git clone https://github.com/Joyeeta-goswami/AI-Portfolio-Management-System.git
cd AI-Portfolio-Management-System
```

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\activate
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Suggested Execution Order

```text
1. Data ingestion
2. Risk / exploratory analysis
3. Portfolio optimization
4. Backtesting
5. Regime detection
6. Regime strategy analysis
7. Walk-forward evaluation
8. Benchmark comparison
9. Transaction-cost analysis
10. Monte Carlo simulation
11. Monte Carlo visualization
12. Regime-switching Monte Carlo
13. Final recommendation engine
14. Streamlit dashboard
```

The dashboard primarily reads generated outputs from `data/processed/`.

## Project Structure

```text
AI-Portfolio-Management-System/
│
├── ai_engine/
│   ├── final_recommendation.py
│   ├── monte_carlo.py
│   ├── monte_carlo_regime.py
│   ├── regime_detection.py
│   ├── regime_strategy.py
│   └── walk_forward.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── data_ingestion/
│   ├── benchmark_fetcher.py
│   └── yahoo_fetcher.py
│
├── database/
│   └── data_loader.py
│
├── metrics/
│   ├── benchmark_adjusted_comparison.py
│   ├── benchmark_comparison.py
│   ├── risk_engine.py
│   └── transaction_cost_analysis.py
│
├── notebooks/
│   ├── 01_exploration.ipynb
│   └── 02_portfolio_analysis.ipynb
│
├── optimization/
│   ├── backtest.py
│   ├── markowitz.py
│   └── risk_parity.py
│
├── visualization/
│   ├── monte_carlo_plots.py
│   └── outputs/
│
├── app.py
├── config/
│   └── config.yaml
├── requirements.txt
└── README.md
```

## Limitations

- The asset universe contains eight stocks rather than the full Indian equity market.
- Historical expected returns and covariance estimates may be unstable.
- The sample period is limited relative to long-horizon market history.
- Transaction costs are represented using a simplified proportional model.
- Bootstrap Monte Carlo assumes the empirical historical return distribution is informative for scenario generation.
- Regime identification and regime-specific strategy selection can be sample-sensitive.
- The regime-switching layer did not consistently outperform the fixed strategies in the evaluated sample.
- The recommendation engine depends on the selected criterion weights and should therefore be treated as decision support rather than an objective universal ranking.

## Future Scope

- Expand the asset universe and sector coverage
- Use alternative covariance estimators and shrinkage methods
- Add conditional or volatility-aware optimization
- Apply stricter nested validation for adaptive strategy selection
- Model security-level turnover and trading costs more realistically
- Generate regime-specific Monte Carlo scenarios
- Add Expected Shortfall and other tail-risk measures
- Evaluate additional machine-learning allocation approaches
- Integrate real-time market data
- Add user-specific portfolio constraints
- Deploy and monitor the application in the cloud

## Research Interpretation

The project evaluates whether additional modelling complexity improves portfolio decision-making. It compares traditional portfolio optimization with risk-balanced allocation and an adaptive regime-based strategy while accounting for benchmark performance, implementation costs, and simulated outcome distributions.

A central methodological principle is to distinguish historical evidence, out-of-sample evaluation, scenario simulation, and decision support rather than treating any component as a guaranteed predictor of future market behavior.

## Disclaimer

This project is intended for academic, research, and educational purposes. The outputs are model-based analyses and should not be interpreted as personalized investment advice or guarantees of future returns.

## Author

**Joyeeta Goswami**  
M.Sc. Statistics

GitHub: https://github.com/Joyeeta-goswami/AI-Portfolio-Management-System
