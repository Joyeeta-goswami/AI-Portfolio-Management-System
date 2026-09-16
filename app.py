"""
AI Portfolio Management System
--------------------------------
Streamlit dashboard for:
- Portfolio risk analysis
- Markowitz optimization
- Risk Parity
- Backtesting
- Benchmark comparison
- Market regime detection
- Transaction costs
- Monte Carlo simulation
- Integrated portfolio recommendation
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "visualization"
    / "outputs"
)

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Portfolio Manager",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #6b7280;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .metric-card {
        padding: 1rem;
        border-radius: 0.8rem;
        background-color: #f8fafc;
        border: 1px solid #e5e7eb;
    }

    .recommendation-card {
    padding: 1.4rem;
    border-radius: 1rem;
    background-color: #f8fafc;
    border: 1px solid #d1d5db;
    margin-bottom: 1rem;
    color: #111827;
}

    .recommendation-card h1,
    .recommendation-card h2,
    .recommendation-card p {
    color: #111827;
}

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPER: LOAD CSV
# ============================================================

@st.cache_data
def load_csv(filename):

    path = DATA_DIR / filename

    if not path.exists():
        return None

    return pd.read_csv(path)


# ============================================================
# LOAD ALL DATA
# ============================================================

@st.cache_data
def load_all_data():

    data = {}

    files = [
        "prices.csv",
        "returns.csv",
        "risk_summary.csv",
        "efficient_frontier.csv",
        "backtest_results.csv",
        "cumulative_wealth.csv",
        "market_regimes.csv",
        "regime_switching_results.csv",
        "regime_switching_results_v2.csv",
        "regime_strategy_test_returns.csv",
        "walk_forward_results.csv",
        "walk_forward_results_v2.csv",
        "nifty50_returns.csv",
        "benchmark_adjusted_results.csv",
        "benchmark_adjusted_wealth.csv",
        "transaction_cost_results.csv",
        "transaction_cost_sensitivity.csv",
        "monte_carlo_results.csv",
        "monte_carlo_terminal_wealth.csv",
        "monte_carlo_drawdowns.csv",
        "monte_carlo_regime_results.csv",
        "monte_carlo_regime_terminal_wealth.csv",
        "monte_carlo_regime_drawdowns.csv",
        "final_recommendation_results.csv",
        "final_recommendation_summary.csv",
    ]

    for filename in files:

        data[filename] = load_csv(
            filename
        )

    return data


# ============================================================
# FORMAT HELPERS
# ============================================================

def format_pct(value):

    if pd.isna(value):
        return "N/A"

    return f"{value:.2%}"


def format_number(value):

    if pd.isna(value):
        return "N/A"

    return f"{value:.2f}"


# ============================================================
# DATA
# ============================================================

data = load_all_data()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📊 AI Portfolio Manager")

st.sidebar.markdown(
    """
    **M.Sc. Statistics Portfolio Management System**

    Compare portfolio construction strategies,
    analyze market regimes, simulate risk, and
    obtain an integrated portfolio recommendation.
    """
)

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Overview",
        "📈 Market & Risk",
        "⚙️ Portfolio Optimization",
        "🧪 Backtesting",
        "🌍 Benchmark",
        "🔄 Regime Analysis",
        "💰 Transaction Costs",
        "🎲 Monte Carlo",
        "🤖 AI Recommendation",
    ]
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">AI Portfolio Management System</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Statistical portfolio optimization, machine-learning
    regime detection, out-of-sample validation and
    risk-aware decision support.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.header("Portfolio Management Overview")

    recommendation = data.get(
        "final_recommendation_results.csv"
    )

    if recommendation is not None:

        recommended_rows = recommendation[
            recommendation["Recommended"] == True
        ]

        if len(recommended_rows) > 0:

            row = recommended_rows.iloc[0]

            st.markdown(
                f"""
                <div class="recommendation-card">

                <h2>Recommended Strategy</h2>

                <h1>{row['Strategy']}</h1>

                <p>
                Integrated score:
                <b>{row['Final Score']:.4f}</b>
                </p>

                <p>
                Risk profile:
                <b>{row['Risk Profile']}</b>
                </p>

                <p>
                {row['Evidence Summary']}
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

    st.subheader("System Components")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Assets",
            "8"
        )

    with col2:

        st.metric(
            "Strategies",
            "4"
        )

    with col3:

        st.metric(
            "Monte Carlo Paths",
            "10,000"
        )

    with col4:

        st.metric(
            "ML Method",
            "K-Means"
        )

    st.subheader("Portfolio Strategies")

    strategy_description = pd.DataFrame({
        "Strategy": [
            "Max Sharpe",
            "Min Variance",
            "Risk Parity",
            "Risk-Aware Regime Switching"
        ],
        "Purpose": [
            "Risk-adjusted return optimization",
            "Minimum portfolio variance",
            "Balanced contribution to portfolio risk",
            "Adaptive strategy selection using market regimes"
        ]
    })

    st.dataframe(
        strategy_description,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        """
        The recommendation engine is an interpretable
        multi-criteria decision system. The machine-learning
        component of the system is K-Means market-regime
        detection.
        """
    )


# ============================================================
# MARKET & RISK
# ============================================================

elif page == "📈 Market & Risk":

    st.header("Market & Risk Analysis")

    prices = data.get(
        "prices.csv"
    )

    risk_summary = data.get(
        "risk_summary.csv"
    )

    if prices is None:

        st.error(
            "prices.csv was not found."
        )

    else:

        prices = prices.copy()

        date_column = "date"

        prices[date_column] = pd.to_datetime(
            prices[date_column]
        )

        assets = [
            col
            for col in prices.columns
            if col != "date"
        ]

        selected_assets = st.multiselect(
            "Select assets",
            assets,
            default=assets[:4]
        )

        if selected_assets:

            plot_data = prices[
                ["date"] + selected_assets
            ].melt(
                id_vars="date",
                var_name="Asset",
                value_name="Price"
            )

            fig = px.line(
                plot_data,
                x="date",
                y="Price",
                color="Asset",
                title="Historical Asset Prices"
            )

            fig.update_layout(
                height=500
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        if risk_summary is not None:

            st.subheader(
                "Historical Risk Summary"
            )

            st.dataframe(
                risk_summary,
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# PORTFOLIO OPTIMIZATION
# ============================================================

elif page == "⚙️ Portfolio Optimization":

    st.header("Portfolio Optimization")

    risk_summary = data.get(
        "risk_summary.csv"
    )

    efficient_frontier = data.get(
        "efficient_frontier.csv"
    )

    st.subheader(
        "Risk Characteristics of Individual Assets"
    )

    if risk_summary is not None:

        st.dataframe(
            risk_summary,
            use_container_width=True,
            hide_index=True
        )

    st.subheader(
        "Efficient Frontier"
    )

    if efficient_frontier is not None:

        frontier = efficient_frontier.copy()

        fig = px.scatter(
    frontier,
    x="Risk",
    y="Return",
    title="Markowitz Efficient Frontier",
    labels={
        "Risk": "Annualized Portfolio Risk",
        "Return": "Annualized Portfolio Return"
    }
)

        fig.update_layout(
            height=550
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader(
        "Optimization Methodology"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Markowitz",
            "Maximum Sharpe"
        )

    with col2:

        st.metric(
            "Markowitz",
            "Minimum Variance"
        )

    with col3:

        st.metric(
            "Alternative",
            "Risk Parity"
        )

    st.write(
        """
        Markowitz portfolios use expected returns and the
        covariance matrix estimated from the training period.
        Risk Parity allocates capital so that portfolio risk
        contributions are approximately balanced across assets.
        """
    )


# ============================================================
# BACKTESTING
# ============================================================

elif page == "🧪 Backtesting":

    st.header(
        "Out-of-Sample Backtesting"
    )

    backtest = data.get(
        "backtest_results.csv"
    )

    wealth = data.get(
        "cumulative_wealth.csv"
    )

    if backtest is not None:

        st.subheader(
            "Test-Period Performance"
        )

        st.dataframe(
            backtest,
            use_container_width=True,
            hide_index=True
        )

    if wealth is not None:

        wealth = wealth.copy()

        wealth["date"] = pd.to_datetime(
            wealth["date"]
        )

        plot_columns = [
            col
            for col in wealth.columns
            if col != "date"
        ]

        plot_data = wealth[
            ["date"] + plot_columns
        ].melt(
            id_vars="date",
            var_name="Strategy",
            value_name="Wealth"
        )

        fig = px.line(
            plot_data,
            x="date",
            y="Wealth",
            color="Strategy",
            title="Out-of-Sample Cumulative Wealth"
        )

        fig.add_hline(
            y=100,
            line_dash="dash"
        )

        fig.update_layout(
            height=500
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.info(
        """
        The test period is kept completely separate from
        the portfolio optimization stage to evaluate
        out-of-sample performance.
        """
    )


# ============================================================
# BENCHMARK
# ============================================================

elif page == "🌍 Benchmark":

    st.header(
        "NIFTY 50 Benchmark Comparison"
    )

    benchmark = data.get(
        "benchmark_adjusted_results.csv"
    )

    wealth = data.get(
        "benchmark_adjusted_wealth.csv"
    )

    if benchmark is not None:

        st.subheader(
            "Common-Date Benchmark Comparison"
        )

        st.dataframe(
            benchmark,
            use_container_width=True,
            hide_index=True
        )

    if wealth is not None:

        wealth = wealth.copy()

        wealth["date"] = pd.to_datetime(
            wealth["date"]
        )

        columns = [
            col
            for col in wealth.columns
            if col != "date"
        ]

        plot_data = wealth[
            ["date"] + columns
        ].melt(
            id_vars="date",
            var_name="Strategy",
            value_name="Wealth"
        )

        fig = px.line(
            plot_data,
            x="date",
            y="Wealth",
            color="Strategy",
            title="Portfolio vs NIFTY 50"
        )

        fig.add_hline(
            y=100,
            line_dash="dash"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# REGIME ANALYSIS
# ============================================================

elif page == "🔄 Regime Analysis":

    st.header(
        "Market Regime Detection"
    )

    regimes = data.get(
        "market_regimes.csv"
    )

    regime_returns = data.get(
        "regime_strategy_test_returns.csv"
    )

    if regimes is not None:

        regimes = regimes.copy()

        regimes["date"] = pd.to_datetime(
            regimes["date"]
        )

        st.subheader(
            "Detected Market Regimes"
        )

        if "regime" in regimes.columns:

            regime_counts = (
                regimes["regime"]
                .value_counts()
                .reset_index()
            )

            regime_counts.columns = [
                "Regime",
                "Observations"
            ]

            fig = px.bar(
                regime_counts,
                x="Regime",
                y="Observations",
                title="Regime Frequency"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.dataframe(
            regimes.tail(20),
            use_container_width=True,
            hide_index=True
        )

    if regime_returns is not None:

        st.subheader(
            "Regime Strategy Selection"
        )

        selection_counts = (
            regime_returns[
                "Selected Strategy"
            ]
            .value_counts()
            .reset_index()
        )

        selection_counts.columns = [
            "Strategy",
            "Observations"
        ]

        fig = px.bar(
            selection_counts,
            x="Strategy",
            y="Observations",
            title="Selected Strategy During Test Period"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# TRANSACTION COSTS
# ============================================================

elif page == "💰 Transaction Costs":

    st.header(
        "Transaction Cost Analysis"
    )

    transaction = data.get(
        "transaction_cost_results.csv"
    )

    sensitivity = data.get(
        "transaction_cost_sensitivity.csv"
    )

    if transaction is not None:

        st.subheader(
            "Net Performance After Transaction Costs"
        )

        st.dataframe(
            transaction,
            use_container_width=True,
            hide_index=True
        )

    if sensitivity is not None:

        st.subheader(
            "Transaction Cost Sensitivity"
        )

        selected_strategy = st.selectbox(
            "Select strategy",
            sensitivity[
                "Strategy"
            ].unique()
        )

        selected = sensitivity[
            sensitivity["Strategy"]
            == selected_strategy
        ]

        fig = px.line(
            selected,
            x="Transaction Cost (%)",
            y="Net Total Return",
            markers=True,
            title=(
                "Net Total Return vs "
                "Transaction Cost"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# MONTE CARLO
# ============================================================

elif page == "🎲 Monte Carlo":

    st.header(
        "Monte Carlo Risk Simulation"
    )

    monte_carlo = data.get(
        "monte_carlo_results.csv"
    )

    terminal = data.get(
        "monte_carlo_terminal_wealth.csv"
    )

    drawdowns = data.get(
        "monte_carlo_drawdowns.csv"
    )

    regime_mc = data.get(
        "monte_carlo_regime_results.csv"
    )

    if monte_carlo is not None:

        st.subheader(
            "Fixed Portfolio Simulation"
        )

        st.dataframe(
            monte_carlo,
            use_container_width=True,
            hide_index=True
        )

    if regime_mc is not None:

        st.subheader(
            "Regime-Switching Monte Carlo"
        )

        st.dataframe(
            regime_mc,
            use_container_width=True,
            hide_index=True
        )

    if terminal is not None:

        selected_strategy = st.selectbox(
            "Terminal wealth strategy",
            terminal[
                "Strategy"
            ].unique()
        )

        selected_terminal = terminal[
            terminal["Strategy"]
            == selected_strategy
        ]

        fig = px.histogram(
            selected_terminal,
            x="Terminal Wealth",
            nbins=60,
            title=(
                "Monte Carlo Terminal Wealth "
                "Distribution"
            )
        )

        fig.add_vline(
            x=100,
            line_dash="dash",
            annotation_text="Initial Wealth"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    if drawdowns is not None:

        selected_strategy_dd = st.selectbox(
            "Maximum drawdown strategy",
            drawdowns[
                "Strategy"
            ].unique()
        )

        selected_drawdowns = drawdowns[
            drawdowns["Strategy"]
            == selected_strategy_dd
        ]

        fig = px.histogram(
            selected_drawdowns,
            x="Maximum Drawdown",
            nbins=60,
            title=(
                "Monte Carlo Maximum "
                "Drawdown Distribution"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# AI RECOMMENDATION
# ============================================================

elif page == "🤖 AI Recommendation":

    st.header(
        "Integrated Portfolio Recommendation"
    )

    recommendation = data.get(
        "final_recommendation_results.csv"
    )

    if recommendation is None:

        st.error(
            "Final recommendation results were not found."
        )

    else:

        recommended = recommendation[
            recommendation["Recommended"] == True
        ]

        if len(recommended) > 0:

            row = recommended.iloc[0]

            st.success(
                f"Recommended Strategy: "
                f"{row['Strategy']}"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Integrated Score",
                    f"{row['Final Score']:.4f}"
                )

            with col2:

                st.metric(
                    "Risk Profile",
                    row["Risk Profile"]
                )

            with col3:

                st.metric(
                    "Recommendation",
                    "Selected"
                )

            st.subheader(
                "Evidence Summary"
            )

            st.write(
                row["Evidence Summary"]
            )

        st.subheader(
            "Strategy Comparison"
        )

        score_columns = [
            "Strategy",
            "Backtest Score",
            "Benchmark Score",
            "Transaction Cost Score",
            "Monte Carlo Score",
            "Downside Risk Score",
            "Final Score"
        ]

        st.dataframe(
            recommendation[
                score_columns
            ],
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # Score visualization
        # ----------------------------------------------------

        score_data = recommendation[
            [
                "Strategy",
                "Backtest Score",
                "Benchmark Score",
                "Transaction Cost Score",
                "Monte Carlo Score",
                "Downside Risk Score"
            ]
        ].melt(
            id_vars="Strategy",
            var_name="Criterion",
            value_name="Score"
        )

        fig = px.bar(
            score_data,
            x="Strategy",
            y="Score",
            color="Criterion",
            barmode="group",
            title="Integrated Evidence Scores"
        )

        fig.update_layout(
            height=550
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader(
            "Decision Framework"
        )

        framework = pd.DataFrame({
            "Evidence Category": [
                "Out-of-Sample Backtest",
                "Benchmark Comparison",
                "Transaction Costs",
                "Monte Carlo Simulation",
                "Downside Risk"
            ],
            "Weight": [
                "25%",
                "15%",
                "15%",
                "25%",
                "20%"
            ]
        })

        st.dataframe(
            framework,
            use_container_width=True,
            hide_index=True
        )

        st.info(
            """
            The recommendation is generated from an
            interpretable weighted multi-criteria decision
            framework. It should be understood as decision
            support rather than a guarantee of future returns.
            """
        )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.markdown("---")

st.sidebar.caption(
    "AI Portfolio Management System | M.Sc. Statistics"
)

st.sidebar.caption(
    "For research and educational use."
)