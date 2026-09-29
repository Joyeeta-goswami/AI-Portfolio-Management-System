"""
AI Portfolio Management System
------------------------------
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

Run with:  python -m streamlit run app.py
The dashboard only reads the CSV files in data/processed/.
"""

from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from theme import (  # noqa: E402  (needs the sys.path entry above)
    AMBER,
    BORDER,
    COLORWAY,
    MUTED,
    PLOTLY_TEMPLATE_NAME,
    PRIMARY,
    REGIME_COLORS,
    STRATEGY_COLORS,
    SURFACE,
    TEXT,
    register_plotly_template,
)

DATA_DIR = PROJECT_ROOT / "data" / "processed"

register_plotly_template()


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Portfolio Manager",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS  (colours come from theme.py)
# ============================================================

st.markdown(
    f"""
    <style>
    .main-title {{
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, {PRIMARY}, {AMBER});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        width: fit-content;
    }}

    .subtitle {{
        color: {MUTED};
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }}

    .recommendation-card {{
        padding: 1.4rem 1.6rem;
        border-radius: 1rem;
        background-color: {SURFACE};
        border: 1px solid {BORDER};
        border-left: 5px solid {PRIMARY};
        margin-bottom: 1rem;
        color: {TEXT};
    }}

    .recommendation-card h1,
    .recommendation-card h2,
    .recommendation-card p {{
        color: {TEXT};
        margin: 0.2rem 0;
    }}

    .recommendation-card h2 {{
        font-size: 1rem;
        font-weight: 600;
        color: {MUTED};
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }}

    .recommendation-card h1 {{
        color: {PRIMARY};
    }}

    div[data-testid="stMetric"] {{
        background-color: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 0.8rem;
        padding: 0.9rem 1rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_csv(filename):
    path = DATA_DIR / filename
    if not path.exists():
        return None
    return pd.read_csv(path)


def get(filename):
    """Return one processed CSV as a DataFrame, or None if it is missing."""
    return load_csv(filename)


# ============================================================
# DISPLAY HELPERS
# ============================================================

# Columns that hold fractions (0.05 = 5%) and should be shown as percentages.
_PERCENT_KEYWORDS = ("Return", "Volatility", "Drawdown", "VaR", "Probability")


def _column_format(name):
    """Return (scale, format) for a column, or None for text columns."""
    if "(%)" in name:                      # already in percent units
        return 1, "%.2f"
    if name == "Transaction Cost":         # 0.001 -> 0.10%
        return 100, "%.2f%%"
    if "Sharpe Ratio" in name or name.endswith("Sharpe"):
        return 1, "%.3f"
    if "Score" in name:
        return 1, "%.3f"
    if any(k in name for k in _PERCENT_KEYWORDS) and "Wealth" not in name:
        return 100, "%.2f%%"
    return 1, "%.2f"


def show_table(df, height="auto"):
    """Show a DataFrame with sensible number formats (percent, ratios, ...)."""
    view = df.copy()
    config = {}

    for col in view.columns:
        if not pd.api.types.is_numeric_dtype(view[col]) or pd.api.types.is_bool_dtype(view[col]):
            continue
        scale, fmt = _column_format(col)
        if scale != 1:
            view[col] = view[col] * scale
        config[col] = st.column_config.NumberColumn(col, format=fmt)

    st.dataframe(
        view,
        column_config=config,
        width="stretch",
        hide_index=True,
        height=height,
    )


def style_figure(fig, height=500):
    fig.update_layout(
        template=PLOTLY_TEMPLATE_NAME,
        height=height,
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis=dict(automargin=True),
        yaxis=dict(automargin=True),
    )
    return fig


def show_figure(fig, height=500):
    st.plotly_chart(style_figure(fig, height), theme=None)


def wide_to_long(df, id_col, var_name, value_name):
    columns = [c for c in df.columns if c != id_col]
    return df[[id_col] + columns].melt(
        id_vars=id_col, var_name=var_name, value_name=value_name
    )


def wealth_chart(df, title):
    """Line chart of cumulative wealth (base = 100), one line per strategy."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    long = wide_to_long(df, "date", "Strategy", "Wealth")

    fig = px.line(
        long,
        x="date",
        y="Wealth",
        color="Strategy",
        color_discrete_map=STRATEGY_COLORS,
        title=title,
    )
    fig.add_hline(y=100, line_dash="dash", line_color=MUTED,
                  annotation_text="Start = 100")
    # NIFTY 50 is the reference line, so draw it dashed.
    for trace in fig.data:
        if trace.name in ("NIFTY 50", "NIFTY50"):
            trace.line.dash = "dot"
    fig.update_layout(legend_title_text="", yaxis_title="Wealth (start = 100)",
                      xaxis_title="")
    return fig


def combine(*filenames):
    """Stack several processed CSVs into one DataFrame (None if none exist)."""
    frames = [get(f) for f in filenames]
    frames = [f for f in frames if f is not None]
    return pd.concat(frames, ignore_index=True) if frames else None


def recommended_row(recommendation):
    """First row flagged Recommended, or None."""
    if recommendation is None or "Recommended" not in recommendation.columns:
        return None
    rows = recommendation[recommendation["Recommended"].astype(bool)]
    return rows.iloc[0] if len(rows) else None


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
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("AI Portfolio Management System | M.Sc. Statistics")
st.sidebar.caption("For research and educational use.")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">AI Portfolio Management System</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="subtitle">
    Statistical portfolio optimization, machine-learning
    regime detection, out-of-sample validation and
    risk-aware decision support.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.header("Portfolio Management Overview")

    recommendation = get("final_recommendation_results.csv")
    prices = get("prices.csv")
    mc_wealth = get("monte_carlo_terminal_wealth.csv")

    row = recommended_row(recommendation)

    if row is not None:
        st.markdown(
            f"""
            <div class="recommendation-card">
            <h2>Recommended Strategy</h2>
            <h1>{row['Strategy']}</h1>
            <p>Integrated score: <b>{row['Final Score']:.4f}</b></p>
            <p>Risk profile: <b>{row['Risk Profile']}</b></p>
            <p>{row['Evidence Summary']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("System Components")

    n_assets = (len(prices.columns) - 1) if prices is not None else 8
    n_strategies = len(recommendation) if recommendation is not None else 4
    n_paths = (
        f"{mc_wealth['Simulation'].nunique():,}"
        if mc_wealth is not None and "Simulation" in mc_wealth.columns
        else "10,000"
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Assets", n_assets)
    col2.metric("Strategies", n_strategies)
    col3.metric("Monte Carlo Paths", n_paths)
    col4.metric("ML Method", "K-Means")

    st.subheader("Portfolio Strategies")

    show_table(pd.DataFrame({
        "Strategy": [
            "Max Sharpe",
            "Min Variance",
            "Risk Parity",
            "Risk-Aware Regime Switching",
        ],
        "Purpose": [
            "Risk-adjusted return optimization",
            "Minimum portfolio variance",
            "Balanced contribution to portfolio risk",
            "Adaptive strategy selection using market regimes",
        ],
    }))

    st.info(
        "The recommendation engine is an interpretable multi-criteria "
        "decision system. The machine-learning component of the system "
        "is K-Means market-regime detection."
    )


# ============================================================
# MARKET & RISK
# ============================================================

elif page == "📈 Market & Risk":

    st.header("Market & Risk Analysis")

    prices = get("prices.csv")
    risk_summary = get("risk_summary.csv")

    if prices is None:
        st.error("prices.csv was not found in data/processed/.")
    else:
        prices = prices.copy()
        prices["date"] = pd.to_datetime(prices["date"])

        assets = [c for c in prices.columns if c != "date"]

        selected_assets = st.multiselect(
            "Select assets", assets, default=assets[:4]
        )

        if selected_assets:
            long = prices[["date"] + selected_assets].melt(
                id_vars="date", var_name="Asset", value_name="Price"
            )
            fig = px.line(
                long, x="date", y="Price", color="Asset",
                color_discrete_sequence=COLORWAY,
                title="Historical Asset Prices",
            )
            fig.update_layout(legend_title_text="", xaxis_title="",
                              yaxis_title="Price (₹)")
            show_figure(fig)
        else:
            st.info("Select at least one asset to draw the price chart.")

    if risk_summary is not None:
        st.subheader("Historical Risk Summary")
        show_table(risk_summary)


# ============================================================
# PORTFOLIO OPTIMIZATION
# ============================================================

elif page == "⚙️ Portfolio Optimization":

    st.header("Portfolio Optimization")

    risk_summary = get("risk_summary.csv")
    frontier = get("efficient_frontier.csv")

    if risk_summary is not None:
        st.subheader("Risk Characteristics of Individual Assets")
        show_table(risk_summary)

    if frontier is not None:
        st.subheader("Efficient Frontier")

        frontier = frontier.sort_values("Return")
        fig = px.line(
            frontier, x="Risk", y="Return", markers=True,
            title="Markowitz Efficient Frontier",
            labels={
                "Risk": "Annualized Portfolio Risk",
                "Return": "Annualized Portfolio Return",
            },
        )
        fig.update_traces(line_color=PRIMARY, marker_color=PRIMARY)
        fig.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%")
        show_figure(fig, height=550)

    st.subheader("Optimization Methodology")

    col1, col2, col3 = st.columns(3)
    col1.metric("Markowitz", "Maximum Sharpe")
    col2.metric("Markowitz", "Minimum Variance")
    col3.metric("Alternative", "Risk Parity")

    st.write(
        "Markowitz portfolios use expected returns and the covariance "
        "matrix estimated from the training period. Risk Parity allocates "
        "capital so that portfolio risk contributions are approximately "
        "balanced across assets."
    )


# ============================================================
# BACKTESTING
# ============================================================

elif page == "🧪 Backtesting":

    st.header("Out-of-Sample Backtesting")

    backtest = get("backtest_results.csv")
    wealth = get("cumulative_wealth.csv")

    if backtest is not None:
        st.subheader("Test-Period Performance")
        show_table(backtest)

    if wealth is not None:
        show_figure(wealth_chart(wealth, "Out-of-Sample Cumulative Wealth"))

    st.info(
        "The test period is kept completely separate from the portfolio "
        "optimization stage to evaluate out-of-sample performance."
    )


# ============================================================
# BENCHMARK
# ============================================================

elif page == "🌍 Benchmark":

    st.header("NIFTY 50 Benchmark Comparison")

    benchmark = get("benchmark_adjusted_results.csv")
    wealth = get("benchmark_adjusted_wealth.csv")

    if benchmark is not None:
        st.subheader("Common-Date Benchmark Comparison")
        show_table(benchmark)

    if wealth is not None:
        show_figure(wealth_chart(wealth, "Portfolio vs NIFTY 50"))


# ============================================================
# REGIME ANALYSIS
# ============================================================

elif page == "🔄 Regime Analysis":

    st.header("Market Regime Detection")

    regimes = get("market_regimes.csv")
    regime_returns = get("regime_strategy_test_returns.csv")

    if regimes is not None:
        regimes = regimes.copy()
        regimes["date"] = pd.to_datetime(regimes["date"])

        st.subheader("Detected Market Regimes")

        if "regime" in regimes.columns:
            counts = regimes["regime"].value_counts().reset_index()
            counts.columns = ["Regime", "Observations"]

            fig = px.bar(
                counts, x="Regime", y="Observations", color="Regime",
                color_discrete_map=REGIME_COLORS,
                title="Regime Frequency",
            )
            fig.update_layout(showlegend=False)
            show_figure(fig, height=400)

        st.caption("Most recent 20 observations")
        regimes_view = regimes.tail(20).copy()
        regimes_view["date"] = regimes_view["date"].dt.date
        show_table(regimes_view)

    if regime_returns is not None:
        st.subheader("Regime Strategy Selection")

        counts = regime_returns["Selected Strategy"].value_counts().reset_index()
        counts.columns = ["Strategy", "Observations"]

        fig = px.bar(
            counts, x="Strategy", y="Observations", color="Strategy",
            color_discrete_map=STRATEGY_COLORS,
            title="Selected Strategy During Test Period",
        )
        fig.update_layout(showlegend=False)
        show_figure(fig, height=400)


# ============================================================
# TRANSACTION COSTS
# ============================================================

elif page == "💰 Transaction Costs":

    st.header("Transaction Cost Analysis")

    transaction = get("transaction_cost_results.csv")
    sensitivity = get("transaction_cost_sensitivity.csv")

    if transaction is not None:
        st.subheader("Net Performance After Transaction Costs")
        show_table(transaction)

    if sensitivity is not None:
        st.subheader("Transaction Cost Sensitivity")

        strategies = list(sensitivity["Strategy"].unique())
        selected_strategy = st.selectbox("Select strategy", strategies)

        selected = sensitivity[sensitivity["Strategy"] == selected_strategy]

        fig = px.line(
            selected, x="Transaction Cost (%)", y="Net Total Return",
            markers=True,
            title="Net Total Return vs Transaction Cost",
        )
        color = STRATEGY_COLORS.get(selected_strategy, PRIMARY)
        fig.update_traces(line_color=color, marker_color=color)
        fig.update_layout(yaxis_tickformat=".2%")
        show_figure(fig, height=450)


# ============================================================
# MONTE CARLO
# ============================================================

elif page == "🎲 Monte Carlo":

    st.header("Monte Carlo Risk Simulation")

    monte_carlo = get("monte_carlo_results.csv")
    regime_mc = get("monte_carlo_regime_results.csv")

    # Fixed strategies and the regime-switching strategy are simulated
    # separately; combine them so every strategy can be inspected below.
    terminal = combine(
        "monte_carlo_terminal_wealth.csv",
        "monte_carlo_regime_terminal_wealth.csv",
    )
    drawdowns = combine(
        "monte_carlo_drawdowns.csv",
        "monte_carlo_regime_drawdowns.csv",
    )

    if monte_carlo is not None:
        st.subheader("Fixed Portfolio Simulation")
        show_table(monte_carlo)

    if regime_mc is not None:
        st.subheader("Regime-Switching Monte Carlo")
        show_table(regime_mc)

    if terminal is not None:
        st.subheader("Terminal Wealth Distribution")

        strategy = st.selectbox(
            "Terminal wealth strategy", list(terminal["Strategy"].unique())
        )
        subset = terminal[terminal["Strategy"] == strategy]

        fig = px.histogram(
            subset, x="Terminal Wealth", nbins=60,
            color_discrete_sequence=[STRATEGY_COLORS.get(strategy, PRIMARY)],
            title="Monte Carlo Terminal Wealth Distribution",
        )
        fig.add_vline(x=100, line_dash="dash", line_color=TEXT,
                      annotation_text="Initial Wealth")
        fig.update_layout(bargap=0.03)
        show_figure(fig, height=450)

    if drawdowns is not None:
        st.subheader("Maximum Drawdown Distribution")

        strategy_dd = st.selectbox(
            "Maximum drawdown strategy", list(drawdowns["Strategy"].unique())
        )
        subset = drawdowns[drawdowns["Strategy"] == strategy_dd]

        fig = px.histogram(
            subset, x="Maximum Drawdown", nbins=60,
            color_discrete_sequence=[STRATEGY_COLORS.get(strategy_dd, PRIMARY)],
            title="Monte Carlo Maximum Drawdown Distribution",
        )
        fig.update_layout(bargap=0.03, xaxis_tickformat=".0%")
        show_figure(fig, height=450)


# ============================================================
# AI RECOMMENDATION
# ============================================================

elif page == "🤖 AI Recommendation":

    st.header("Integrated Portfolio Recommendation")

    recommendation = get("final_recommendation_results.csv")

    if recommendation is None:
        st.error("Final recommendation results were not found.")
    else:
        row = recommended_row(recommendation)

        if row is not None:
            st.success(f"Recommended Strategy: {row['Strategy']}")

            col1, col2, col3 = st.columns(3)
            col1.metric("Integrated Score", f"{row['Final Score']:.4f}")
            col2.metric("Risk Profile", row["Risk Profile"])
            col3.metric("Recommendation", "Selected")

            st.subheader("Evidence Summary")
            st.write(row["Evidence Summary"])

        st.subheader("Strategy Comparison")

        criteria = [
            "Backtest Score",
            "Benchmark Score",
            "Transaction Cost Score",
            "Monte Carlo Score",
            "Downside Risk Score",
        ]

        show_table(recommendation[["Strategy"] + criteria + ["Final Score"]])

        score_data = recommendation[["Strategy"] + criteria].melt(
            id_vars="Strategy", var_name="Criterion", value_name="Score"
        )

        fig = px.bar(
            score_data, x="Strategy", y="Score", color="Criterion",
            barmode="group", color_discrete_sequence=COLORWAY,
            title="Integrated Evidence Scores",
        )
        fig.update_layout(legend_title_text="")
        show_figure(fig, height=550)

        st.subheader("Decision Framework")

        # Keep in sync with WEIGHTS in ai_engine/final_recommendation.py
        show_table(pd.DataFrame({
            "Evidence Category": [
                "Out-of-Sample Backtest",
                "Benchmark Comparison",
                "Transaction Costs",
                "Monte Carlo Simulation",
                "Downside Risk",
            ],
            "Weight": ["25%", "15%", "15%", "25%", "20%"],
        }))

        st.info(
            "The recommendation is generated from an interpretable weighted "
            "multi-criteria decision framework. It should be understood as "
            "decision support rather than a guarantee of future returns."
        )
