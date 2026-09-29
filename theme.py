"""
Shared colour theme for the AI Portfolio Manager.

One place defines the palette so the Streamlit dashboard, the Plotly
charts and the Matplotlib figures written to visualization/outputs/
all look the same. Change a colour here and it changes everywhere.

The Streamlit base theme (background, text, widgets) lives in
.streamlit/config.toml and uses the same hex values.
"""

# ------------------------------------------------------------
# Palette ("Midnight & Teal")
# ------------------------------------------------------------

BACKGROUND = "#0B1220"
SURFACE = "#131C2E"
BORDER = "#24304A"
TEXT = "#E6EDF7"
MUTED = "#93A1B8"

PRIMARY = "#2DD4BF"   # teal
AMBER = "#F5B942"
VIOLET = "#A78BFA"
BLUE = "#60A5FA"
CORAL = "#FB7185"
SLATE = "#94A3B8"

# Colour cycle used for any series that has no fixed colour
COLORWAY = [PRIMARY, AMBER, BLUE, VIOLET, CORAL, "#34D399", "#F472B6", SLATE]

# Fixed colour per strategy so a strategy looks the same on every chart.
# Several spellings are listed because the CSV outputs are not consistent.
STRATEGY_COLORS = {
    "Max Sharpe": PRIMARY,
    "Maximum Sharpe": PRIMARY,
    "Min Variance": BLUE,
    "Minimum Variance": BLUE,
    "Risk Parity": AMBER,
    "Risk-Aware Regime Switching": VIOLET,
    "Regime Switching": VIOLET,
    "NIFTY 50": SLATE,
    "NIFTY50": SLATE,
}

# Fixed colour per market regime
REGIME_COLORS = {
    "Calm": PRIMARY,
    "Defensive": AMBER,
    "High Volatility": CORAL,
}


# ------------------------------------------------------------
# Plotly
# ------------------------------------------------------------

PLOTLY_TEMPLATE_NAME = "portfolio_dark"


def register_plotly_template(set_default=True):
    """Register the dark template with Plotly (and make it the default)."""
    import plotly.graph_objects as go
    import plotly.io as pio

    axis = dict(
        gridcolor=BORDER,
        zerolinecolor=BORDER,
        linecolor=BORDER,
        tickfont=dict(color=MUTED),
        title=dict(font=dict(color=MUTED)),
    )

    template = go.layout.Template()
    template.layout = go.Layout(
        colorway=COLORWAY,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT),
        title=dict(font=dict(color=TEXT, size=18)),
        xaxis=axis,
        yaxis=axis,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT),
        ),
        hoverlabel=dict(
            bgcolor=SURFACE,
            bordercolor=BORDER,
            font=dict(color=TEXT),
        ),
    )

    pio.templates[PLOTLY_TEMPLATE_NAME] = template
    if set_default:
        pio.templates.default = PLOTLY_TEMPLATE_NAME
    return template


# ------------------------------------------------------------
# Matplotlib
# ------------------------------------------------------------

def apply_matplotlib_theme():
    """Apply the dark theme to every Matplotlib figure created afterwards."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": BACKGROUND,
        "savefig.facecolor": BACKGROUND,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": BORDER,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": TEXT,
        "axes.prop_cycle": mpl.cycler(color=COLORWAY),
        "text.color": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": BORDER,
        "legend.facecolor": SURFACE,
        "legend.edgecolor": BORDER,
        "legend.labelcolor": TEXT,
    })
