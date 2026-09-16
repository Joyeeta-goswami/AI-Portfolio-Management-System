import sys
import numpy as np
import pandas as pd
from pathlib import Path
from contextlib import redirect_stdout
import io


# --------------------------------------------------
# Make project root importable
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from optimization.backtest import PortfolioBacktester


class RegimeStrategyAnalyzer:
    """
    Risk-aware regime-based portfolio strategy selection.

    The model:
        1. Uses market regimes learned from training data.
        2. Evaluates three portfolio strategies within each
           training regime.
        3. Builds a risk-aware score for each strategy.
        4. Selects the best strategy for each regime.
        5. Freezes the learned mapping.
        6. Applies it to the unseen test period.

    Strategies:
        Maximum Sharpe
        Minimum Variance
        Risk Parity
    """

    def __init__(
        self,
        returns_file="data/processed/returns.csv",
        regime_file="data/processed/market_regimes.csv",
        train_ratio=0.80
    ):
        self.returns_file = Path(returns_file)
        self.regime_file = Path(regime_file)
        self.train_ratio = train_ratio

        self.returns = None
        self.regimes = None

        self.train_returns = None
        self.test_returns = None

        self.train_regimes = None
        self.test_regimes = None

        self.train_data = None
        self.test_data = None

        self.training_weights = None

        self.regime_metrics = None
        self.regime_scores = None

        self.best_strategy_by_regime = None
        self.regime_strategy_metrics = None

    # ==================================================
    # DATA LOADING
    # ==================================================

    def load_data(self):
        """
        Load asset returns and classified market regimes.
        """

        self.returns = pd.read_csv(
            self.returns_file,
            parse_dates=["date"]
        )

        self.regimes = pd.read_csv(
            self.regime_file,
            parse_dates=["date"]
        )

        self.returns = (
            self.returns
            .sort_values("date")
            .reset_index(drop=True)
        )

        self.regimes = (
            self.regimes
            .sort_values("date")
            .reset_index(drop=True)
        )

        print(
            "Returns and regime data loaded successfully."
        )

    # ==================================================
    # TRAIN / TEST SPLIT
    # ==================================================

    def split_data(self):
        """
        Split returns and regime observations chronologically.
        """

        split_index = int(
            len(self.returns) * self.train_ratio
        )

        split_date = self.returns[
            "date"
        ].iloc[split_index]

        self.train_returns = self.returns[
            self.returns["date"] < split_date
        ].copy()

        self.test_returns = self.returns[
            self.returns["date"] >= split_date
        ].copy()

        self.train_regimes = self.regimes[
            (self.regimes["dataset"] == "train")
            & (self.regimes["date"] < split_date)
        ].copy()

        self.test_regimes = self.regimes[
            (self.regimes["dataset"] == "test")
            & (self.regimes["date"] >= split_date)
        ].copy()

        print("\nData Split")
        print("-" * 55)

        print(
            f"Training returns : "
            f"{len(self.train_returns)}"
        )

        print(
            f"Testing returns  : "
            f"{len(self.test_returns)}"
        )

        print(
            f"Training regimes : "
            f"{len(self.train_regimes)}"
        )

        print(
            f"Testing regimes  : "
            f"{len(self.test_regimes)}"
        )

    # ==================================================
    # ALIGN REGIMES WITH RETURNS
    # ==================================================

    def align_data(self):
        """
        Align regime classifications with return observations.
        """

        self.train_data = pd.merge(
            self.train_returns,
            self.train_regimes[
                ["date", "regime"]
            ],
            on="date",
            how="inner"
        )

        self.test_data = pd.merge(
            self.test_returns,
            self.test_regimes[
                ["date", "regime"]
            ],
            on="date",
            how="inner"
        )

        self.train_data = (
            self.train_data
            .sort_values("date")
            .reset_index(drop=True)
        )

        self.test_data = (
            self.test_data
            .sort_values("date")
            .reset_index(drop=True)
        )

        print("\nData Alignment")
        print("-" * 55)

        print(
            f"Aligned training observations : "
            f"{len(self.train_data)}"
        )

        print(
            f"Aligned testing observations  : "
            f"{len(self.test_data)}"
        )

    # ==================================================
    # CALCULATE FIXED STRATEGY RETURNS
    # ==================================================

    def calculate_strategy_returns(self):
        """
        Generate the three portfolio strategies using
        training data only.

        Their weights remain frozen when applied to
        the test period.
        """

        backtester = PortfolioBacktester(
            returns_file=str(self.returns_file),
            train_ratio=self.train_ratio
        )

        # Suppress repeated diagnostic output from
        # the optimization engine.
        with redirect_stdout(io.StringIO()):

            backtester.load_returns()
            backtester.split_data()

            max_sharpe, min_variance = (
                backtester
                .generate_training_markowitz_portfolios()
            )

            covariance = (
                backtester
                .calculate_training_covariance()
            )

            risk_parity = (
                backtester
                .optimize_risk_parity(
                    covariance
                )
            )

        # Freeze training weights
        self.training_weights = {
            "Maximum Sharpe": max_sharpe.x,
            "Minimum Variance": min_variance.x,
            "Risk Parity": risk_parity.x
        }

        asset_columns = (
            self.train_returns.columns[1:]
        )

        # ------------------------------------------------
        # Training strategy returns
        # ------------------------------------------------

        self.train_data["Maximum Sharpe"] = (
            self.train_data[asset_columns].values
            @ self.training_weights["Maximum Sharpe"]
        )

        self.train_data["Minimum Variance"] = (
            self.train_data[asset_columns].values
            @ self.training_weights["Minimum Variance"]
        )

        self.train_data["Risk Parity"] = (
            self.train_data[asset_columns].values
            @ self.training_weights["Risk Parity"]
        )

        # ------------------------------------------------
        # Test strategy returns
        # ------------------------------------------------

        self.test_data["Maximum Sharpe"] = (
            self.test_data[asset_columns].values
            @ self.training_weights["Maximum Sharpe"]
        )

        self.test_data["Minimum Variance"] = (
            self.test_data[asset_columns].values
            @ self.training_weights["Minimum Variance"]
        )

        self.test_data["Risk Parity"] = (
            self.test_data[asset_columns].values
            @ self.training_weights["Risk Parity"]
        )

        print(
            "\nTraining and test strategy returns calculated."
        )

    # ==================================================
    # REGIME-SPECIFIC METRICS
    # ==================================================

    def calculate_regime_metrics(self):
        """
        Calculate risk-return statistics for each strategy
        separately within each training regime.

        Metrics:
            Annualized Return
            Annualized Volatility
            Sharpe Ratio
            Maximum Drawdown
            VaR (95%)
        """

        strategies = [
            "Maximum Sharpe",
            "Minimum Variance",
            "Risk Parity"
        ]

        trading_days = 252
        risk_free_rate = 0.065

        rows = []

        for regime in sorted(
            self.train_data["regime"].unique()
        ):

            regime_data = self.train_data[
                self.train_data["regime"] == regime
            ]

            for strategy in strategies:

                returns = (
                    regime_data[strategy]
                    .dropna()
                )

                if len(returns) == 0:
                    continue

                # Arithmetic annualized return
                annualized_return = (
                    returns.mean()
                    * trading_days
                )

                # Annualized volatility
                annualized_volatility = (
                    returns.std()
                    * np.sqrt(trading_days)
                )

                # Sharpe ratio
                if annualized_volatility == 0:
                    sharpe_ratio = np.nan
                else:
                    sharpe_ratio = (
                        annualized_return
                        - risk_free_rate
                    ) / annualized_volatility

                # Wealth path
                wealth = (
                    1 + returns
                ).cumprod()

                running_max = wealth.cummax()

                drawdown = (
                    wealth / running_max
                ) - 1

                maximum_drawdown = (
                    drawdown.min()
                )

                # 95% Historical VaR
                var_95 = returns.quantile(0.05)

                rows.append(
                    {
                        "Regime": regime,
                        "Strategy": strategy,
                        "Annualized Return":
                            annualized_return,
                        "Annualized Volatility":
                            annualized_volatility,
                        "Sharpe Ratio":
                            sharpe_ratio,
                        "Maximum Drawdown":
                            maximum_drawdown,
                        "VaR (95%)":
                            var_95,
                        "Observations":
                            len(returns)
                    }
                )

        self.regime_metrics = pd.DataFrame(rows)

        print(
            "\nRisk-Return Metrics by Training Regime"
        )
        print("-" * 90)

        print(
            self.regime_metrics.to_string(
                index=False,
                float_format=lambda x: f"{x:.4%}"
            )
        )

        return self.regime_metrics

    # ==================================================
    # RISK-AWARE SCORING
    # ==================================================

    def calculate_regime_scores(self):
        """
        Calculate a risk-aware multi-criteria score
        for each strategy within each regime.

        Higher is better:
            Annualized Return
            Sharpe Ratio

        Lower is better:
            Annualized Volatility
            Maximum Drawdown magnitude
            VaR magnitude

        Weights:
            Return       = 30%
            Sharpe       = 25%
            Volatility   = 15%
            Drawdown     = 20%
            VaR          = 10%
        """

        metrics = self.regime_metrics.copy()

        # Convert negative risk measures to positive magnitudes
        metrics["Drawdown Risk"] = (
            metrics["Maximum Drawdown"].abs()
        )

        metrics["VaR Risk"] = (
            metrics["VaR (95%)"].abs()
        )

        weights = {
            "Annualized Return": 0.30,
            "Sharpe Ratio": 0.25,
            "Annualized Volatility": 0.15,
            "Drawdown Risk": 0.20,
            "VaR Risk": 0.10
        }

        scored_groups = []

        for regime, group in metrics.groupby(
            "Regime"
        ):

            group = group.copy()

            scores = pd.DataFrame(
                index=group.index
            )

            # -------------------------------
            # Higher is better
            # -------------------------------

            for metric in [
                "Annualized Return",
                "Sharpe Ratio"
            ]:

                minimum = group[metric].min()
                maximum = group[metric].max()

                if (
                    pd.isna(minimum)
                    or maximum == minimum
                ):
                    scores[metric] = 1.0
                else:
                    scores[metric] = (
                        group[metric] - minimum
                    ) / (
                        maximum - minimum
                    )

            # -------------------------------
            # Lower is better
            # -------------------------------

            for metric in [
                "Annualized Volatility",
                "Drawdown Risk",
                "VaR Risk"
            ]:

                minimum = group[metric].min()
                maximum = group[metric].max()

                if (
                    pd.isna(minimum)
                    or maximum == minimum
                ):
                    scores[metric] = 1.0
                else:
                    scores[metric] = (
                        maximum - group[metric]
                    ) / (
                        maximum - minimum
                    )

            # -------------------------------
            # Weighted score
            # -------------------------------

            group["Overall Score"] = 0.0

            for metric, weight in weights.items():
                group["Overall Score"] += (
                    scores[metric]
                    * weight
                )

            # Keep individual criterion scores
            group["Return Score"] = scores[
                "Annualized Return"
            ]

            group["Sharpe Score"] = scores[
                "Sharpe Ratio"
            ]

            group["Volatility Score"] = scores[
                "Annualized Volatility"
            ]

            group["Drawdown Score"] = scores[
                "Drawdown Risk"
            ]

            group["VaR Score"] = scores[
                "VaR Risk"
            ]

            scored_groups.append(group)

        self.regime_scores = pd.concat(
            scored_groups,
            ignore_index=True
        )

        print(
            "\nRisk-Aware Strategy Scores by Regime"
        )
        print("-" * 90)

        print(
            self.regime_scores[
                [
                    "Regime",
                    "Strategy",
                    "Return Score",
                    "Sharpe Score",
                    "Volatility Score",
                    "Drawdown Score",
                    "VaR Score",
                    "Overall Score"
                ]
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}"
            )
        )

        return self.regime_scores

    # ==================================================
    # LEARN STRATEGY FOR EACH REGIME
    # ==================================================

    def learn_best_strategy_by_regime(self):
        """
        Select the strategy with the highest risk-aware
        score within each training regime.
        """

        best_rows = (
            self.regime_scores
            .loc[
                self.regime_scores
                .groupby("Regime")[
                    "Overall Score"
                ].idxmax()
            ]
            .copy()
        )

        self.best_strategy_by_regime = (
            best_rows
            .set_index("Regime")["Strategy"]
            .to_dict()
        )

        print(
            "\nBest Risk-Aware Strategy Learned for Each Regime"
        )
        print("-" * 70)

        for regime, strategy in (
            self.best_strategy_by_regime.items()
        ):
            score = best_rows.loc[
                best_rows["Regime"] == regime,
                "Overall Score"
            ].iloc[0]

            print(
                f"{regime:<20} → "
                f"{strategy:<20} "
                f"(Score = {score:.4f})"
            )

        return self.best_strategy_by_regime

    # ==================================================
    # APPLY TO TEST PERIOD
    # ==================================================

    def apply_regime_strategy_to_test(self):
        """
        Apply the frozen regime-to-strategy mapping
        to the unseen test period.
        """

        self.test_data["Selected Strategy"] = (
            self.test_data["regime"].map(
                self.best_strategy_by_regime
            )
        )

        self.test_data[
            "Regime Strategy Return"
        ] = np.nan

        strategies = [
            "Maximum Sharpe",
            "Minimum Variance",
            "Risk Parity"
        ]

        for strategy in strategies:

            mask = (
                self.test_data["Selected Strategy"]
                == strategy
            )

            self.test_data.loc[
                mask,
                "Regime Strategy Return"
            ] = self.test_data.loc[
                mask,
                strategy
            ]

        print(
            "\nRisk-Aware Regime-Switching Test Results"
        )
        print("-" * 80)

        print(
            self.test_data[
                [
                    "date",
                    "regime",
                    "Selected Strategy",
                    "Regime Strategy Return"
                ]
            ].head(10).to_string(
                index=False
            )
        )

        print(
            "\nStrategy Selection Counts"
        )
        print("-" * 55)

        print(
            self.test_data[
                "Selected Strategy"
            ].value_counts()
        )

        return self.test_data

    # ==================================================
    # TEST PERFORMANCE
    # ==================================================

    def calculate_regime_strategy_performance(self):
        """
        Evaluate the risk-aware regime-switching strategy
        on the unseen test period.
        """

        returns = (
            self.test_data[
                "Regime Strategy Return"
            ]
            .dropna()
        )

        if len(returns) == 0:
            raise ValueError(
                "No valid regime-switching test returns."
            )

        trading_days = 252
        risk_free_rate = 0.065

        # Total return
        total_return = (
            (1 + returns).prod() - 1
        )

        # Annualized return
        annualized_return = (
            (1 + total_return)
            ** (trading_days / len(returns))
            - 1
        )

        # Annualized volatility
        annualized_volatility = (
            returns.std()
            * np.sqrt(trading_days)
        )

        # Sharpe
        if annualized_volatility == 0:
            sharpe_ratio = np.nan
        else:
            sharpe_ratio = (
                annualized_return
                - risk_free_rate
            ) / annualized_volatility

        # Drawdown
        wealth = (
            1 + returns
        ).cumprod()

        running_max = wealth.cummax()

        drawdown = (
            wealth / running_max
        ) - 1

        maximum_drawdown = (
            drawdown.min()
        )

        # VaR
        var_95 = returns.quantile(0.05)

        self.regime_strategy_metrics = {
            "Total Return": total_return,
            "Annualized Return":
                annualized_return,
            "Annualized Volatility":
                annualized_volatility,
            "Sharpe Ratio":
                sharpe_ratio,
            "Maximum Drawdown":
                maximum_drawdown,
            "VaR (95%)":
                var_95
        }

        print(
            "\nRisk-Aware Regime-Switching Strategy Performance"
        )
        print("-" * 70)

        print(
            f"Observations          : {len(returns)}"
        )

        print(
            f"Total Return          : "
            f"{total_return:.4%}"
        )

        print(
            f"Annualized Return     : "
            f"{annualized_return:.4%}"
        )

        print(
            f"Annualized Volatility : "
            f"{annualized_volatility:.4%}"
        )

        print(
            f"Sharpe Ratio          : "
            f"{sharpe_ratio:.3f}"
        )

        print(
            f"Maximum Drawdown      : "
            f"{maximum_drawdown:.4%}"
        )

        print(
            f"VaR (95%)             : "
            f"{var_95:.4%}"
        )

        return self.regime_strategy_metrics

    # ==================================================
    # SAVE OUTPUTS
    # ==================================================

    def save_outputs(
        self,
        score_file=(
            "data/processed/"
            "regime_strategy_scores.csv"
        ),
        test_file=(
            "data/processed/"
            "regime_strategy_test_returns.csv"
        ),
        result_file=(
            "data/processed/"
            "regime_switching_results_v2.csv"
        )
    ):
        """
        Save regime scores, test history, and
        regime-switching performance.
        """

        self.regime_scores.to_csv(
            score_file,
            index=False
        )

        self.test_data[
            [
                "date",
                "regime",
                "Selected Strategy",
                "Regime Strategy Return"
            ]
        ].to_csv(
            test_file,
            index=False
        )

        pd.DataFrame(
            [self.regime_strategy_metrics],
            index=["Risk-Aware Regime Switching"]
        ).to_csv(
            result_file
        )

        print(
            "\nRisk-aware strategy scores saved to:"
        )
        print(score_file)

        print(
            "\nRisk-aware test history saved to:"
        )
        print(test_file)

        print(
            "\nRisk-aware performance saved to:"
        )
        print(result_file)


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":

    analyzer = RegimeStrategyAnalyzer()

    analyzer.load_data()

    analyzer.split_data()

    analyzer.align_data()

    analyzer.calculate_strategy_returns()

    analyzer.calculate_regime_metrics()

    analyzer.calculate_regime_scores()

    analyzer.learn_best_strategy_by_regime()

    analyzer.apply_regime_strategy_to_test()

    analyzer.calculate_regime_strategy_performance()

    analyzer.save_outputs()