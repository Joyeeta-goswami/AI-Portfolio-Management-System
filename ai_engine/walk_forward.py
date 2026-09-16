import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class WalkForwardValidator:
    """
    Expanding-window walk-forward validation for:

        1. Maximum Sharpe
        2. Minimum Variance
        3. Risk Parity
        4. Risk-Aware Regime Switching

    Every window follows:

        Training data
            ↓
        Portfolio optimization
            ↓
        Regime model fitted on training data
            ↓
        Risk-aware strategy selection by regime
            ↓
        Freeze all learned information
            ↓
        Unseen test window
    """

    def __init__(
        self,
        returns_file="data/processed/returns.csv",
        initial_train_size=400,
        test_size=60,
        rolling_window=20,
        n_regimes=3
    ):
        self.returns_file = Path(returns_file)

        self.initial_train_size = initial_train_size
        self.test_size = test_size

        self.rolling_window = rolling_window
        self.n_regimes = n_regimes

        self.returns = None
        self.results = []

    # ==================================================
    # DATA
    # ==================================================

    def load_returns(self):
        """
        Load chronological asset returns.
        """

        self.returns = pd.read_csv(
            self.returns_file,
            parse_dates=["date"]
        )

        self.returns = (
            self.returns
            .sort_values("date")
            .reset_index(drop=True)
        )

        print("Returns loaded successfully.")
        print(
            f"Total observations: "
            f"{len(self.returns)}"
        )

    # ==================================================
    # PORTFOLIO OPTIMIZATION
    # ==================================================

    def optimize_markowitz(
        self,
        expected_returns,
        covariance_matrix,
        objective="max_sharpe"
    ):
        """
        Long-only Markowitz optimization.
        """

        n_assets = len(expected_returns)

        initial_weights = (
            np.ones(n_assets) / n_assets
        )

        bounds = tuple(
            (0, 1)
            for _ in range(n_assets)
        )

        constraints = {
            "type": "eq",
            "fun": lambda weights:
                np.sum(weights) - 1
        }

        def portfolio_return(weights):

            return np.dot(
                weights,
                expected_returns.values
            )

        def portfolio_risk(weights):

            variance = (
                weights.T
                @ covariance_matrix.values
                @ weights
            )

            return np.sqrt(
                max(variance, 0)
            )

        def negative_sharpe(weights):

            risk = portfolio_risk(weights)

            if risk <= 0:
                return 1e10

            return -(
                portfolio_return(weights)
                - 0.065
            ) / risk

        def portfolio_variance(weights):

            return (
                weights.T
                @ covariance_matrix.values
                @ weights
            )

        if objective == "max_sharpe":

            objective_function = (
                negative_sharpe
            )

        else:

            objective_function = (
                portfolio_variance
            )

        result = minimize(
            objective_function,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )

        if not result.success:

            raise RuntimeError(
                "Markowitz optimization failed: "
                + result.message
            )

        return result.x

    def optimize_risk_parity(
        self,
        covariance_matrix
    ):
        """
        Equal Risk Contribution optimization.
        """

        n_assets = covariance_matrix.shape[0]

        covariance = covariance_matrix.values

        initial_weights = (
            np.ones(n_assets) / n_assets
        )

        def portfolio_risk(weights):

            variance = (
                weights.T
                @ covariance
                @ weights
            )

            return np.sqrt(
                max(variance, 0)
            )

        def risk_contributions(weights):

            risk = portfolio_risk(weights)

            if risk <= 0:
                return np.zeros(n_assets)

            marginal_contribution = (
                covariance @ weights
            )

            component_contribution = (
                weights * marginal_contribution
            )

            return (
                component_contribution / risk
            )

        def objective(weights):

            contributions = (
                risk_contributions(weights)
            )

            if np.any(contributions <= 0):
                return 1e10

            log_contributions = np.log(
                contributions
            )

            return np.sum(
                (
                    log_contributions
                    - log_contributions.mean()
                ) ** 2
            )

        constraints = {
            "type": "eq",
            "fun": lambda weights:
                np.sum(weights) - 1
        }

        bounds = tuple(
            (0.001, 1)
            for _ in range(n_assets)
        )

        result = minimize(
            objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )

        if not result.success:

            raise RuntimeError(
                "Risk Parity optimization failed: "
                + result.message
            )

        return result.x

    # ==================================================
    # PERFORMANCE METRICS
    # ==================================================

    def calculate_metrics(
        self,
        portfolio_returns
    ):
        """
        Calculate performance metrics.
        """

        trading_days = 252
        risk_free_rate = 0.065

        portfolio_returns = (
            pd.Series(
                portfolio_returns
            )
            .dropna()
        )

        if len(portfolio_returns) == 0:
            return {
                "Total Return": np.nan,
                "Annualized Return": np.nan,
                "Annualized Volatility": np.nan,
                "Sharpe Ratio": np.nan,
                "Maximum Drawdown": np.nan,
                "VaR (95%)": np.nan
            }

        total_return = (
            (1 + portfolio_returns)
            .prod()
            - 1
        )

        annualized_return = (
            (1 + total_return)
            ** (
                trading_days
                / len(portfolio_returns)
            )
            - 1
        )

        annualized_volatility = (
            portfolio_returns.std()
            * np.sqrt(trading_days)
        )

        if annualized_volatility == 0:

            sharpe_ratio = np.nan

        else:

            sharpe_ratio = (
                annualized_return
                - risk_free_rate
            ) / annualized_volatility

        wealth = (
            1 + portfolio_returns
        ).cumprod()

        running_max = (
            wealth.cummax()
        )

        drawdown = (
            wealth / running_max
        ) - 1

        maximum_drawdown = (
            drawdown.min()
        )

        var_95 = (
            portfolio_returns
            .quantile(0.05)
        )

        return {
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

    # ==================================================
    # MARKET REGIME FEATURES
    # ==================================================

    def create_regime_features(
        self,
        data
    ):
        """
        Create regime features using information BEFORE
        each day's return.

        Features:
            rolling_return
            rolling_volatility
            rolling_correlation
        """

        asset_returns = (
            data.iloc[:, 1:]
        )

        market_return = (
            asset_returns.mean(axis=1)
        )

        rolling_return = (
            market_return
            .rolling(
                self.rolling_window,
                min_periods=self.rolling_window
            )
            .mean()
            .shift(1)
        )

        rolling_volatility = (
            market_return
            .rolling(
                self.rolling_window,
                min_periods=self.rolling_window
            )
            .std()
            .shift(1)
            * np.sqrt(252)
        )

        rolling_correlation = []

        for i in range(
            len(asset_returns)
        ):

            if i < self.rolling_window:

                rolling_correlation.append(
                    np.nan
                )

                continue

            window = asset_returns.iloc[
                i - self.rolling_window:i
            ]

            correlation_matrix = (
                window.corr()
            )

            upper_triangle = (
                correlation_matrix.where(
                    np.triu(
                        np.ones(
                            correlation_matrix.shape
                        ),
                        k=1
                    ).astype(bool)
                )
            )

            correlation_value = (
                upper_triangle
                .stack()
                .mean()
            )

            rolling_correlation.append(
                correlation_value
            )

        return pd.DataFrame(
            {
                "date": data["date"],
                "rolling_return":
                    rolling_return,
                "rolling_volatility":
                    rolling_volatility,
                "rolling_correlation":
                    rolling_correlation
            }
        )

    # ==================================================
    # FIT REGIME MODEL
    # ==================================================

    def fit_regime_model(
        self,
        training_data
    ):
        """
        Fit StandardScaler + K-Means using training data only.
        """

        features = (
            self.create_regime_features(
                training_data
            )
            .dropna()
            .reset_index(drop=True)
        )

        feature_columns = [
            "rolling_return",
            "rolling_volatility",
            "rolling_correlation"
        ]

        X = features[
            feature_columns
        ]

        scaler = StandardScaler()

        X_scaled = (
            scaler.fit_transform(X)
        )

        kmeans = KMeans(
            n_clusters=self.n_regimes,
            random_state=42,
            n_init=10
        )

        features["cluster"] = (
            kmeans.fit_predict(
                X_scaled
            )
        )

        return (
            scaler,
            kmeans,
            features
        )

    # ==================================================
    # CLASSIFY REGIMES
    # ==================================================

    def classify_test_regimes(
        self,
        scaler,
        kmeans,
        full_data,
        test_data
    ):
        """
        Classify test observations using the training-fitted
        scaler and K-Means model.

        Features are calculated using historical data preceding
        each test day.
        """

        all_features = (
            self.create_regime_features(
                full_data
            )
        )

        test_dates = set(
            test_data["date"]
        )

        test_features = (
            all_features[
                all_features["date"].isin(
                    test_dates
                )
            ]
            .dropna()
            .copy()
        )

        feature_columns = [
            "rolling_return",
            "rolling_volatility",
            "rolling_correlation"
        ]

        X_test = test_features[
            feature_columns
        ]

        X_test_scaled = (
            scaler.transform(
                X_test
            )
        )

        test_features["cluster"] = (
            kmeans.predict(
                X_test_scaled
            )
        )

        return test_features

    # ==================================================
    # ASSIGN ECONOMIC REGIME LABELS
    # ==================================================

    def assign_regime_labels(
        self,
        training_features,
        test_features
    ):
        """
        Map K-Means cluster IDs to:
            Calm
            Defensive
            High Volatility

        Mapping is learned from training characteristics only.
        """

        summary = (
            training_features
            .groupby("cluster")
            [
                [
                    "rolling_return",
                    "rolling_volatility",
                    "rolling_correlation"
                ]
            ]
            .mean()
        )

        calm_cluster = (
            summary[
                "rolling_volatility"
            ]
            .idxmin()
        )

        high_vol_cluster = (
            summary[
                "rolling_volatility"
            ]
            .idxmax()
        )

        remaining = (
            set(summary.index)
            - {
                calm_cluster,
                high_vol_cluster
            }
        )

        if len(remaining) != 1:
            raise ValueError(
                "Unable to identify defensive regime."
            )

        defensive_cluster = (
            remaining.pop()
        )

        mapping = {
            calm_cluster: "Calm",
            defensive_cluster:
                "Defensive",
            high_vol_cluster:
                "High Volatility"
        }

        training_features["regime"] = (
            training_features[
                "cluster"
            ].map(mapping)
        )

        test_features["regime"] = (
            test_features[
                "cluster"
            ].map(mapping)
        )

        return (
            training_features,
            test_features
        )

    # ==================================================
    # RISK-AWARE REGIME STRATEGY
    # ==================================================

    def learn_regime_strategy(
        self,
        training_returns,
        training_regimes,
        strategy_returns
    ):
        """
        Learn the best strategy for each regime using
        the same risk-aware scoring approach as V2.

        Criteria:
            Return       = 30%
            Sharpe       = 25%
            Volatility   = 15%
            Drawdown     = 20%
            VaR          = 10%
        """

        strategies = [
            "Maximum Sharpe",
            "Minimum Variance",
            "Risk Parity"
        ]

        data = training_returns[
            ["date"]
        ].copy()

        data["regime"] = (
            training_regimes["regime"].values
        )

        for strategy in strategies:

            data[strategy] = (
                strategy_returns[
                    strategy
                ]
            )

        metric_rows = []

        for regime in sorted(
            data["regime"].unique()
        ):

            regime_data = data[
                data["regime"] == regime
            ]

            for strategy in strategies:

                returns = (
                    regime_data[strategy]
                    .dropna()
                )

                if len(returns) == 0:
                    continue

                metrics = (
                    self.calculate_metrics(
                        returns
                    )
                )

                metric_rows.append(
                    {
                        "Regime": regime,
                        "Strategy": strategy,
                        **metrics,
                        "Observations":
                            len(returns)
                    }
                )

        metrics_df = pd.DataFrame(
            metric_rows
        )

        # Positive risk magnitudes
        metrics_df["Drawdown Risk"] = (
            metrics_df[
                "Maximum Drawdown"
            ].abs()
        )

        metrics_df["VaR Risk"] = (
            metrics_df[
                "VaR (95%)"
            ].abs()
        )

        score_rows = []

        for regime, group in (
            metrics_df.groupby("Regime")
        ):

            group = group.copy()

            score_data = pd.DataFrame(
                index=group.index
            )

            # Higher is better
            for metric in [
                "Annualized Return",
                "Sharpe Ratio"
            ]:

                min_value = (
                    group[metric].min()
                )

                max_value = (
                    group[metric].max()
                )

                if (
                    pd.isna(min_value)
                    or max_value == min_value
                ):

                    score_data[metric] = 1.0

                else:

                    score_data[metric] = (
                        group[metric]
                        - min_value
                    ) / (
                        max_value
                        - min_value
                    )

            # Lower is better
            for metric in [
                "Annualized Volatility",
                "Drawdown Risk",
                "VaR Risk"
            ]:

                min_value = (
                    group[metric].min()
                )

                max_value = (
                    group[metric].max()
                )

                if (
                    pd.isna(min_value)
                    or max_value == min_value
                ):

                    score_data[metric] = 1.0

                else:

                    score_data[metric] = (
                        max_value
                        - group[metric]
                    ) / (
                        max_value
                        - min_value
                    )

            group["Overall Score"] = (
                0.30
                * score_data[
                    "Annualized Return"
                ]
                + 0.25
                * score_data[
                    "Sharpe Ratio"
                ]
                + 0.15
                * score_data[
                    "Annualized Volatility"
                ]
                + 0.20
                * score_data[
                    "Drawdown Risk"
                ]
                + 0.10
                * score_data[
                    "VaR Risk"
                ]
            )

            score_rows.append(
                group
            )

        scores = pd.concat(
            score_rows,
            ignore_index=True
        )

        best_rows = (
            scores.loc[
                scores.groupby(
                    "Regime"
                )[
                    "Overall Score"
                ].idxmax()
            ]
        )

        mapping = (
            best_rows
            .set_index("Regime")[
                "Strategy"
            ]
            .to_dict()
        )

        return (
            metrics_df,
            scores,
            mapping
        )

    # ==================================================
    # APPLY REGIME STRATEGY
    # ==================================================

    def apply_regime_strategy(
        self,
        test_data,
        mapping
    ):
        """
        Apply the frozen regime-to-strategy mapping.
        """

        test_data = test_data.copy()

        test_data[
            "Selected Strategy"
        ] = (
            test_data["regime"]
            .map(mapping)
        )

        test_data[
            "Regime Strategy Return"
        ] = np.nan

        for strategy in [
            "Maximum Sharpe",
            "Minimum Variance",
            "Risk Parity"
        ]:

            mask = (
                test_data[
                    "Selected Strategy"
                ] == strategy
            )

            test_data.loc[
                mask,
                "Regime Strategy Return"
            ] = test_data.loc[
                mask,
                strategy
            ]

        return test_data

    # ==================================================
    # WALK-FORWARD
    # ==================================================

    def run(self):
        """
        Execute expanding-window walk-forward validation.
        """

        asset_columns = (
            self.returns.columns[1:]
        )

        start_test = (
            self.initial_train_size
        )

        window_number = 1

        while (
            start_test < len(self.returns)
        ):

            end_test = min(
                start_test + self.test_size,
                len(self.returns)
            )

            train_data = (
                self.returns.iloc[
                    :start_test
                ].copy()
            )

            test_data = (
                self.returns.iloc[
                    start_test:end_test
                ].copy()
            )

            print(
                f"\nWindow {window_number}"
            )

            print("-" * 70)

            print(
                f"Training: "
                f"{train_data['date'].iloc[0].date()}"
                f" → "
                f"{train_data['date'].iloc[-1].date()}"
            )

            print(
                f"Testing : "
                f"{test_data['date'].iloc[0].date()}"
                f" → "
                f"{test_data['date'].iloc[-1].date()}"
            )

            print(
                f"Observations: "
                f"{len(train_data)} train / "
                f"{len(test_data)} test"
            )

            # ------------------------------------------
            # Training estimates
            # ------------------------------------------

            training_assets = (
                train_data[
                    asset_columns
                ]
            )

            expected_returns = (
                training_assets.mean()
                * 252
            )

            covariance = (
                training_assets.cov()
                * 252
            )

            # ------------------------------------------
            # Three fixed portfolio strategies
            # ------------------------------------------

            max_sharpe_weights = (
                self.optimize_markowitz(
                    expected_returns,
                    covariance,
                    "max_sharpe"
                )
            )

            min_variance_weights = (
                self.optimize_markowitz(
                    expected_returns,
                    covariance,
                    "min_variance"
                )
            )

            risk_parity_weights = (
                self.optimize_risk_parity(
                    covariance
                )
            )

            # ------------------------------------------
            # Frozen strategy returns
            # ------------------------------------------

            train_assets = (
                train_data[
                    asset_columns
                ]
            )

            test_assets = (
                test_data[
                    asset_columns
                ]
            )

            training_strategy_returns = {
                "Maximum Sharpe":
                    train_assets.values
                    @ max_sharpe_weights,

                "Minimum Variance":
                    train_assets.values
                    @ min_variance_weights,

                "Risk Parity":
                    train_assets.values
                    @ risk_parity_weights
            }

            test_strategy_returns = {
                "Maximum Sharpe":
                    test_assets.values
                    @ max_sharpe_weights,

                "Minimum Variance":
                    test_assets.values
                    @ min_variance_weights,

                "Risk Parity":
                    test_assets.values
                    @ risk_parity_weights
            }

            # ------------------------------------------
            # Fit regime model on training data ONLY
            # ------------------------------------------

            scaler, kmeans, train_features = (
                self.fit_regime_model(
                    train_data
                )
            )

            # ------------------------------------------
            # Classify test using training model
            # ------------------------------------------

            full_data = pd.concat(
                [
                    train_data,
                    test_data
                ],
                ignore_index=True
            )

            test_features = (
                self.classify_test_regimes(
                    scaler,
                    kmeans,
                    full_data,
                    test_data
                )
            )

            # ------------------------------------------
            # Meaningful regime names
            # ------------------------------------------

            (
                train_features,
                test_features
            ) = self.assign_regime_labels(
                train_features,
                test_features
            )

            # ------------------------------------------
            # Align regime labels with returns
            # ------------------------------------------

            train_regime_map = (
                train_features[
                    ["date", "regime"]
                ]
            )

            test_regime_map = (
                test_features[
                    ["date", "regime"]
                ]
            )

            train_regime_data = pd.merge(
                train_data,
                train_regime_map,
                on="date",
                how="inner"
            )

            test_regime_data = pd.merge(
                test_data,
                test_regime_map,
                on="date",
                how="inner"
            )

            # ------------------------------------------
            # Strategy returns aligned to dates
            # ------------------------------------------

            train_strategy_df = pd.DataFrame(
                {
                    "date":
                        train_data["date"],
                    "Maximum Sharpe":
                        training_strategy_returns[
                            "Maximum Sharpe"
                        ],
                    "Minimum Variance":
                        training_strategy_returns[
                            "Minimum Variance"
                        ],
                    "Risk Parity":
                        training_strategy_returns[
                            "Risk Parity"
                        ]
                }
            )

            test_strategy_df = pd.DataFrame(
                {
                    "date":
                        test_data["date"],
                    "Maximum Sharpe":
                        test_strategy_returns[
                            "Maximum Sharpe"
                        ],
                    "Minimum Variance":
                        test_strategy_returns[
                            "Minimum Variance"
                        ],
                    "Risk Parity":
                        test_strategy_returns[
                            "Risk Parity"
                        ]
                }
            )

            # ------------------------------------------
            # Learn risk-aware regime mapping
            # ------------------------------------------

            (
                regime_metrics,
                regime_scores,
                strategy_mapping
            ) = self.learn_regime_strategy(
                train_regime_data[
                    ["date"]
                ],
                train_regime_data[
                    ["regime"]
                ],
                train_strategy_df
            )

            # ------------------------------------------
            # Apply frozen mapping to test
            # ------------------------------------------

            test_combined = pd.merge(
                test_regime_data[
                    ["date", "regime"]
                ],
                test_strategy_df,
                on="date",
                how="inner"
            )

            switching_test = (
                self.apply_regime_strategy(
                    test_combined,
                    strategy_mapping
                )
            )

            # ------------------------------------------
            # Fixed strategy metrics
            # ------------------------------------------

            strategies = {
                "Maximum Sharpe":
                    test_strategy_returns[
                        "Maximum Sharpe"
                    ],

                "Minimum Variance":
                    test_strategy_returns[
                        "Minimum Variance"
                    ],

                "Risk Parity":
                    test_strategy_returns[
                        "Risk Parity"
                    ],

                "Risk-Aware Regime Switching":
                    switching_test[
                        "Regime Strategy Return"
                    ].values
            }

            # ------------------------------------------
            # Store results
            # ------------------------------------------

            for strategy, returns in (
                strategies.items()
            ):

                metrics = (
                    self.calculate_metrics(
                        returns
                    )
                )

                row = {
                    "Window":
                        window_number,

                    "Train End Date":
                        train_data[
                            "date"
                        ].iloc[-1],

                    "Test Start Date":
                        test_data[
                            "date"
                        ].iloc[0],

                    "Test End Date":
                        test_data[
                            "date"
                        ].iloc[-1],

                    "Training Observations":
                        len(train_data),

                    "Testing Observations":
                        len(test_data),

                    "Strategy":
                        strategy,

                    **metrics
                }

                self.results.append(
                    row
                )

            # ------------------------------------------
            # Print learned regime mapping
            # ------------------------------------------

            print(
                "\nRisk-Aware Regime Mapping"
            )

            for regime, strategy in (
                strategy_mapping.items()
            ):

                print(
                    f"{regime:<20} → "
                    f"{strategy}"
                )

            print(
                "\nTest Strategy Selection Counts"
            )

            print(
                switching_test[
                    "Selected Strategy"
                ].value_counts()
            )

            window_number += 1

            start_test += self.test_size

        self.results = pd.DataFrame(
            self.results
        )

        return self.results

    # ==================================================
    # SUMMARY
    # ==================================================

    def summarize_results(self):
        """
        Calculate average window-level performance.
        """

        summary = (
            self.results
            .groupby("Strategy")
            [
                [
                    "Total Return",
                    "Annualized Return",
                    "Annualized Volatility",
                    "Sharpe Ratio",
                    "Maximum Drawdown",
                    "VaR (95%)"
                ]
            ]
            .mean()
        )

        print(
            "\nWalk-Forward Average Performance"
        )

        print("-" * 80)

        print(
            summary.to_string(
                float_format=lambda x:
                    f"{x:.4%}"
            )
        )

        return summary
    def calculate_pooled_performance(self):
        """
        Calculate performance using all walk-forward test
        observations as one chronological out-of-sample path.

        This measures the compounded performance across
        all unseen test periods rather than averaging
        window-level metrics.
        """

        strategies = [
            "Maximum Sharpe",
            "Minimum Variance",
            "Risk Parity",
            "Risk-Aware Regime Switching"
        ]

        pooled_returns = {
            strategy: []
            for strategy in strategies
        }

        # Re-run each walk-forward window so that we can
        # collect the actual chronological test returns.
        start_test = self.initial_train_size

        while start_test < len(self.returns):

            end_test = min(
                start_test + self.test_size,
                len(self.returns)
            )

            train_data = self.returns.iloc[
                :start_test
            ].copy()

            test_data = self.returns.iloc[
                start_test:end_test
            ].copy()

            asset_columns = (
                self.returns.columns[1:]
            )

            # ------------------------------------------
            # Training estimates
            # ------------------------------------------

            training_assets = (
                train_data[asset_columns]
            )

            expected_returns = (
                training_assets.mean() * 252
            )

            covariance = (
                training_assets.cov() * 252
            )

            # ------------------------------------------
            # Optimize fixed strategies
            # ------------------------------------------

            max_sharpe_weights = (
                self.optimize_markowitz(
                    expected_returns,
                    covariance,
                    "max_sharpe"
                )
            )

            min_variance_weights = (
                self.optimize_markowitz(
                    expected_returns,
                    covariance,
                    "min_variance"
                )
            )

            risk_parity_weights = (
                self.optimize_risk_parity(
                    covariance
                )
            )

            test_assets = (
                test_data[asset_columns]
            )

            max_sharpe_test = (
                test_assets.values
                @ max_sharpe_weights
            )

            min_variance_test = (
                test_assets.values
                @ min_variance_weights
            )

            risk_parity_test = (
                test_assets.values
                @ risk_parity_weights
            )

            # ------------------------------------------
            # Regime model
            # ------------------------------------------

            scaler, kmeans, train_features = (
                self.fit_regime_model(
                    train_data
                )
            )

            full_data = pd.concat(
                [
                    train_data,
                    test_data
                ],
                ignore_index=True
            )

            test_features = (
                self.classify_test_regimes(
                    scaler,
                    kmeans,
                    full_data,
                    test_data
                )
            )

            (
                train_features,
                test_features
            ) = self.assign_regime_labels(
                train_features,
                test_features
            )

            # ------------------------------------------
            # Training strategy returns
            # ------------------------------------------

            train_assets = (
                train_data[asset_columns]
            )

            training_strategy_returns = {
                "Maximum Sharpe":
                    train_assets.values
                    @ max_sharpe_weights,

                "Minimum Variance":
                    train_assets.values
                    @ min_variance_weights,

                "Risk Parity":
                    train_assets.values
                    @ risk_parity_weights
            }

            train_regime_data = pd.merge(
                train_data[["date"]],
                train_features[
                    ["date", "regime"]
                ],
                on="date",
                how="inner"
            )

            train_strategy_data = pd.DataFrame(
                {
                    "date":
                        train_data["date"],
                    "Maximum Sharpe":
                        training_strategy_returns[
                            "Maximum Sharpe"
                        ],
                    "Minimum Variance":
                        training_strategy_returns[
                            "Minimum Variance"
                        ],
                    "Risk Parity":
                        training_strategy_returns[
                            "Risk Parity"
                        ]
                }
            )

            train_regime_strategy_data = pd.merge(
                train_regime_data,
                train_strategy_data,
                on="date",
                how="inner"
            )

            # ------------------------------------------
            # Learn risk-aware regime strategy
            # ------------------------------------------

            (
                _,
                _,
                strategy_mapping
            ) = self.learn_regime_strategy(
                train_regime_strategy_data[
                    ["date"]
                ],
                train_regime_strategy_data[
                    ["regime"]
                ],
                train_strategy_data
            )

            # ------------------------------------------
            # Test regime + strategy returns
            # ------------------------------------------

            test_regime_data = test_features[
                ["date", "regime"]
            ].copy()

            test_strategy_data = pd.DataFrame(
                {
                    "date":
                        test_data["date"],
                    "Maximum Sharpe":
                        max_sharpe_test,
                    "Minimum Variance":
                        min_variance_test,
                    "Risk Parity":
                        risk_parity_test
                }
            )

            test_combined = pd.merge(
                test_regime_data,
                test_strategy_data,
                on="date",
                how="inner"
            )

            switching_test = (
                self.apply_regime_strategy(
                    test_combined,
                    strategy_mapping
                )
            )

            # ------------------------------------------
            # Store chronological returns
            # ------------------------------------------

            pooled_returns[
                "Maximum Sharpe"
            ].extend(
                max_sharpe_test
            )

            pooled_returns[
                "Minimum Variance"
            ].extend(
                min_variance_test
            )

            pooled_returns[
                "Risk Parity"
            ].extend(
                risk_parity_test
            )

            pooled_returns[
                "Risk-Aware Regime Switching"
            ].extend(
                switching_test[
                    "Regime Strategy Return"
                ].dropna().values
            )

            start_test += self.test_size

        # ----------------------------------------------
        # Calculate pooled metrics
        # ----------------------------------------------

        pooled_metrics = []

        for strategy in strategies:

            metrics = self.calculate_metrics(
                pooled_returns[strategy]
            )

            pooled_metrics.append(
                {
                    "Strategy": strategy,
                    **metrics,
                    "Observations":
                        len(pooled_returns[strategy])
                }
            )

        pooled_table = pd.DataFrame(
            pooled_metrics
        )

        print(
            "\nPooled Walk-Forward Performance"
        )
        print("-" * 85)

        print(
            pooled_table.to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.4%}"
            )
        )

        self.pooled_results = pooled_table

        return pooled_table

    # ==================================================
    # SAVE
    # ==================================================

    def save_results(
        self,
        output_file=(
            "data/processed/"
            "walk_forward_results_v2.csv"
        )
    ):
        """
        Save walk-forward results.
        """

        output_path = Path(
            output_file
        )

        self.results.to_csv(
            output_path,
            index=False
        )

        print(
            "\nWalk-forward results saved to:"
        )

        print(output_path)


if __name__ == "__main__":

    validator = WalkForwardValidator(
        initial_train_size=400,
        test_size=60
    )

    validator.load_returns()

    validator.run()

    validator.summarize_results()

    validator.calculate_pooled_performance()

    validator.save_results()