import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class MarketRegimeDetector:
    """
    Detect market regimes using rolling market-level features.

    Important:
    Regime features for day t use information available BEFORE
    day t's portfolio return. This avoids look-ahead bias.

    K-Means is fitted only on the training period.
    The fitted model is then used to classify the test period.
    """

    def __init__(
        self,
        returns_file="data/processed/returns.csv",
        train_ratio=0.80,
        rolling_window=20,
        n_regimes=3
    ):
        self.returns_file = Path(returns_file)
        self.train_ratio = train_ratio
        self.rolling_window = rolling_window
        self.n_regimes = n_regimes

        self.returns = None
        self.train_returns = None
        self.test_returns = None

        self.features = None
        self.train_features = None
        self.test_features = None

        self.scaler = None
        self.kmeans = None
        self.label_mapping = None

    def load_returns(self):
        """
        Load historical daily asset returns.
        """

        self.returns = pd.read_csv(
            self.returns_file,
            parse_dates=["date"]
        )

        self.returns = self.returns.sort_values(
            "date"
        ).reset_index(drop=True)

        print("Returns loaded successfully.")

    def split_data(self):
        """
        Split returns chronologically into training
        and testing periods.
        """

        split_index = int(
            len(self.returns) * self.train_ratio
        )

        self.train_returns = self.returns.iloc[
            :split_index
        ].copy()

        self.test_returns = self.returns.iloc[
            split_index:
        ].copy()

        print("\nRegime Model Data Split")
        print("-" * 60)

        print(
            f"Training observations : "
            f"{len(self.train_returns)}"
        )

        print(
            f"Testing observations  : "
            f"{len(self.test_returns)}"
        )

        print(
            f"Training period       : "
            f"{self.train_returns['date'].iloc[0].date()}"
            f" → "
            f"{self.train_returns['date'].iloc[-1].date()}"
        )

        print(
            f"Testing period        : "
            f"{self.test_returns['date'].iloc[0].date()}"
            f" → "
            f"{self.test_returns['date'].iloc[-1].date()}"
        )

    def create_features(self):
        """
        Create market regime features for the COMPLETE
        return history.

        Features for day t are based on the previous
        rolling_window observations, excluding day t itself.

        This prevents look-ahead bias when the regime is
        used to select a portfolio for day t.
        """

        asset_returns = self.returns.iloc[:, 1:].copy()

        # Equal-weighted market return
        market_return = asset_returns.mean(axis=1)

        # Previous-window rolling return
        rolling_return = (
            market_return
            .rolling(
                self.rolling_window,
                min_periods=self.rolling_window
            )
            .mean()
            .shift(1)
        )

        # Previous-window annualized volatility
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

        # Average pairwise correlation
        rolling_correlation = []

        for i in range(len(asset_returns)):

            # Need previous rolling_window observations
            # and must exclude current day i
            if i < self.rolling_window:
                rolling_correlation.append(np.nan)
                continue

            window = asset_returns.iloc[
                i - self.rolling_window:i
            ]

            correlation_matrix = window.corr()

            upper_triangle = correlation_matrix.where(
                np.triu(
                    np.ones(
                        correlation_matrix.shape
                    ),
                    k=1
                ).astype(bool)
            )

            rolling_correlation.append(
                upper_triangle.stack().mean()
            )

        self.features = pd.DataFrame(
            {
                "date": self.returns["date"],
                "rolling_return": rolling_return,
                "rolling_volatility": rolling_volatility,
                "rolling_correlation": rolling_correlation
            }
        )

        return self.features

    def prepare_features(self):
        """
        Split the complete feature dataset into training
        and test periods.

        Because the rolling features were constructed from
        the complete history, the test period retains all
        128 observations once sufficient pre-test history exists.
        """

        split_date = self.test_returns["date"].iloc[0]

        self.train_features = self.features[
            self.features["date"] < split_date
        ].copy()

        self.test_features = self.features[
            self.features["date"] >= split_date
        ].copy()

        # Remove only rows that do not have enough historical
        # observations to calculate the rolling features.
        self.train_features = (
            self.train_features
            .dropna()
            .reset_index(drop=True)
        )

        self.test_features = (
            self.test_features
            .dropna()
            .reset_index(drop=True)
        )

        print("\nFeature Preparation")
        print("-" * 60)

        print(
            f"Training feature observations : "
            f"{len(self.train_features)}"
        )

        print(
            f"Testing feature observations  : "
            f"{len(self.test_features)}"
        )

    def fit_training_regimes(self):
        """
        Fit StandardScaler and K-Means using TRAINING
        features only.
        """

        feature_columns = [
            "rolling_return",
            "rolling_volatility",
            "rolling_correlation"
        ]

        X_train = self.train_features[
            feature_columns
        ]

        self.scaler = StandardScaler()

        X_train_scaled = self.scaler.fit_transform(
            X_train
        )

        self.kmeans = KMeans(
            n_clusters=self.n_regimes,
            random_state=42,
            n_init=10
        )

        self.train_features["regime_cluster"] = (
            self.kmeans.fit_predict(
                X_train_scaled
            )
        )

        print(
            "\nK-Means fitted using training data only."
        )

    def classify_test_regimes(self):
        """
        Classify test observations using the already
        fitted scaler and K-Means model.
        """

        feature_columns = [
            "rolling_return",
            "rolling_volatility",
            "rolling_correlation"
        ]

        X_test = self.test_features[
            feature_columns
        ]

        X_test_scaled = self.scaler.transform(
            X_test
        )

        self.test_features["regime_cluster"] = (
            self.kmeans.predict(
                X_test_scaled
            )
        )

        print(
            "Test regimes classified using training model."
        )

    def assign_regime_labels(self):
        """
        Assign economically meaningful names to clusters
        based only on training-period characteristics.
        """

        summary = (
            self.train_features
            .groupby("regime_cluster")
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
            summary["rolling_volatility"].idxmin()
        )

        high_vol_cluster = (
            summary["rolling_volatility"].idxmax()
        )

        remaining_clusters = (
            set(summary.index)
            - {calm_cluster, high_vol_cluster}
        )

        if len(remaining_clusters) != 1:
            raise ValueError(
                "Unable to uniquely identify defensive regime."
            )

        defensive_cluster = (
            remaining_clusters.pop()
        )

        self.label_mapping = {
            calm_cluster: "Calm",
            defensive_cluster: "Defensive",
            high_vol_cluster: "High Volatility"
        }

        self.train_features["regime"] = (
            self.train_features["regime_cluster"]
            .map(self.label_mapping)
        )

        self.test_features["regime"] = (
            self.test_features["regime_cluster"]
            .map(self.label_mapping)
        )

    def summarize_training_regimes(self):
        """
        Display average characteristics of each training regime.
        """

        summary = (
            self.train_features
            .groupby("regime")
            [
                [
                    "rolling_return",
                    "rolling_volatility",
                    "rolling_correlation"
                ]
            ]
            .mean()
        )

        print("\nTraining Regime Summary")
        print("-" * 65)

        print(
            summary.to_string(
                float_format=lambda x: f"{x:.6f}"
            )
        )

        return summary

    def print_regime_counts(self):
        """
        Display regime observation counts.
        """

        print("\nTraining Regime Counts")
        print("-" * 50)

        print(
            self.train_features["regime"]
            .value_counts()
        )

        print("\nTesting Regime Counts")
        print("-" * 50)

        print(
            self.test_features["regime"]
            .value_counts()
        )

    def print_latest_test_regime(self):
        """
        Display the most recent classified test regime.
        """

        latest = self.test_features.iloc[-1]

        print("\nLatest Test Market Regime")
        print("-" * 55)

        print(
            f"Date                : "
            f"{latest['date'].date()}"
        )

        print(
            f"Regime              : "
            f"{latest['regime']}"
        )

        print(
            f"Rolling Return      : "
            f"{latest['rolling_return']:.4%}"
        )

        print(
            f"Rolling Volatility  : "
            f"{latest['rolling_volatility']:.4%}"
        )

        print(
            f"Rolling Correlation : "
            f"{latest['rolling_correlation']:.4f}"
        )

    def save_regime_data(
        self,
        output_file="data/processed/market_regimes.csv"
    ):
        """
        Save training and test regime classifications.
        """

        train_output = self.train_features.copy()
        train_output["dataset"] = "train"

        test_output = self.test_features.copy()
        test_output["dataset"] = "test"

        combined = pd.concat(
            [train_output, test_output],
            ignore_index=True
        )

        output_path = Path(output_file)

        combined.to_csv(
            output_path,
            index=False
        )

        print("\nMarket regime data saved to:")
        print(output_path)

        return combined


if __name__ == "__main__":

    detector = MarketRegimeDetector()

    detector.load_returns()

    detector.split_data()

    detector.create_features()

    detector.prepare_features()

    detector.fit_training_regimes()

    detector.classify_test_regimes()

    detector.assign_regime_labels()

    detector.summarize_training_regimes()

    detector.print_regime_counts()

    detector.print_latest_test_regime()

    detector.save_regime_data()