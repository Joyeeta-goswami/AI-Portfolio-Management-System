import pandas as pd
from pathlib import Path


class PortfolioRecommendationEngine:
    """
    Recommend the most suitable portfolio strategy
    using a transparent multi-criteria scoring model.
    """

    def __init__(
        self,
        results_file="data/processed/backtest_results.csv"
    ):
        self.results_file = Path(results_file)
        self.results = None
        self.scores = None

    def load_results(self):
        """
        Load out-of-sample backtest results.
        """

        self.results = pd.read_csv(
            self.results_file,
            index_col="Strategy"
        )

        return self.results

    def calculate_scores(self):
        """
        Calculate a weighted score for each strategy.

        Higher return and Sharpe are better.
        Lower volatility, drawdown magnitude and VaR magnitude
        are better.
        """

        data = self.results.copy()

        # Convert risk measures to positive magnitudes
        data["Drawdown Risk"] = data["Maximum Drawdown"].abs()
        data["VaR Risk"] = data["VaR (95%)"].abs()

        # Metrics where HIGHER is better
        higher_is_better = [
            "Annualized Return",
            "Sharpe Ratio"
        ]

        # Metrics where LOWER is better
        lower_is_better = [
            "Annualized Volatility",
            "Drawdown Risk",
            "VaR Risk"
        ]

        scores = pd.DataFrame(
            index=data.index
        )

        # Normalize higher-is-better metrics
        for metric in higher_is_better:

            minimum = data[metric].min()
            maximum = data[metric].max()

            if maximum == minimum:
                scores[metric] = 1.0
            else:
                scores[metric] = (
                    (data[metric] - minimum)
                    / (maximum - minimum)
                )

        # Normalize lower-is-better metrics
        for metric in lower_is_better:

            minimum = data[metric].min()
            maximum = data[metric].max()

            if maximum == minimum:
                scores[metric] = 1.0
            else:
                scores[metric] = (
                    (maximum - data[metric])
                    / (maximum - minimum)
                )

        # Weights for the decision model
        weights = {
            "Annualized Return": 0.30,
            "Sharpe Ratio": 0.25,
            "Annualized Volatility": 0.15,
            "Drawdown Risk": 0.20,
            "VaR Risk": 0.10
        }

        # Weighted total score
        scores["Overall Score"] = 0.0

        for metric, weight in weights.items():
            scores["Overall Score"] += (
                scores[metric] * weight
            )

        self.scores = scores

        return scores

    def recommend(self):
        """
        Return the strategy with the highest overall score.
        """

        if self.scores is None:
            self.calculate_scores()

        recommended_strategy = (
            self.scores["Overall Score"]
            .idxmax()
        )

        return recommended_strategy

    def print_recommendation(self):
        """
        Print strategy scores and final recommendation.
        """

        recommendation = self.recommend()

        print("\nAI Portfolio Recommendation")
        print("-" * 50)

        print(
            self.scores[
                [
                    "Annualized Return",
                    "Sharpe Ratio",
                    "Annualized Volatility",
                    "Drawdown Risk",
                    "VaR Risk",
                    "Overall Score"
                ]
            ].round(4)
        )

        print("\nRecommended Strategy")
        print("-" * 50)
        print(recommendation)


if __name__ == "__main__":

    engine = PortfolioRecommendationEngine()

    engine.load_results()

    engine.calculate_scores()

    engine.print_recommendation()