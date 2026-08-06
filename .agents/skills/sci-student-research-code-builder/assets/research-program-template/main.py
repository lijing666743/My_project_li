from src.config import interactive_config
from src.dashboard import plot_dashboard
from src.logging_utils import write_metrics_csv
from src.runner import run_training


def main() -> None:
    config = interactive_config()
    metrics = run_training(config)
    csv_path = write_metrics_csv(metrics, config)
    png_path = plot_dashboard(metrics, config)
    print(f"Saved metrics: {csv_path}")
    print(f"Saved dashboard: {png_path}")


if __name__ == "__main__":
    main()
