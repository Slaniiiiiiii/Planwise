from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.planwise_ml import train_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Train all PlanWise Random Forest models.")
    parser.add_argument("--data", default="data/planwise_ml_dataset.csv")
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--iterations", type=int, default=8)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    frame = pd.read_csv(data_path)
    def show_progress(event: dict) -> None:
        if event["phase"] == "tuning":
            name = event["model"].replace("_", " ").title()
            print(
                f"[{event['model_index']}/{event['model_count']}] {name} — "
                f"iteration {event['iteration']}/{event['total_iterations']} — "
                f"score {event['score']:.4f} — best {event['best_score']:.4f} — "
                f"{event['elapsed_seconds']:.1f}s",
                flush=True,
            )
        elif event["phase"] == "model_complete":
            print(f"✓ {event['message']}", flush=True)

    print(f"Training 4 Random Forest models with {args.iterations} tuning iterations each…", flush=True)
    result = train_all(frame, args.output, iterations=args.iterations, progress_callback=show_progress)
    delay = result["metrics"]["delay_classifier"]
    risk = result["metrics"]["risk_classifier"]
    print(f"Delay ROC-AUC: {delay['roc_auc']:.3f}; recall: {delay['recall']:.3f}")
    print(f"Risk macro-F1: {risk['macro_f1']:.3f}")
    print(f"Artifacts saved to: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
