from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data_loading import load_dataset
from src.model import estimate_models, generate_intervention_plots


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the transport demand and economics coursework analysis.")
    parser.add_argument(
        "--scenario",
        choices=["baseline", "generalised", "totalcost", "intervention", "all"],
        default="all",
        help="Which model or scenario to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory used to store generated artifacts such as summaries, tables, and plots.",
    )
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = Path("data/raw/Dataset 1 2024.csv")
    df = load_dataset(dataset_path)
    requested_models = {args.scenario}
    if args.scenario == "all":
        requested_models = {"baseline", "generalised", "totalcost", "intervention"}
    results = estimate_models(df, requested_models)

    if args.scenario in {"baseline", "all"}:
        parameter_table = results["baseline"].get_estimated_parameters()
        parameter_path = output_dir / "baseline_parameters.csv"
        parameter_table.to_csv(parameter_path)
        print(f"Saved baseline model parameters to: {parameter_path}")
        print("\nBaseline model estimated:")
        print(parameter_table.head())

    if args.scenario in {"generalised", "all"}:
        generalised_path = output_dir / "generalised_model_summary.csv"
        pd.DataFrame({
            "parameter": list(results["generalised"].get_estimated_parameters().index),
            "value": list(results["generalised"].get_estimated_parameters()["Value"]),
        }).to_csv(generalised_path, index=False)
        print(f"Saved generalised model summary to: {generalised_path}")

    if args.scenario in {"totalcost", "all"}:
        totalcost_path = output_dir / "totalcost_model_summary.csv"
        pd.DataFrame({
            "parameter": list(results["totalcost"].get_estimated_parameters().index),
            "value": list(results["totalcost"].get_estimated_parameters()["Value"]),
        }).to_csv(totalcost_path, index=False)
        print(f"Saved total-cost model summary to: {totalcost_path}")

    if args.scenario in {"generalised", "totalcost", "all"}:
        lr_path = output_dir / "likelihood_ratio_tests.csv"
        lr_payload = {"model": [], "statistic": [], "threshold": [], "message": []}
        for model_name, label in [("generalised", "generalised_vs_baseline"), ("totalcost", "totalcost_vs_baseline")]:
            key = f"likelihood_ratio_{model_name}_vs_baseline"
            if key not in results:
                continue
            test = results[key]
            lr_payload["model"].append(label)
            lr_payload["statistic"].append(test.statistic)
            lr_payload["threshold"].append(test.threshold)
            lr_payload["message"].append(test.message)
        pd.DataFrame(lr_payload).to_csv(lr_path, index=False)
        print(f"Saved likelihood-ratio summaries to: {lr_path}")
        for model_name in ("generalised", "totalcost"):
            key = f"likelihood_ratio_{model_name}_vs_baseline"
            if key in results:
                print(f"\n{model_name.title()} model likelihood ratio test:")
                print(results[key])

    if args.scenario in {"intervention", "all"}:
        intervention_summary = generate_intervention_plots(df, results, output_dir)
        for key, path in intervention_summary.items():
            print(f"Saved {key} plot to: {path}")


if __name__ == "__main__":
    main()
