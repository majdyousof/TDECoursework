from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from biogeme import models
from biogeme.biogeme import BIOGEME
from biogeme.database import Database
from biogeme.expressions import Beta
from biogeme.tools import likelihood_ratio_test


def build_database(df: pd.DataFrame) -> Database:
    """Build the Biogeme database from the coursework dataset."""
    cleaned_df = df.copy()
    for col in cleaned_df.columns:
        if pd.api.types.is_categorical_dtype(cleaned_df[col]):
            cleaned_df[col] = cleaned_df[col].astype(str)

    dbase = Database("Dataset 1 2024", cleaned_df)
    globals().update(dbase.variables)
    return dbase


def define_model_parameters() -> tuple[dict, dict]:
    """Create the Biogeme utility parameters for the multinomial logit model."""
    beta = {
        "BetaTT": Beta("BetaTT", 0, None, None, 0),
        "BetaTC": Beta("BetaTC", 0, None, None, 0),
        "BetaTC_car": Beta("BetaTC_car", 0, None, None, 0),
        "BetaTC_PT": Beta("BetaTC_PT", 0, None, None, 0),
        "BetaTC_cycling": Beta("BetaTC_cycling", 0, None, None, 0),
        "BetaTC_walking": Beta("BetaTC_walking", 0, None, None, 0),
        "Beta0_car": Beta("Beta0_car", 0, None, None, 0),
        "BetaTT_car": Beta("BetaTT_car", 0, None, None, 0),
        "BetaFuel_car": Beta("BetaFuel_car", 0, None, None, 0),
        "BetaParking_car": Beta("BetaParking_car", 0, None, None, 0),
        "BetaCongestion_car": Beta("BetaCongestion_car", 0, None, None, 0),
        "Beta0_PT": Beta("Beta0_PT", 0, None, None, 0),
        "BetaTT_PT": Beta("BetaTT_PT", 0, None, None, 0),
        "BetaFare_PT": Beta("BetaFare_PT", 0, None, None, 0),
        "BetaAC_PT": Beta("BetaAC_PT", 0, None, None, 0),
        "BetaAT_PT": Beta("BetaAT_PT", 0, None, None, 0),
        "BetaWT_PT": Beta("BetaWT_PT", 0, None, None, 0),
        "Beta0_cycling": Beta("Beta0_cycling", 0, None, None, 0),
        "BetaTT_cycling": Beta("BetaTT_cycling", 0, None, None, 0),
        "BetaBikeability_cycling": Beta("BetaBikeability_cycling", 0, None, None, 0),
        "Beta0_walking": Beta("Beta0_walking", 0, None, None, 1),
        "BetaTT_walking": Beta("BetaTT_walking", 0, None, None, 0),
        "BetaWalkability_walking": Beta("BetaWalkability_walking", 0, None, None, 0),
        "BetaAge": Beta("BetaAge", 0, None, None, 0),
        "BetaGender": Beta("BetaGender", 0, None, None, 0),
        "BetaIncome": Beta("BetaIncome", 0, None, None, 0),
        "BetaCarOwnership": Beta("BetaCarOwnership", 0, None, None, 0),
    }

    utilities = {
        "baseline_user": (
            beta["BetaAge"] * Beta("AGE", 1, None, None, 0) if False else None
        )
    }
    return beta, utilities


def estimate_models(df: pd.DataFrame, scenarios: set[str] | None = None) -> dict:
    """Estimate only the model specifications required by the selected scenarios."""
    scenarios = scenarios or {"baseline", "generalised", "totalcost"}
    required_models = set(scenarios)
    if required_models.intersection({"generalised", "totalcost", "intervention"}):
        required_models.add("baseline")

    dbase = build_database(df)
    globals().update(dbase.variables)

    beta = define_model_parameters()[0]

    baseline_user = beta["BetaAge"] * AGE + beta["BetaGender"] * GENDER + beta["BetaIncome"] * INCOME + beta["BetaCarOwnership"] * CAR_OWNERSHIP

    V_baseline_car = (
        beta["Beta0_car"]
        + beta["BetaTT_car"] * TT_CAR
        + beta["BetaFuel_car"] * FC_CAR
        + beta["BetaParking_car"] * PC_CAR
        + beta["BetaCongestion_car"] * CC_CAR
        + baseline_user
    )
    V_baseline_PT = (
        beta["Beta0_PT"]
        + beta["BetaTT_PT"] * TT_PT
        + beta["BetaFare_PT"] * FARE_PT
        + beta["BetaAC_PT"] * AC_PT
        + beta["BetaAT_PT"] * AT_PT
        + beta["BetaWT_PT"] * WT_PT
        + baseline_user
    )
    V_baseline_cycling = (
        beta["Beta0_cycling"]
        + beta["BetaTT_cycling"] * TT_CYCLE
        + beta["BetaBikeability_cycling"] * BIKEABILITY_INDEX
        + baseline_user
    )
    V_baseline_walking = (
        beta["Beta0_walking"]
        + beta["BetaTT_walking"] * TT_WALK
        + beta["BetaWalkability_walking"] * WALKABILITY_INDEX
        + baseline_user
    )

    V_generalised_car = beta["Beta0_car"] + beta["BetaTT"] * TT_CAR + beta["BetaTC"] * (FC_CAR + PC_CAR + CC_CAR) + baseline_user
    V_generalised_PT = beta["Beta0_PT"] + beta["BetaTT"] * TT_PT + beta["BetaTC"] * (FARE_PT + AC_PT + AT_PT + WT_PT) + baseline_user
    V_generalised_cycling = beta["Beta0_cycling"] + beta["BetaTT"] * TT_CYCLE + beta["BetaTC"] * BIKEABILITY_INDEX + baseline_user
    V_generalised_walking = beta["Beta0_walking"] + beta["BetaTT"] * TT_WALK + beta["BetaTC"] * WALKABILITY_INDEX + baseline_user

    V_totalcost_car = beta["Beta0_car"] + beta["BetaTT_car"] * TT_CAR + beta["BetaTC_car"] * (FC_CAR + PC_CAR + CC_CAR) + baseline_user
    V_totalcost_PT = beta["Beta0_PT"] + beta["BetaTT_PT"] * TT_PT + beta["BetaTC_PT"] * (FARE_PT + AC_PT + AT_PT + WT_PT) + baseline_user
    V_totalcost_cycling = beta["Beta0_cycling"] + beta["BetaTT_cycling"] * TT_CYCLE + beta["BetaTC_cycling"] * BIKEABILITY_INDEX + baseline_user
    V_totalcost_walking = beta["Beta0_walking"] + beta["BetaTT_walking"] * TT_WALK + beta["BetaTC_walking"] * WALKABILITY_INDEX + baseline_user

    utilities = {
        4: V_baseline_car,
        3: V_baseline_PT,
        2: V_baseline_cycling,
        1: V_baseline_walking,
    }
    utilities_generalised = {
        4: V_generalised_car,
        3: V_generalised_PT,
        2: V_generalised_cycling,
        1: V_generalised_walking,
    }
    utilities_totalcost = {
        4: V_totalcost_car,
        3: V_totalcost_PT,
        2: V_totalcost_cycling,
        1: V_totalcost_walking,
    }
    availability = {4: 1, 3: 1, 2: 1, 1: 1}

    logprob_base = models.loglogit(utilities, availability, CHOSEN_MODE)
    logprob_generalised = models.loglogit(utilities_generalised, availability, CHOSEN_MODE)
    logprob_totalcost = models.loglogit(utilities_totalcost, availability, CHOSEN_MODE)

    baseline_model = BIOGEME(dbase, logprob_base)
    baseline_model.modelName = "baseline_model"
    generalised_model = BIOGEME(dbase, logprob_generalised)
    generalised_model.modelName = "baseline_generalised_model"
    totalcost_model = BIOGEME(dbase, logprob_totalcost)
    totalcost_model.modelName = "baseline_totalcost_model"

    results = {}
    if "baseline" in required_models:
        results["baseline"] = baseline_model.estimate()
    if "generalised" in required_models:
        results["generalised"] = generalised_model.estimate()
    if "totalcost" in required_models:
        results["totalcost"] = totalcost_model.estimate()

    baseline_stats = results["baseline"].getGeneralStatistics()
    LRbase = [baseline_stats["Final log likelihood"].value, baseline_stats["Number of estimated parameters"].value]
    if "generalised" in results:
        generalised_stats = results["generalised"].getGeneralStatistics()
        LRgeneralised = [generalised_stats["Final log likelihood"].value, generalised_stats["Number of estimated parameters"].value]
        results["likelihood_ratio_generalised_vs_baseline"] = likelihood_ratio_test(LRgeneralised, LRbase)
    if "totalcost" in results:
        totalcost_stats = results["totalcost"].getGeneralStatistics()
        LRtotalcost = [totalcost_stats["Final log likelihood"].value, totalcost_stats["Number of estimated parameters"].value]
        results["likelihood_ratio_totalcost_vs_baseline"] = likelihood_ratio_test(LRtotalcost, LRbase)

    return results


def generate_intervention_plots(df: pd.DataFrame, results: dict, output_dir: str | Path) -> dict[str, Path]:
    """Run the report's 400 intervention scenarios with sample enumeration."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_df = df.copy()
    base_df.rename(columns={"Unnamed: 0": "ID", "CAR OWNERSHIP": "CAR_OWNERSHIP"}, inplace=True, errors="ignore")
    percentage_times = np.linspace(0, 1, 20)
    percentage_costs = np.linspace(0, 1, 20)
    values = {column: base_df[column].to_numpy(dtype=float) for column in [
        "TT_WALK", "WALKABILITY_INDEX", "TT_CYCLE", "BIKEABILITY_INDEX", "TT_PT",
        "FARE_PT", "WT_PT", "AT_PT", "AC_PT", "TT_CAR", "FC_CAR", "PC_CAR",
        "CC_CAR", "INCOME", "GENDER", "AGE", "CAR_OWNERSHIP",
    ]}
    beta = results["baseline"].get_beta_values()
    baseline_user = (
        beta["BetaAge"] * values["AGE"]
        + beta["BetaGender"] * values["GENDER"]
        + beta["BetaIncome"] * values["INCOME"]
        + beta["BetaCarOwnership"] * values["CAR_OWNERSHIP"]
    )
    utilities = {
        "car": beta["Beta0_car"] + beta["BetaTT_car"] * values["TT_CAR"] + beta["BetaFuel_car"] * values["FC_CAR"] + beta["BetaParking_car"] * values["PC_CAR"] + beta["BetaCongestion_car"] * values["CC_CAR"] + baseline_user,
        "cycling": beta["Beta0_cycling"] + beta["BetaTT_cycling"] * values["TT_CYCLE"] + beta["BetaBikeability_cycling"] * values["BIKEABILITY_INDEX"] + baseline_user,
        "walking": 1.0 + beta["BetaTT_walking"] * values["TT_WALK"] + beta["BetaWalkability_walking"] * values["WALKABILITY_INDEX"] + baseline_user,
    }
    pt_base = beta["Beta0_PT"] + beta["BetaTT_PT"] * values["TT_PT"] + beta["BetaFare_PT"] * values["FARE_PT"] + beta["BetaAC_PT"] * values["AC_PT"] + beta["BetaAT_PT"] * values["AT_PT"] + beta["BetaWT_PT"] * values["WT_PT"] + baseline_user

    # Broadcast all observations over the 20 x 20 intervention grid, then average.
    time_reduction = percentage_times[:, None, None]
    cost_reduction = percentage_costs[None, :, None]
    pt_utility = pt_base[None, None, :] - beta["BetaAC_PT"] * values["AC_PT"][None, None, :] * cost_reduction - beta["BetaAT_PT"] * values["AT_PT"][None, None, :] * time_reduction
    utility_stack = np.stack([
        np.broadcast_to(utilities["car"], pt_utility.shape),
        pt_utility,
        np.broadcast_to(utilities["cycling"], pt_utility.shape),
        np.broadcast_to(utilities["walking"], pt_utility.shape),
    ])
    utility_stack -= utility_stack.max(axis=0, keepdims=True)
    probabilities = np.exp(utility_stack)
    probabilities /= probabilities.sum(axis=0, keepdims=True)
    market_shares = probabilities.mean(axis=3)

    figures = {}
    plot_names = {
        "public_transport": market_shares[1],
        "cycling": market_shares[2],
        "walking": market_shares[3],
        "car": market_shares[0],
    }

    for name, values in plot_names.items():
        fig, ax = plt.subplots(figsize=(8, 6))
        X, Y = np.meshgrid(percentage_times, percentage_costs)
        contour = ax.contourf(X, Y, values, levels=20)
        fig.colorbar(contour, ax=ax)
        ax.set_title(f"Market share for {name.replace('_', ' ').title()}")
        ax.set_xlabel("Reduction in access time")
        ax.set_ylabel("Reduction in access costs")
        path = output_dir / "plots" / f"{name}_market_share.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        figures[name] = path

    return figures
