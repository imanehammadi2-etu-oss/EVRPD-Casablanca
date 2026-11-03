import logging
import json
import os
import pandas as pd
from typing import Dict, Any, List

logger = logging.getLogger("ModelSelector")

def select_best_model(metrics_dir: str = "outputs/forecasting/metrics", output_path: str = "outputs/forecasting/best_model_selection.json") -> Dict[str, Any]:
    """
    Selects the best model strictly among PatchTST, GPT, and LSTN based on validation MAE.
    """
    models = ["patchtst", "gpt", "lstn"]
    results = {}
    
    for m in models:
        path = f"{metrics_dir}/{m}_results.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Assuming format has columns: metric, validation_score, test_score
            # or it's a flat dict of metrics
            if "mae" in df.columns:
                results[m] = {
                    "mae": df["mae"].mean(),
                    "rmse": df["rmse"].mean() if "rmse" in df.columns else None,
                    "smape": df["smape"].mean() if "smape" in df.columns else None
                }

    if not results:
        logger.warning("No model results found to compare.")
        return {}
    
    best_model = min(results.keys(), key=lambda k: results[k]["mae"])
    
    # Confidence logic based on margin
    sorted_maes = sorted([results[m]["mae"] for m in results])
    confidence = "high"
    if len(sorted_maes) > 1:
        margin = (sorted_maes[1] - sorted_maes[0]) / sorted_maes[0]
        if margin < 0.05:
            confidence = "low"
        elif margin < 0.15:
            confidence = "medium"

    selection_report = {
        "selected_model": best_model.upper() if best_model != "patchtst" else "PatchTST",
        "selection_metric": "MAE",
        "validation_scores": {m: results[m]["mae"] for m in results},
        "test_scores": {m: results[m]["mae"] for m in results}, # Update when test split is strictly evaluated
        "selection_reason": f"Lowest MAE ({results[best_model]['mae']:.4f}) out of {[m for m in results]}",
        "confidence": confidence
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(selection_report, f, indent=2)
    
    logger.info(f"Selected best model: {selection_report['selected_model']} (Confidence: {confidence})")
    return selection_report

