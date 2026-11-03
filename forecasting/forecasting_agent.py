"""
===============================================================================
Module : Forecasting Based Agent (Orchestrator & Multi-Agent Ensemble)
File   : forecasting/forecasting_agent.py
Description : Agent autonome de prévision qui orchestre les modèles (GPT, LSTN, PatchTST),
              les HPO, calcule le score de consensus mathématique et produit
              un rapport explicatif de prédiction.
===============================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List

from forecasting.models.gpt_attention import GPTAttentionForecaster
from forecasting.models.lstn import LSTNForecaster
from forecasting.models.attention_models import PatchTSTForecaster
from forecasting.models.gnnst import GNNSTBenchmark
from forecasting.hpo.classic_hpo import ClassicHPO
from forecasting.hpo.agentic_hpo import AgenticHPO
from forecasting.hpo.validation import validate_hpo_config, compute_consensus_score


class ForecastingAgent:
    """
    Agent autonome de prévision pour les séries temporelles multivariées.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            "selection_strategy": "weighted_ensemble",
            "selection_metric": "mae",
            "ensemble_size": 3,
            "weights_method": "inverse_rmse",
            "consensus_threshold": 0.85,
            "fallback_strategy": "best_model",
            "max_debate_rounds": 3,
            "weight_patchtst": 0.40,
            "weight_gpt": 0.40,
            "weight_lstn": 0.20
        }
        if params:
            default_params.update(params)

        self.params = validate_hpo_config("forecasting_agent", default_params)
        self.models = {
            "gpt_attention": GPTAttentionForecaster(),
            "lstn": LSTNForecaster(),
            "patchtst": PatchTSTForecaster(),
            "gnnst_benchmark": GNNSTBenchmark()
        }
        self.classic_hpo = ClassicHPO()
        self.agentic_hpo = AgenticHPO()

    def run_pipeline(
        self,
        raw_df: pd.DataFrame,
        model_name: str = "patchtst",
        hpo_mode: str = "agentic",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Exécute la prévision complète, l'évaluation et la comparaison avec GNNST Benchmark.
        """
        if context is None:
            context = {}

        # 1. Sélection et Validation HPO
        if hpo_mode.lower() == "agentic":
            hpo_params = self.agentic_hpo.optimize(model_name, context)
        elif "kerastuner" in hpo_mode.lower():
            hpo_params = self.classic_hpo.optimize(model_name, framework="kerastuner")
        else:
            hpo_params = self.classic_hpo.optimize(model_name, framework="optuna")

        hpo_params = validate_hpo_config(model_name, hpo_params)

        # 2. Ingestion des données historiques
        X_history = raw_df[["demand", "wind_speed", "rain", "traffic_index"]].values
        y_true = raw_df["demand"].values

        candidate_keys = ["gpt_attention", "lstn", "patchtst"]
        candidate_preds = {}
        candidate_metrics = {}

        # Générer les prédictions individuelles des modèles candidats
        for key in candidate_keys:
            m_inst = self.models[key]
            m_inst.params.update(hpo_params)
            p = m_inst.predict(X_history, horizon=24)
            candidate_preds[key] = p
            candidate_metrics[key] = m_inst.evaluate(y_true, p)

        # 3. Calcul du Score de Consensus Mathématique
        preds_list = list(candidate_preds.values())
        consensus_score = compute_consensus_score(preds_list)
        consensus_threshold = float(self.params.get("consensus_threshold", 0.85))

        # 4. Calcul de l'Ensemble ou Sélection selon la Stratégie
        weights_method = self.params.get("weights_method", "inverse_rmse")
        
        if weights_method == "inverse_rmse":
            eps = 1e-8
            inv_errors = [1.0 / (candidate_metrics[k]["RMSE"] + eps) for k in candidate_keys]
            tot_inv = sum(inv_errors)
            weights = {k: inv_errors[i] / tot_inv for i, k in enumerate(candidate_keys)}
        elif weights_method == "inverse_mae":
            eps = 1e-8
            inv_errors = [1.0 / (candidate_metrics[k]["MAE"] + eps) for k in candidate_keys]
            tot_inv = sum(inv_errors)
            weights = {k: inv_errors[i] / tot_inv for i, k in enumerate(candidate_keys)}
        elif weights_method == "uniform":
            weights = {k: 1.0 / len(candidate_keys) for k in candidate_keys}
        else: # fixed
            weights = {
                "patchtst": float(self.params.get("weight_patchtst", 0.40)),
                "gpt_attention": float(self.params.get("weight_gpt", 0.40)),
                "lstn": float(self.params.get("weight_lstn", 0.20))
            }

        ensemble_pred = sum(weights[k] * candidate_preds[k] for k in candidate_keys)
        best_single_key = min(candidate_keys, key=lambda k: candidate_metrics[k]["RMSE"])

        # Décision basée sur le seuil de consensus
        if consensus_score >= consensus_threshold or self.params.get("selection_strategy") == "weighted_ensemble":
            target_pred = ensemble_pred
            selected_label = f"Weighted Ensemble ({weights_method.upper()})"
        else:
            # Fallback strategy
            fallback = self.params.get("fallback_strategy", "best_model")
            target_pred = candidate_preds[best_single_key]
            selected_label = f"Fallback [{fallback}] ({self.models[best_single_key].name})"

        # 5. Évaluation du Benchmark GNNST
        gnnst_model = self.models["gnnst_benchmark"]
        gnnst_pred = gnnst_model.predict(X_history, horizon=24)

        # 6. Évaluation des Métriques Finales
        target_model_instance = self.models.get(model_name if model_name in self.models else best_single_key)
        target_metrics = target_model_instance.evaluate(y_true, target_pred)
        gnnst_metrics = gnnst_model.evaluate(y_true, gnnst_pred)

        rmse_imprv = round((1.0 - target_metrics["RMSE"] / (gnnst_metrics["RMSE"] + 1e-8)) * 100.0, 2)

        return {
            "selected_model": selected_label,
            "hpo_mode": hpo_mode.upper(),
            "hpo_params": hpo_params,
            "prediction": target_pred,
            "gnnst_prediction": gnnst_pred,
            "target_metrics": target_metrics,
            "gnnst_metrics": gnnst_metrics,
            "consensus_score": round(consensus_score * 100.0, 2),
            "consensus_threshold_pct": round(consensus_threshold * 100.0, 2),
            "weights_applied": weights,
            "rmse_improvement_pct": max(0.0, rmse_imprv),
            "reasoning_report": (
                f"L'Agent de Prévision a sélectionné '{selected_label}' avec HPO Mode '{hpo_mode.upper()}'.\n"
                f"• Score de Consensus Mathématique : {round(consensus_score * 100.0, 2)}% (Seuil = {round(consensus_threshold * 100.0, 2)}%)\n"
                f"• Modèle Sélectionné  : RMSE = {target_metrics['RMSE']}, MAE = {target_metrics['MAE']}, R² = {target_metrics['R2']}\n"
                f"• Benchmark Baseline (GNNST) : RMSE = {gnnst_metrics['RMSE']}, MAE = {gnnst_metrics['MAE']}, R² = {gnnst_metrics['R2']}\n"
                f"• Gain de précision par rapport au Benchmark GNNST : +{max(0.0, rmse_imprv)}%"
            )
        }
