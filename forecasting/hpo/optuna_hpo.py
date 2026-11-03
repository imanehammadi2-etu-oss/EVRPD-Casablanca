"""
===============================================================================
Module : Optuna HPO Engine
File   : forecasting/hpo/optuna_hpo.py
Description : Optimisation d'hyperparamètres bayésienne avec Optuna
              uniquement sur Train + Validation sans fuite sur Test.
===============================================================================
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple, List

from forecasting.hpo.validation import validate_hpo_config, compute_consensus_score

logger = logging.getLogger("HPOEngine")


def build_model_from_trial(model_name: str, trial: Any, config: Optional[Dict[str, Any]] = None) -> Any:
    """
    Construit une instance de modèle de prévision à partir d'un trial Optuna.
    
    Args:
        model_name: Nom du modèle ("patchtst", "gpt_attention", "lstn", "forecasting_agent")
        trial: Objet optuna.Trial
        config: Configuration globale optionnelle
        
    Returns:
        Instance du modèle configuré et validé.
    """
    from forecasting.models.attention_models import PatchTSTForecaster
    from forecasting.models.gpt_attention import GPTAttentionForecaster
    from forecasting.models.lstn import LSTNForecaster
    from forecasting.forecasting_agent import ForecastingAgent

    m = model_name.lower().strip()
    
    if "patchtst" in m:
        ctx_len = trial.suggest_categorical("context_length", [24, 48, 96, 168])
        pred_len = trial.suggest_categorical("prediction_length", [6, 12, 24])
        p_len = trial.suggest_categorical("patch_len", [4, 8, 12, 16, 24])
        # Ne garder que p_len <= ctx_len
        if p_len > ctx_len:
            p_len = min(p_len, ctx_len)
            
        stride_mode = trial.suggest_categorical("patch_stride_mode", ["half_patch", "full_patch"])
        stride = max(1, p_len // 2) if stride_mode == "half_patch" else p_len
        
        d_model = trial.suggest_categorical("d_model", [64, 128, 256])
        n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])
        if d_model % n_heads != 0:
            d_model = ((d_model // n_heads) + 1) * n_heads
            
        params = {
            "context_length": ctx_len,
            "sequence_length": ctx_len,
            "prediction_length": pred_len,
            "patch_len": p_len,
            "stride": stride,
            "patch_stride_mode": stride_mode,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": trial.suggest_int("n_layers", 2, 6),
            "d_ff_multiplier": trial.suggest_categorical("d_ff_multiplier", [2, 4]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.30),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        }
        val_params = validate_hpo_config("patchtst", params)
        return PatchTSTForecaster(params=val_params)

    elif "gpt" in m:
        ctx_len = trial.suggest_categorical("context_length", [48, 96, 168, 336])
        pred_len = trial.suggest_categorical("prediction_length", [6, 12, 24])
        p_len = trial.suggest_categorical("patch_len", [4, 8, 12, 16, 24])
        if p_len > ctx_len:
            p_len = min(p_len, ctx_len)

        stride_mode = trial.suggest_categorical("patch_stride_mode", ["half_patch", "full_patch"])
        stride = max(1, p_len // 2) if stride_mode == "half_patch" else p_len

        d_model = trial.suggest_categorical("d_model", [64, 128, 256])
        n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])
        if d_model % n_heads != 0:
            d_model = ((d_model // n_heads) + 1) * n_heads

        freeze = trial.suggest_categorical("freeze_backbone", [True, False])
        t_layers = 0 if freeze else trial.suggest_categorical("trainable_layers", [0, 1, 2, 4])

        params = {
            "context_length": ctx_len,
            "sequence_length": ctx_len,
            "prediction_length": pred_len,
            "patch_len": p_len,
            "stride": stride,
            "patch_stride": stride,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": trial.suggest_categorical("n_layers", [2, 4, 6]),
            "d_ff_multiplier": trial.suggest_categorical("d_ff_multiplier", [2, 4]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.30),
            "attention_dropout": trial.suggest_float("attention_dropout", 0.0, 0.20),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "freeze_backbone": freeze,
            "trainable_layers": t_layers,
            "use_revin": trial.suggest_categorical("use_revin", [True, False])
        }
        val_params = validate_hpo_config("gpt_attention", params)
        return GPTAttentionForecaster(params=val_params)

    elif "lstn" in m:
        ctx_len = trial.suggest_categorical("context_length", [24, 48, 96, 168])
        pred_len = trial.suggest_categorical("prediction_length", [6, 12, 24])
        short_win = trial.suggest_categorical("short_term_window", [6, 12, 24])
        if short_win > ctx_len:
            short_win = min(short_win, ctx_len)

        d_model = trial.suggest_categorical("d_model", [64, 128, 256])
        n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])
        if d_model % n_heads != 0:
            d_model = ((d_model // n_heads) + 1) * n_heads

        params = {
            "context_length": ctx_len,
            "sequence_length": ctx_len,
            "prediction_length": pred_len,
            "short_term_window": short_win,
            "rnn_type": trial.suggest_categorical("rnn_type", ["LSTM", "GRU"]),
            "lstm_units": trial.suggest_categorical("lstm_units", [32, 64, 128, 256]),
            "num_lstm_layers": trial.suggest_int("num_lstm_layers", 1, 3),
            "rnn_dropout": trial.suggest_float("rnn_dropout", 0.0, 0.30),
            "recurrent_dropout": trial.suggest_float("recurrent_dropout", 0.0, 0.30),
            "attention_units": trial.suggest_categorical("attention_units", [32, 64, 128, 256]),
            "d_model": d_model,
            "n_heads": n_heads,
            "num_transformer_layers": trial.suggest_int("num_transformer_layers", 1, 3),
            "fusion_method": trial.suggest_categorical("fusion_method", ["concat", "sum", "gated"]),
            "fusion_hidden_dim": trial.suggest_categorical("fusion_hidden_dim", [64, 128, 256]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.30),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        }
        val_params = validate_hpo_config("lstn", params)
        return LSTNForecaster(params=val_params)

    else:
        # forecasting_agent
        weights_method = trial.suggest_categorical("weights_method", ["uniform", "fixed", "inverse_mae", "inverse_rmse"])
        if weights_method == "fixed":
            w_patch = 0.40
            w_gpt = 0.40
            w_lstn = 0.20
        else:
            w_patch, w_gpt, w_lstn = 0.33, 0.33, 0.34

        params = {
            "selection_strategy": trial.suggest_categorical("selection_strategy", ["best_model", "weighted_ensemble", "dynamic_selection"]),
            "selection_metric": trial.suggest_categorical("selection_metric", ["mae", "rmse", "smape"]),
            "ensemble_size": trial.suggest_categorical("ensemble_size", [2, 3, 4]),
            "weights_method": weights_method,
            "consensus_threshold": trial.suggest_categorical("consensus_threshold", [0.70, 0.75, 0.80, 0.85, 0.90]),
            "fallback_strategy": trial.suggest_categorical("fallback_strategy", ["best_model", "seasonal_naive", "xgboost", "robust_scenarios"]),
            "max_debate_rounds": trial.suggest_categorical("max_debate_rounds", [1, 2, 3, 4]),
            "weight_patchtst": w_patch,
            "weight_gpt": w_gpt,
            "weight_lstn": w_lstn
        }
        val_params = validate_hpo_config("forecasting_agent", params)
        return ForecastingAgent()


class OptunaHPOEngine:
    """
    Orchestrateur Optuna HPO strict pour le module Forecasting.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}
        self.config = config
        self.hpo_cfg = config.get("hpo", {})
        self.n_trials = self.hpo_cfg.get("n_trials", 10)
        self.seed = self.hpo_cfg.get("seed", 42)
        self.objective_metric = self.hpo_cfg.get("objective_metric", "mae").lower()
        self.direction = self.hpo_cfg.get("direction", "minimize")

    def optimize_model(
        self,
        model_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: Optional[int] = None
    ) -> Tuple[Dict[str, Any], float]:
        """
        Exécute la recherche Optuna HPO uniquement sur Train et Validation.
        """
        if n_trials is None:
            n_trials = self.n_trials

        logger.info(f"Lancement Optuna HPO ({model_type}) : {n_trials} trials, seed={self.seed}, metric={self.objective_metric}...")

        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            sampler = optuna.samplers.TPESampler(seed=self.seed)
            pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=15, reduction_factor=3)
            study = optuna.create_study(direction=self.direction, sampler=sampler, pruner=pruner)

            def objective(trial: optuna.Trial):
                try:
                    model = build_model_from_trial(model_type, trial, self.config)
                    
                    if hasattr(model, "fit"):
                        model.fit(X_train, y_train)
                        
                    if hasattr(model, "predict"):
                        p_val = model.predict(X_val, horizon=len(y_val) if len(y_val) > 0 else 24)
                    else:
                        p_val = np.zeros_like(y_val)

                    # Métrique d'évaluation sur le jeu de validation
                    if hasattr(model, "evaluate"):
                        metrics = model.evaluate(y_val, p_val)
                    else:
                        mae = float(np.mean(np.abs(y_val - p_val))) if len(y_val) > 0 else 0.0
                        rmse = float(np.sqrt(np.mean((y_val - p_val)**2))) if len(y_val) > 0 else 0.0
                        metrics = {"MAE": mae, "RMSE": rmse}

                    metric_val = metrics.get("MAE" if self.objective_metric == "mae" else "RMSE", metrics.get("MAE", 0.0))
                    
                    # Signal de pruning pour Optuna
                    trial.report(metric_val, step=1)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

                    return metric_val

                except Exception as e:
                    if isinstance(e, optuna.TrialPruned):
                        raise e
                    logger.warning(f"Trial {trial.number} échoué : {e}")
                    raise optuna.TrialPruned()

            study.optimize(objective, n_trials=n_trials)
            
            best_params = study.best_params
            best_value = study.best_value
            
            # Valider et assainir les paramètres finaux
            sanitized_params = validate_hpo_config(model_type, best_params)
            
            logger.info(f"HPO terminé pour '{model_type}'. Meilleure métrique Val ({self.objective_metric}) = {best_value:.4f}")
            return sanitized_params, best_value

        except ImportError:
            logger.warning("Optuna non installé. Retour des paramètres par défaut.")
            return validate_hpo_config(model_type, {}), 0.0
