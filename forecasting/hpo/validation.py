"""
===============================================================================
Module : Centralized HPO Validation & Consensus Score Engine
File   : forecasting/hpo/validation.py
Description : Validation centralisée des configurations d'hyperparamètres et
              calcul mathématique du score de consensus pour le pipeline.
===============================================================================
"""

import numpy as np
from typing import Dict, Any, List, Optional


def compute_consensus_score(predictions: List[np.ndarray], eps: float = 1e-8) -> float:
    """
    Calcule le score de consensus mathématique entre plusieurs prédictions.
    
    Formule :
      consensus_forecast(t) = mean(predictions_m(t))
      relative_disagreement(t) = std(predictions_m(t)) / (abs(consensus_forecast(t)) + eps)
      consensus_score(t) = 1 / (1 + relative_disagreement(t))
      
    Returns:
      float: Score de consensus moyen normalisé dans [0.0, 1.0].
    """
    if not predictions or len(predictions) == 0:
        return 0.0
        
    arr = np.array([np.asarray(p, dtype=float) for p in predictions])
    if arr.ndim != 2 or arr.shape[0] == 0:
        return 0.0

    mean_pred = np.mean(arr, axis=0)
    std_pred = np.std(arr, axis=0)
    
    relative_disagreement = std_pred / (np.abs(mean_pred) + eps)
    consensus_score_t = 1.0 / (1.0 + relative_disagreement)
    
    final_score = float(np.mean(consensus_score_t))
    return float(np.clip(final_score, 0.0, 1.0))


def validate_hpo_config(model_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Valide et assainit la configuration d'hyperparamètres pour un modèle donné.
    
    Args:
        model_name: "patchtst", "gpt_attention", "gpt_forecaster", "lstn", "forecasting_agent"
        params: Dictionnaire d'hyperparamètres à valider.
        
    Returns:
        Dict[str, Any]: Paramètres validés et nettoyés.
        
    Raises:
        ValueError: Si une contrainte critique n'est pas respectée.
    """
    p = dict(params)
    m = model_name.lower().strip()

    # -------------------------------------------------------------------------
    # 1. PATCHTST
    # -------------------------------------------------------------------------
    if "patchtst" in m:
        ctx_len = p.get("context_length", p.get("sequence_length", 24))
        p_len = p.get("patch_len", 8)
        
        if p_len > ctx_len:
            raise ValueError(f"PatchTST invalid: patch_len ({p_len}) > context_length ({ctx_len})")
        if p_len < 1:
            raise ValueError(f"PatchTST invalid: patch_len ({p_len}) must be >= 1")
            
        mode = p.get("patch_stride_mode", None)
        if mode == "half_patch":
            stride = max(1, p_len // 2)
        elif mode == "full_patch":
            stride = p_len
        else:
            stride = p.get("stride", max(1, p_len // 2))

        if stride <= 0:
            raise ValueError(f"PatchTST invalid: stride ({stride}) must be > 0")
        if stride > p_len:
            raise ValueError(f"PatchTST invalid: stride ({stride}) > patch_len ({p_len})")
            
        p["stride"] = stride
        p["context_length"] = ctx_len
        p["sequence_length"] = ctx_len

        d_model = p.get("d_model", 128)
        n_heads = p.get("n_heads", 8)
        if d_model % n_heads != 0:
            # Auto-ajustement propre pour éviter de bloquer
            d_model = ((d_model // n_heads) + 1) * n_heads
            p["d_model"] = d_model
            
        n_layers = p.get("n_layers", 3)
        if n_layers < 1:
            raise ValueError(f"PatchTST invalid: n_layers ({n_layers}) must be >= 1")
            
        pred_len = p.get("prediction_length", 24)
        if pred_len < 1:
            raise ValueError(f"PatchTST invalid: prediction_length ({pred_len}) must be >= 1")

    # -------------------------------------------------------------------------
    # 2. GPT FORECASTING (NUMÉRIQUE DIRECT)
    # -------------------------------------------------------------------------
    elif "gpt" in m:
        # Suppression explicite de temperature et top_p pour forecasting numérique direct (CAS A)
        if "temperature" in p:
            del p["temperature"]
        if "top_p" in p:
            del p["top_p"]

        ctx_len = p.get("context_length", p.get("sequence_length", 48))
        p_len = p.get("patch_len", 8)
        if p_len > ctx_len:
            raise ValueError(f"GPTForecaster invalid: patch_len ({p_len}) > context_length ({ctx_len})")

        mode = p.get("patch_stride_mode", None)
        if mode == "half_patch":
            stride = max(1, p_len // 2)
        elif mode == "full_patch":
            stride = p_len
        else:
            stride = p.get("stride", p.get("patch_stride", max(1, p_len // 2)))

        if stride <= 0 or stride > p_len:
            raise ValueError(f"GPTForecaster invalid: stride ({stride}) must be in (0, {p_len}]")

        p["stride"] = stride
        p["patch_stride"] = stride
        p["context_length"] = ctx_len
        p["sequence_length"] = ctx_len

        d_model = p.get("d_model", 128)
        n_heads = p.get("n_heads", 8)
        if d_model % n_heads != 0:
            d_model = ((d_model // n_heads) + 1) * n_heads
            p["d_model"] = d_model

        if p.get("freeze_backbone", False):
            p["trainable_layers"] = 0

        trainable_layers = p.get("trainable_layers", 2)
        if trainable_layers < 0:
            raise ValueError(f"GPTForecaster invalid: trainable_layers ({trainable_layers}) must be >= 0")

    # -------------------------------------------------------------------------
    # 3. LSTN (HYBRID LSTM-ATTENTION FORECASTER)
    # -------------------------------------------------------------------------
    elif "lstn" in m:
        ctx_len = p.get("context_length", p.get("sequence_length", 24))
        short_win = p.get("short_term_window", 12)
        if short_win > ctx_len:
            raise ValueError(f"LSTN invalid: short_term_window ({short_win}) > context_length ({ctx_len})")

        num_lstm = p.get("num_lstm_layers", p.get("rnn_num_layers", 2))
        if num_lstm < 1:
            raise ValueError(f"LSTN invalid: num_lstm_layers ({num_lstm}) must be >= 1")

        units = p.get("lstm_units", p.get("rnn_hidden_size", p.get("short_term_dim", 64)))
        if units < 1:
            raise ValueError(f"LSTN invalid: lstm_units ({units}) must be >= 1")

        drop = p.get("dropout", 0.10)
        rec_drop = p.get("recurrent_dropout", p.get("rnn_dropout", 0.10))
        if not (0.0 <= drop < 1.0) or not (0.0 <= rec_drop < 1.0):
            raise ValueError(f"LSTN invalid: dropout and recurrent_dropout must be in [0, 1)")

        if "n_heads" in p and "d_model" in p:
            d_m = p["d_model"]
            n_h = p["n_heads"]
            if d_m % n_h != 0:
                p["d_model"] = ((d_m // n_h) + 1) * n_h

        p["context_length"] = ctx_len
        p["sequence_length"] = ctx_len

    # -------------------------------------------------------------------------
    # 4. FORECASTING AGENT (RULE-BASED / ORCHESTRATOR)
    # -------------------------------------------------------------------------
    elif "agent" in m:
        # Supprimer learning_rate pour agent orchestrateur (CAS A)
        if "learning_rate" in p:
            del p["learning_rate"]

        thresh = float(p.get("consensus_threshold", 0.85))
        if not (0.0 < thresh <= 1.0):
            raise ValueError(f"ForecastingAgent invalid: consensus_threshold ({thresh}) must be in (0, 1]")

        ens_size = int(p.get("ensemble_size", 3))
        if ens_size < 1:
            raise ValueError(f"ForecastingAgent invalid: ensemble_size ({ens_size}) must be >= 1")

        weights_method = p.get("weights_method", "inverse_rmse")
        if weights_method == "fixed":
            w_patch = float(p.get("weight_patchtst", 0.40))
            w_gpt = float(p.get("weight_gpt", 0.40))
            w_lstn = float(p.get("weight_lstn", 0.20))
            tot_w = w_patch + w_gpt + w_lstn
            if abs(tot_w - 1.0) > 1e-6:
                # Normalisation propre si somme != 1
                if tot_w > 0:
                    p["weight_patchtst"] = round(w_patch / tot_w, 4)
                    p["weight_gpt"] = round(w_gpt / tot_w, 4)
                    p["weight_lstn"] = round(w_lstn / tot_w, 4)
                else:
                    raise ValueError("ForecastingAgent invalid: sum of fixed weights must be 1.0")

        rounds = int(p.get("max_debate_rounds", 3))
        if rounds < 1:
            raise ValueError(f"ForecastingAgent invalid: max_debate_rounds ({rounds}) must be >= 1")

    return p
