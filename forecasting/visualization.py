"""
===============================================================================
Module : Forecasting Visualizations & Plots Generator
File   : forecasting/visualization.py
Description : Génération et sauvegarde automatique des figures scientifiques
              (Courbes de prévision, comparaison multi-modèles, métriques par horizon,
              convergence HPO, score de consensus).
===============================================================================
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

logger = logging.getLogger("Visualization")


def plot_forecast_comparison(
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    timestamps: Optional[List[str]] = None,
    title: str = "Comparaison des Modèles de Forecasting vs Réel",
    output_path: str = "outputs/figures/forecast_comparison.png"
):
    """
    Génère et sauvegarde un graphique comparant les prédictions des différents modèles avec le réel.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        y_t = np.asarray(y_true).flatten()
        x_axis = np.arange(len(y_t)) if timestamps is None else timestamps[:len(y_t)]

        plt.figure(figsize=(12, 6))
        plt.plot(x_axis, y_t, label="Réel (Ground Truth)", color="#38BDF8", linewidth=2.5, linestyle="-")

        colors = ["#8B5CF6", "#10B981", "#F59E0B", "#EF4444", "#EC4899"]
        for idx, (m_name, y_p) in enumerate(predictions_dict.items()):
            y_p_flat = np.asarray(y_p).flatten()[:len(y_t)]
            c = colors[idx % len(colors)]
            plt.plot(x_axis, y_p_flat, label=f"Prévision {m_name}", color=c, linewidth=1.8, linestyle="--")

        plt.title(title, fontsize=12, fontweight="bold")
        plt.xlabel("Horizon Temporel (Heures)")
        plt.ylabel("Demande / Colis Prédits")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="upper right")
        plt.tight_layout()

        plt.savefig(output_path, dpi=150)
        plt.close()
        logger.info(f"Graphique comparatif sauvegardé : {output_path}")
    except Exception as e:
        logger.warning(f"Impossible de sauvegarder le graphique comparatif : {e}")


def plot_consensus_timeline(
    consensus_timeline: np.ndarray,
    overall_score: float,
    output_path: str = "outputs/figures/consensus_score.png"
):
    """
    Génère la courbe d'évolution du score de consensus inter-modèles sur l'horizon de prévision.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        plt.figure(figsize=(10, 4))
        plt.plot(consensus_timeline * 100.0, color="#10B981", linewidth=2, marker="o", label="Consensus Score (%)")
        plt.axhline(y=overall_score * 100.0, color="#F59E0B", linestyle="--", label=f"Moyenne Global ({overall_score * 100.0:.1f}%)")

        plt.title("Évolution du Score de Consensus Inter-Modèles", fontsize=11, fontweight="bold")
        plt.xlabel("Pas d'Horizon (Heures)")
        plt.ylabel("Consensus (%)")
        plt.ylim(0, 105)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="lower right")
        plt.tight_layout()

        plt.savefig(output_path, dpi=150)
        plt.close()
        logger.info(f"Graphique de consensus sauvegardé : {output_path}")
    except Exception as e:
        logger.warning(f"Impossible de sauvegarder le graphique de consensus : {e}")
