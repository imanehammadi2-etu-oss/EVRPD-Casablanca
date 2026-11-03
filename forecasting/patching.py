"""
===============================================================================
Module : Temporal Patching for Transformer/GPT Models
File   : forecasting/patching.py
Description : Découpage des fenêtres de séries temporelles en patchs 3D.
              Réservé exclusivement aux architectures Transformer/GPT (PatchTST, CustomGPT).
===============================================================================
"""

import math
import logging
import numpy as np
from typing import Tuple, Optional

logger = logging.getLogger("Patching")


def create_patches(
    X: np.ndarray,
    patch_length: int = 12,
    patch_stride: int = 6,
    padding_mode: Optional[str] = None
) -> Tuple[np.ndarray, int]:
    """
    Découpe une fenêtre ou un lot de fenêtres [n_samples, context_length, n_features] en patchs 3D/4D.
    
    Formule mathématique :
    n_patches = floor((context_length - patch_length) / patch_stride) + 1
    
    Arguments :
    ------------
    X : np.ndarray
        Forme (n_samples, context_length, n_features) ou (context_length, n_features)
    patch_length : int
        Taille du patch temporel (longueur de sous-série)
    patch_stride : int
        Pas de décalage entre deux patchs successifs
    padding_mode : Optional[str]
        Mode de rembourrage ('edge', 'zeros', None)
        
    Renvoie :
    ---------
    patches : np.ndarray
        Forme (n_samples, n_patches, patch_length, n_features) ou (n_patches, patch_length, n_features)
    n_patches : int
    """
    orig_ndim = X.ndim
    if orig_ndim == 2:
        X_batch = np.expand_dims(X, axis=0)
    else:
        X_batch = X

    n_samples, context_length, n_features = X_batch.shape

    # Contrôles de validité
    if patch_stride <= 0:
        raise ValueError(f"Le pas de patch (patch_stride={patch_stride}) doit être strictement positif (> 0).")
    
    if patch_length > context_length:
        if padding_mode:
            pad_len = patch_length - context_length
            X_batch = np.pad(X_batch, ((0, 0), (0, pad_len), (0, 0)), mode=padding_mode if padding_mode != 'zeros' else 'constant')
            context_length = patch_length
        else:
            raise ValueError(f"patch_length ({patch_length}) dépasse context_length ({context_length}) !")

    # Formule exacte du nombre de patchs
    n_patches = math.floor((context_length - patch_length) / patch_stride) + 1

    patches_list = []
    for i in range(n_patches):
        start_idx = i * patch_stride
        end_idx = start_idx + patch_length
        patch_slice = X_batch[:, start_idx:end_idx, :]
        patches_list.append(patch_slice)

    # Forme : (n_patches, n_samples, patch_length, n_features) -> transposer en (n_samples, n_patches, patch_length, n_features)
    patches_arr = np.stack(patches_list, axis=1)

    if orig_ndim == 2:
        patches_arr = np.squeeze(patches_arr, axis=0)

    logger.info(f"Patching appliqué : {n_patches} patchs générés (patch_len={patch_length}, stride={patch_stride}). Shape final = {patches_arr.shape}")
    return patches_arr, n_patches
