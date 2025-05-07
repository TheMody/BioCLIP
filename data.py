from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset



# -----------------------------------------------------------------------------
#  Collate → (wave_tensor, caption_str)
# -----------------------------------------------------------------------------

# def make_ecg_collate_fn():
#     def _collate(batch):
#         waves, label_lists = zip(*batch)  # list length B
#         caps = [", ".join(lbls) for lbls in label_lists]
#         return torch.stack(waves), caps
#     return _collate

def make_cifar_collate_fn(class_names):
    """Factory that creates a collate_fn mapping CIFAR labels → class‑name captions."""
    def _collate(batch):
        imgs, labels = zip(*batch)
        texts = [class_names[l] for l in labels]
        return torch.stack(imgs), list(texts)
    return _collate