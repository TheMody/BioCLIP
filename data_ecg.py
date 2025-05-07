# ptbxl_lazy_dataset.py
import ast
from pathlib import Path
from typing import List, Tuple, Dict, Any

import pandas as pd
import torch
from torch.utils.data import Dataset
import wfdb
import numpy as np


class PTBXLWaveformDataset(Dataset):
    """
    Lazy‑loading PTB‑XL dataset.

    Each __getitem__ reads exactly one WFDB record (≈ 150 kB on disk for 12×10 s @100 Hz)
    instead of pre‑loading the whole archive into RAM.

    Returns
    -------
    wave : torch.FloatTensor               # shape (12, L)      – raw ECG signal
    labels : List[str]                     # diagnostic superclass(es)
    record_id : int                        # ecg_id for bookkeeping
    """
    def __init__(
        self,
        root: str | Path,
        split: str = "train",              # "train", "test", or "all"
        sampling_rate: int = 100,          # 100 Hz (“_lr”) or 500 Hz (“_hr”)
        max_len: int = 1000,               # crop / pad length (1000 = 10 s @100 Hz)
    ):
        super().__init__()
        self.root = Path(root)
        self.sr = sampling_rate
        self.max_len = max_len

     #   self.labelcodes_to_text = {}
    #    pd.read_csv(self.root / "scp_statements.csv")

        # -------------------------
        # Metadata
        # -------------------------
        meta = pd.read_csv(self.root / "ptbxl_database.csv", index_col="ecg_id")
     #   meta.scp_codes = meta.scp_codes.apply(ast.literal_eval)

        # # diagnostic superclass map (NORM, AFIB, …)
        # agg_df = (
        #     pd.read_csv(self.root / "scp_statements.csv", index_col=0)
        #     .query("diagnostic == 1")
        # )
        # self._key2superclass = {
        #     k: v for k, v in zip(agg_df.index, agg_df.diagnostic_class)
        # }

        # def _aggregate(y_dic: Dict[str, int]) -> List[str]:
        #     return list(
        #         {self._key2superclass[k] for k in y_dic.keys() if k in self._key2superclass}
        #     )

        # meta["diagnostic_superclass"] = meta.scp_codes.apply(_aggregate)

        # split on official strat_fold column your original script used
        if split == "train":
            meta = meta[meta.strat_fold != 10]
        elif split == "test":
            meta = meta[meta.strat_fold == 10]
        elif split != "all":
            raise ValueError("split must be 'train', 'test', or 'all'")

        self.meta = meta.reset_index()  # keep ecg_id as a column

        # choose filename column
        self.fn_col = "filename_lr" if sampling_rate == 100 else "filename_hr"

    # ---------------------------------------------------------------------
    # PyTorch mandatory API
    # ---------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.meta)

    def _load_waveform(self, record_path: Path) -> torch.Tensor:
        """Read one WFDB record and return (12, max_len) float32 tensor."""
        sig, _ = wfdb.rdsamp(str(record_path))
        sig = sig.T.astype(np.float32)  # (12, n_samples)

        # crop / pad centrally
        L = sig.shape[1]
        if L > self.max_len:
            start = (L - self.max_len) // 2
            sig = sig[:, start : start + self.max_len]
        elif L < self.max_len:
            pad = self.max_len - L
            sig = np.pad(sig, ((0, 0), (0, pad)))

        return torch.from_numpy(sig)


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        row = self.meta.iloc[idx]
        path = self.root / row[self.fn_col]
        wave = self._load_waveform(path)
        #print(row.report)
        labels =row.report# row.diagnostic_superclass or ["UNK"]
        return wave, str(labels)#, row.ecg_id