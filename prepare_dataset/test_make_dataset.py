from typing import Dict

import pandas as pd
import numpy as np

from .make_dataset import make_train_val_test_split


def compute_stats(dataset: Dict[str, pd.DataFrame], column: str) -> np.array:
    """Calculate intersection beetwen splits based on column from DataFrame.

    Args:
        dataset (Dict[str, pd.DataFrame]): Dict of train/val/test split DataFrames
        column (str): Colunm in DataFrame to check intersection beetwen splits

    Returns:
        np.array: Square matrix 3x3 with intersection count beetwen splits.
    """
    stats = [
        [
            len(set(f1_val[f"{column}"]).intersection(f2_val[f"{column}"]))
            for f2_val in dataset.values()
        ]
        for f1_val in dataset.values()
    ]
    return np.array(stats)


def test_intersection_speaker():
    path_to_cv = "/mnt/980pro/datasets/commonvoice14/cv-corpus-14.0-2023-06-23/be/"
    split_to_include = ["validated", "invalidated", "other"]
    dataset = make_train_val_test_split(path_to_cv, split_to_include)
    stats = compute_stats(dataset, "client_id")
    assert np.all(np.array(stats) == np.diag(np.diagonal(np.array(stats))))


def test_intersection_sentence():
    path_to_cv = "/mnt/980pro/datasets/commonvoice14/cv-corpus-14.0-2023-06-23/be/"
    split_to_include = ["validated", "invalidated", "other"]
    dataset = make_train_val_test_split(path_to_cv, split_to_include)
    stats = compute_stats(dataset, "sentence")
    assert np.all(stats == np.diag(np.diagonal(stats)))
