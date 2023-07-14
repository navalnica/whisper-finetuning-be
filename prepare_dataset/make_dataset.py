import os
import argparse
from argparse import Namespace
from typing import List, Dict, Union
from pathlib import Path

import pandas as pd
from datasets import DatasetDict, Dataset, Audio


def make_train_val_test_split(
    path_to_cv: Union[str, Path], split_to_include: List[str]
) -> Dict[str, pd.DataFrame]:
    """Creating a custom train/val/text split based
    on the original splits from CommonVoice.

    The base split from CommonVoice are:
    'train' -> 'train', 'dev' -> 'validation', 'test' -> 'test'.
    Also CommonVoice have 'validated', 'unvalidated', 'other'
    split, and this split can have the same speaker(client_id)
    from base split('train', 'dev', 'test'). Also this split
    can have the same sentences.
    Therefore, need to filter out additional splits to 'test', 'dev'.

    For example 'validated' can contain speaker 1 from 'test',
    and speaker 2 from 'dev', so need to add data
    with this speaker to 'test' and 'dev' split accordingly,
    other data with different speakers go to train.
    After this preproccesing need to remove audio with
    validation and test sentence from train split, and remove
    audio with dev sentence from test split.

    Args:
        path_to_cv (str): Path to folder which contain CommanVoice dataset
        split_to_include (List[str]): Split from CommonVoice dataset
        for spliting into train/val/test and concatenate with base split.

    Returns:
        Dict[str, pd.DataFrame]: Dict of train/val/test split DataFrames
    """
    # Load base Dataframe from CommonVoice
    base_train_df = pd.read_csv(os.path.join(Path(path_to_cv), "train.tsv"), sep="\t")
    base_val_df = pd.read_csv(os.path.join(Path(path_to_cv), "dev.tsv"), sep="\t")
    base_test_df = pd.read_csv(os.path.join(Path(path_to_cv), "test.tsv"), sep="\t")

    # Get list of unique speaker
    val_speaker_list = base_val_df["client_id"].unique()
    test_speaker_list = base_test_df["client_id"].unique()

    # Load other Dataframe of split to include in final dataset
    splits = {
        split: pd.read_csv(os.path.join(Path(path_to_cv), f"{split}.tsv"), sep="\t")
        for split in split_to_include
    }

    # Make list of dataframes with speaker from base Dataframes
    split_for_train = [
        df[
            ~df["client_id"].isin(val_speaker_list)
            & ~df["client_id"].isin(test_speaker_list)
        ]
        for df in splits.values()
    ]
    split_for_val = [
        df[df["client_id"].isin(val_speaker_list)] for df in splits.values()
    ]
    split_for_test = [
        df[df["client_id"].isin(test_speaker_list)] for df in splits.values()
    ]

    # Concatenate all DataFrame for each split in one
    train_final_df = pd.concat(split_for_train + [base_train_df])
    val_final_df = pd.concat(split_for_val + [base_val_df])
    test_final_df = pd.concat(split_for_test + [base_test_df])

    # Reset the index of the final dataframes
    train_final_df = train_final_df.reset_index(drop=True)
    val_final_df = val_final_df.reset_index(drop=True)
    test_final_df = test_final_df.reset_index(drop=True)

    # Get list of unique sentence from dev and test
    val_sentence_list = val_final_df["sentence"].unique()
    test_sentence_list = test_final_df["sentence"].unique()

    # Delete dev and test sentence from train
    train_final_df = train_final_df[
        ~train_final_df["sentence"].isin(val_sentence_list)
        & ~train_final_df["sentence"].isin(test_sentence_list)
    ]

    # Delete dev sentence from test
    val_final_df = val_final_df[~val_final_df["sentence"].isin(test_sentence_list)]

    # Make dict of splits
    final_dict_df = {
        "train": train_final_df,
        "validation": val_final_df,
        "test": test_final_df,
    }
    return final_dict_df


def process_raw_data(
    path_to_cv: Union[str, Path],
    dict_df: Dict[str, pd.DataFrame],
    sampling_rate: int = 16000,
) -> DatasetDict:
    """Create a HuggingFace Dataset from raw train/val/test split dict

    Args:
        path_to_cv (str): Path to folder which contain CommanVoice dataset
        dict_df (Dict[str, pd.DataFrame]): Dict of train/val/test split DataFrames
        sampling_rate (int, optional): Sampling rate for read audio with hfd.Audio.
        Defaults to 16000.

    Returns:
        DatasetDict: Processed HuggingFace DatasetDict
    """

    # Add full path to new column 'audio'
    def add_column(df: pd.DataFrame) -> pd.DataFrame:
        df["audio"] = os.path.join(Path(path_to_cv), "clips", str(df["path"]))
        return df

    dict_df = {split: add_column(df) for split, df in dict_df.items()}

    # Make a hf.DatasetDict from dict of DataFrame
    hf_dataset = DatasetDict(
        {
            split: Dataset.from_pandas(df[["audio", "sentence"]].reset_index(drop=True))
            for split, df in dict_df
        }
    )

    # Process audio column to hf.Audio feature
    hf_dataset = hf_dataset.cast_column(
        column="audio", feature=Audio(sampling_rate=sampling_rate)
    )
    return hf_dataset


def pipline(run_opts: Namespace):
    """Pipline for process raw data and save dataset

    Args:
        run_opts (Namespace): Options from command line
    """
    # Make train/val/test split for dataset
    data_df = make_train_val_test_split(
        path_to_cv=run_opts["path_to_cv"], split_to_include=run_opts["split_to_include"]
    )
    # Process dataset
    hf_dataset = process_raw_data(
        path_to_cv=run_opts["path_to_cv"], dict_df=data_df, sampling_rate=run_opts["sr"]
    )
    # Save to disk
    hf_dataset.save_to_disk(run_opts["output_path"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a CommonVoice dataset for ASR")
    parser.add_argument(
        "--path_to_cv", type=str, help="Directory which contain CommonVoice Dataset"
    )

    parser.add_argument(
        "--split_to_include",
        default="validated",
        nargs="+",
        help="Splits of CommonVoice dataset to include in new dataset"
        "Supported value {validated, invalidated, other}",
    )

    parser.add_argument(
        "--sr",
        type=int,
        default=16000,
        help="Sampling rate to create featutre audio, to create a dataset",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="hf_dataset",
        help="Directory for saving HuggingFace DatasetDict",
    )

    args = parser.parse_args()
    pipline(args)
