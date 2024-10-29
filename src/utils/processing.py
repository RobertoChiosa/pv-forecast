#  Copyright © Roberto Chiosa 2024.
#  Email: roberto.chiosa@polito.it
#  Last edited: 29/10/2024
# Standard library imports
from logging import getLogger

# Third party imports
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

logger = getLogger(__name__)


class Net:
    """
    Model class
    """

    def __init__(self, name: str, config: dict):
        self.name = name
        self.hidden_size = config["hidden_size"]
        self.output_size = config["output_size"]
        self.lookback = config["lookback"]
        self.num_layers = config["num_layers"]
        self.dropout_p = config["dropout_p"]
        self.learning_rate = config["learning_rate"]
        self.optimizer = config["optimizer"]
        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]


def dataset_dataloader(x, y, BATCH_SIZE, shuffle=True):
    """
    Data loader
    :param x:
    :param y:
    :param BATCH_SIZE:
    :param shuffle:
    :return:
    """
    TENSOR = TensorDataset(
        torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32))
    )
    LOADER = DataLoader(TENSOR, shuffle=shuffle, batch_size=BATCH_SIZE, drop_last=True)
    return TENSOR, LOADER


def data_train_test_split(df: pd.DataFrame) -> tuple:
    """
    Split the dataset in training and test set
    Subset the dataset in 4 portions. For each portion select the first 80% of the data as training set and the
    remaining 20% as test set Then merge all in two datasets: train and test
    :param df: the dataset
    :return:  train and test dataset as nunpy arrays
    """
    logger.info(f"Creating train and test dataset")
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()

    portions = 1
    # Calculate the size of each portion
    portion_size = len(df) // portions

    # Iterate over the 4 portions
    for i in range(portions):
        # Calculate the start and end indices for the portion
        start_index = i * portion_size
        end_index = start_index + portion_size

        # Subset the portion from the dataset
        portion = df.iloc[start_index:end_index]

        # Split the portion into training and test data
        train_end_index = int(0.7 * len(portion))
        train_portion = portion.iloc[:train_end_index]
        test_portion = portion.iloc[train_end_index:]

        # Append the training and test data to df_train and df_test
        df_train = pd.concat([df_train, train_portion])
        df_test = pd.concat([df_test, test_portion])

    # Convert the dataframes in numpy arrays
    df_train = df_train.to_numpy().astype(np.float32)
    df_test = df_test.to_numpy().astype(np.float32)

    return df_train, df_test
