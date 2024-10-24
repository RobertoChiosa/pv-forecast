#  Copyright © Roberto Chiosa 2024.
#  Email: roberto.chiosa@polito.it
#  Last edited: 24/10/2024
from logging import getLogger

# Third party imports
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

logger = getLogger(__name__)


def data_preparation_gim(filename: str) -> pd.DataFrame:
    """

    :return:
    """
    logger.info(f"Reading data from {filename}")
    data_df = pd.read_csv(filename)

    col_names = [
        "Timestamp",
        "Tae (degC (Ave))",
        "Cav_Ta_outlet (degC (Ave))",
        "HEX1_Ts_mid_up (degC (Ave))",
        "HEX1_v_outlet (m/s)",
        "Pyra_Out_Ver (mV (Ave))",
        "HEX1_Ta_outlet (degC (Ave))",
    ]

    # from data_df extract the columns that are in col_names
    data_df = data_df[col_names]

    # Convert the 'Timestamp' column to datetime
    data_df["Timestamp"] = pd.to_datetime(data_df["Timestamp"])

    # Create a column with the hour of the day
    data_df["hour"] = data_df["Timestamp"].dt.hour

    # transform the hour column in a sin and cos variable
    data_df["sin_hour"] = np.sin(2 * np.pi * data_df["hour"] / 24)
    data_df["cos_hour"] = np.cos(2 * np.pi * data_df["hour"] / 24)

    # put the HEX1_Ta_outlet (degC (Ave)) column as the last column
    data_df = data_df[
        [col for col in data_df.columns if col != "HEX1_Ta_outlet (degC (Ave))"]
        + ["HEX1_Ta_outlet (degC (Ave))"]
        ]

    # Granularity Check
    # Calculate the difference between consecutive timestamps
    time_diff = data_df["Timestamp"].diff()

    # remove all the rows where the difference is not 15 minutes
    data_df = data_df[time_diff == pd.Timedelta("0 days 00:15:00")]

    return data_df


def data_preparation_pv(filename: str) -> pd.DataFrame:
    """

    :return:
    """
    logger.info(f"Reading data from {filename}")
    data_df = pd.read_csv(filename)
    # coerce avoiding read errors
    data_df["ElectricPower"] = data_df["ElectricPower"].apply(pd.to_numeric, errors='coerce', downcast='float')

    # Convert the 'Timestamp' column to datetime
    data_df["Timestamp"] = pd.to_datetime(data_df["Timestamp"])

    # tod add interpolation and resample if necessary

    # Granularity Check
    # Calculate the difference between consecutive timestamps
    time_diff = data_df["Timestamp"].diff()

    # put the y to be predicted as column as the last column
    data_df = data_df[
        [col for col in data_df.columns if col != "ElectricPower"]
        + ["ElectricPower"]
        ]

    # remove all the rows where the difference is not 15 minutes
    data_df = data_df[time_diff == pd.Timedelta("0 days 00:15:00")]
    # remove all the rows where the ElectricPower is NaN
    data_df = data_df.dropna()
    return data_df


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

    # Calculate the size of each portion
    portion_size = len(df) // 4

    # Iterate over the 4 portions
    for i in range(4):
        # Calculate the start and end indices for the portion
        start_index = i * portion_size
        end_index = start_index + portion_size

        # Subset the portion from the dataset
        portion = df.iloc[start_index:end_index]

        # Split the portion into training and test data
        train_end_index = int(0.8 * len(portion))
        train_portion = portion.iloc[:train_end_index]
        test_portion = portion.iloc[train_end_index:]

        # Append the training and test data to df_train and df_test
        df_train = pd.concat([df_train, train_portion])
        df_test = pd.concat([df_test, test_portion])

    # df_train = data_df.iloc[round(len(data_df)*0.8):]
    # df_test = data_df.iloc[:round(len(data_df)*0.8)]

    # Convert the dataframes in numpy arrays
    df_train = df_train.to_numpy().astype(np.float32)
    df_test = df_test.to_numpy().astype(np.float32)

    return df_train, df_test
