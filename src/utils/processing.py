#  Copyright © Roberto Chiosa 2024.
#  Email: roberto.chiosa@polito.it
#  Last edited: 21/10/2024

# Third party imports
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def data_preparation_gim(filename: str) -> pd.DataFrame:
    """

    :return:
    """
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
    data_df = pd.read_csv(filename)
    # coerce avoiding read errors
    data_df["ElectricPower"] = data_df["ElectricPower"].apply(pd.to_numeric, errors='coerce', downcast='float')

    # Convert the 'Timestamp' column to datetime
    data_df["Timestamp"] = pd.to_datetime(data_df["Timestamp"])

    # Create a column with the hour of the day
    data_df["hour"] = data_df["Timestamp"].dt.hour

    # transform the hour column in a sin and cos variable
    data_df["sin_hour"] = np.sin(2 * np.pi * data_df["hour"] / 24)
    data_df["cos_hour"] = np.cos(2 * np.pi * data_df["hour"] / 24)

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
    # drop column where electric power <0
    data_df = data_df[data_df["ElectricPower"] >= 0]
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
