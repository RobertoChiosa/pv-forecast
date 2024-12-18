#  Copyright © Roberto Chiosa 2024.
#  Email: roberto.chiosa@polito.it
#  Last edited: 18/12/2024

# Standard library imports
import json
import os
from datetime import timedelta
from logging import getLogger

# Third party imports
import pandas as pd
import torch
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from skorch import NeuralNetRegressor
from torch.utils.data import DataLoader

# Project imports
from src.utils.LSTM import LSTM, LSTMSeriesDataset, test_lstm, train_lstm
from src.utils.MLP import MLP, MLPTimeseriesDataset, test_mlp, train_mlp
from src.utils.processing import Net
from utils.visualization import *

if __name__ == "__main__":

    # Set seed for reproducibility
    seed = 123
    np.random.seed(seed)

    # net algorithm
    net_type = "MLP"
    pv_name = "PV_Cittadella"
    grid_search = False

    # setup logging
    logger = getLogger(__name__)

    # Read configuration
    logger.info("Reading configuration")
    with open(os.path.join("config.json")) as f:
        config = json.load(f)

    # Initialize network
    net = Net(name=net_type, config=config[net_type])

    # 1. DATA PREPARATION (already processed from csv generation)

    # Load the data already processed
    df_raw = pd.read_csv(os.path.join("data", f"{pv_name}_preprocessed.csv"))
    df_raw["_time"] = pd.to_datetime(df_raw["_time"])
    df_raw.set_index("_time", inplace=True)
    fig_raw_line_plot = plot_raw(df_raw, title="Test")
    fig_raw_line_plot.savefig(
        os.path.join("out", "plot", f"{pv_name}_{net.name}_raw.png")
    )
    # get subset of the row and plot
    df_raw_subset = df_raw.loc["2023-09-20 00:00:00":"2023-10-05 23:59:59"]
    fig_raw_line_plot = plot_raw(df_raw_subset, title="Test")
    fig_raw_line_plot.savefig(
        os.path.join("out", "plot", f"{pv_name}_{net.name}_raw_subset.png")
    )

    # 2. DATA TRANSFORMATION

    # Ensure data is in the correct order by timestamp
    df_data = df_raw.copy()

    if net.name == "MLP":
        df_data = df_data.sort_index()
        df_data = df_data[df_data["power"] > 0]
        # get the min and max for each column
        df_min = df_data.min()
        df_min["power"] = 0
        df_min["rad"] = 0
        df_min["temp"] = -10
        df_min["zenith"] = 0
        df_min["azimuth"] = 0
        df_min["ghi"] = 0

        # Create a new row with the minimum values
        new_min_row = pd.DataFrame(df_min).transpose()
        new_min_row.index = [df_data.index[-1] + timedelta(minutes=15)]

        df_max = df_data.max()
        # df_max["power"] = 0
        # df_max["rad"] = 0
        df_max["temp"] = 45
        df_max["zenith"] = 180
        df_max["azimuth"] = 360
        df_max["ghi"] = 1000

        # Create a new row with the minimum values
        new_max_row = pd.DataFrame(df_max).transpose()
        new_max_row.index = [df_data.index[-1] + timedelta(minutes=15)]

        # Append the new row to the DataFrame
        df_data = pd.concat([df_data, new_min_row, new_max_row])
        # Scale the data
        scaler = MinMaxScaler()
        df_normalized = scaler.fit_transform(df_data)

        # remove last 2 rows containing fake min max
        df_normalized = df_normalized[:-2]
    elif net.name == "LSTM":
        # Scale the data
        scaler = MinMaxScaler()
        df_normalized = scaler.fit_transform(df_data)
    else:
        raise ValueError("Invalid network type")

    fig_normalized_line_plot = plot_raw(pd.DataFrame(df_normalized), title="Test")
    fig_normalized_line_plot.savefig(
        os.path.join("out", "plot", f"{pv_name}_{net.name}_normalized.png")
    )
    # get subset of the row and plot
    df_normalized_subset = pd.DataFrame(df_normalized)[100:1000]
    df_normalized_subset.rename(columns={
        0: "Electrical Power",
        1: "Solar Radiation",
        2: "Air Temperature",
    }, inplace=True)
    df_normalized_subset_plot = df_normalized_subset.copy()[["Electrical Power", "Solar Radiation", "Air Temperature"]]

    fig_normalized_line_plot = plot_raw(
        data_df=df_normalized_subset_plot,
        title="Normalized input variables",
    )

    fig_normalized_line_plot.savefig(
        os.path.join("out", "plot", f"{pv_name}_{net.name}_normalized_subset.png")
    )

    # Separate features and target variable
    # the first is power so is y the others are x
    y = df_normalized[:, 0]
    x = df_normalized[:, 1:]

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, train_size=0.7, random_state=seed
    )

    # Create the dataset and dataloader for training and testing
    if net.name == "MLP":
        train_dataset = MLPTimeseriesDataset(x_train, y_train)
        test_dataset = MLPTimeseriesDataset(x_test, y_test)

    elif net.name == "LSTM":
        train_dataset = LSTMSeriesDataset(x_train, y_train)
        test_dataset = LSTMSeriesDataset(x_test, y_test)
    else:
        raise ValueError("Invalid network type")

    # Create DataLoaders with batch size (you can adjust batch_size as needed)
    train_loader = DataLoader(train_dataset, batch_size=net.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=net.batch_size, shuffle=False)

    # 3. MODEL INITIALIZATION
    if net.name == "MLP":
        model = MLP(
            input_size=train_dataset.x.shape[1],
            hidden_size=net.hidden_size,
            output_size=net.output_size,
            num_layers=net.num_layers,
            dropout_p=net.dropout_p,
        )

    elif net.name == "LSTM":
        model = LSTM(
            input_size=train_dataset.x.shape[1],
            hidden_size=net.hidden_size,
            output_size=net.output_size,
            num_layers=net.num_layers,
            dropout_p=net.dropout_p,
        )
    else:
        raise ValueError("Invalid network type")

    # HYPERPARAMETERS GRID SEARCH
    # define the grid search parameters
    # create the skorch wrapper
    if grid_search:
        model_grid = NeuralNetRegressor(
            module=model,
            criterion=torch.nn.MSELoss,
            train_split=None,  # GridSearchCV will handle the splits
            verbose=0,

        )

        param_grid = {
            'max_epochs': [10, 50, 100],
            'lr': [0.001, 0.01, 0.1],
            "module__input_size": [train_dataset.x.shape[1]],
            "module__output_size": [net.output_size],
            "module__num_layers": [net.num_layers],
            "module__dropout_p": [net.dropout_p],
            'module__hidden_size': [16, 32, 64, 128, 256],
        }
        grid = GridSearchCV(estimator=model_grid, param_grid=param_grid, n_jobs=-1, cv=3,
                            scoring='neg_mean_squared_error')
        grid_result = grid.fit(x_train.astype(np.float32), y_train.astype(np.float32))

        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))

    # 4. TRAINING
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=net.learning_rate)
    if net.name == "MLP":
        loss_train, actual_values_train, predicted_values_train = train_mlp(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            data_loader=train_loader,
            epochs=net.epochs,
        )

        loss_test, actual_values_test, predicted_values_test = test_mlp(
            model=model,
            criterion=criterion,
            data_loader=test_loader,
        )

    elif net.name == "LSTM":
        loss_train, actual_values_train, predicted_values_train = train_lstm(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            data_loader=train_loader,
            epochs=net.epochs,
        )

        loss_test, actual_values_test, predicted_values_test = test_lstm(
            model=model,
            criterion=criterion,
            data_loader=test_loader,
        )

    else:
        raise ValueError("Invalid network type")

    # Plot
    fig_loss = plot_loss(loss_train)
    fig_loss.savefig(os.path.join("out", "plot", f"{pv_name}_{net.name}_loss.png"))

    fig_error_dist = error_distribution(
        y_real=actual_values_test, y_pred=predicted_values_test
    )
    fig_error_dist.savefig(
        os.path.join("out", "plot", f"{pv_name}_{net.name}_error_test.png")
    )

    fig_scatter = plot_scatter(y_real=actual_values_test, y_pred=predicted_values_test)
    fig_scatter.savefig(
        os.path.join("out", "plot", f"{pv_name}_{net.name}_scatter_test.png")
    )
    
    # TEST
