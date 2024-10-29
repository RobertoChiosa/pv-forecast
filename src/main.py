#  Copyright © Roberto Chiosa 2024.
#  Email: roberto.chiosa@polito.it
#  Last edited: 29/10/2024

# Standard library imports
import json
import logging
import os
from logging import getLogger

# Third party imports
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

# Project imports
from networks import LSTM, MLP, Net, train_mlp, train_lstm, MLPTimeseriesDataset, LSTMSeriesDataset
from utils.visualization import *

if __name__ == "__main__":

    seed = 123
    np.random.seed(seed)

    # net algorithm
    net_type = "MLP"
    pv_name = "PV_Aule_P"

    # setup logging
    logger = getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s.%(funcName)s:%(lineno)d - %(message)s",
    )

    # Read configuration
    logger.info("Reading configuration")
    with open(os.path.join("utils", "config.json")) as f:
        config = json.load(f)

    net = Net(name=net_type, config=config[net_type])

    # 1. DATA PREPARATION (already processed from csv generation)
    data_df = pd.read_csv(
        os.path.join("data", f"{pv_name}_preprocessed.csv")
    )  # already processed
    data_df["_time"] = pd.to_datetime(data_df["_time"])
    data_df.set_index("_time", inplace=True)
    fig_raw_line_plot = plot_raw(data_df, title="Test")
    fig_raw_line_plot.savefig(os.path.join("out", f"{pv_name}_{net.name}_raw.png"))

    # 2. DATA TRANSFORMATION

    # Ensure data is in the correct order by timestamp
    data_df = data_df.sort_index()
    # Scale the data
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data_df)
    # Separate features and target variable
    # the first is power so is y the others are x
    y = data_normalized[:, 0]
    x = data_normalized[:, 1:]

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

    # 4. TRAINING
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=net.learning_rate)
    if net.name == "MLP":
        train_mlp(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            data_loader=train_loader,
            epochs=net.epochs,
        )
    elif net.name == "LSTM":
        train_lstm(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            data_loader=train_loader,
            epochs=net.epochs,
        )
    else:
        raise ValueError("Invalid network type")

    # 5. TESTING
    model.eval()
    if net.name == "LSTM":
        model.init_hidden(net.batch_size)

    with torch.no_grad():
        test_predictions = []
        test_actual = []
        for batch in test_loader:
            input_test, target_test = batch
            if net.name == "LSTM":
                input_test = input_test.unsqueeze(1)
                output, h = model(input_test, h)
                optimizer.zero_grad()
                loss_test = criterion(output, target_test)
            else:
                output = model(input_test)

            test_predictions.append(output.numpy())
            test_actual.append(target_test.numpy())

        test_predictions = np.concatenate(test_predictions, axis=0)
        test_actual = np.concatenate(test_actual, axis=0)

        # Rescale the predictions and actual
        test_predictions = scaler.inverse_transform(
            np.concatenate(
                (test_x[: len(test_predictions)], test_predictions.reshape(-1, 1)),
                axis=1,
            )
        )[:, -1]
        test_actual = scaler.inverse_transform(
            np.concatenate(
                (test_x[: len(test_actual)], test_actual.reshape(-1, 1)), axis=1
            )
        )[:, -1]

        # Calculate performance metrics
        rmse_test = np.sqrt(np.mean((test_predictions - test_actual) ** 2))
        r2_test = 1 - np.sum((test_actual - test_predictions) ** 2) / np.sum(
            (test_actual - np.mean(test_actual)) ** 2
        )

        try:
            mape_test = (
                    np.mean(np.abs((test_actual - test_predictions) / test_actual)) * 100
            )
        except ZeroDivisionError:
            logger.warning("Actual values contain zero values, fixing MAPE calculation")
            mape_test = (
                    np.mean(
                        np.abs((test_actual - test_predictions) / (test_actual + 1e-10))
                    )
                    * 100
            )

        logger.info(
            f"RMSE_test: {rmse_test:.4f}, MAPE_test: {mape_test:.4f}, R2_test: {r2_test:.4f}"
        )

    # Plot the prediction and actual
    fig_line_plot = plot_graph(
        y_pred=test_predictions, y_real=test_actual, title="Test"
    )
    fig_error_dist = error_distribution(y_pred=test_predictions, y_real=test_actual)
    fig_scatter = plot_scatter(y_pred=test_predictions, y_real=test_actual)

    fig_line_plot.savefig(os.path.join("out", f"{pv_name}_{net.name}_line_plot.png"))
    fig_error_dist.savefig(os.path.join("out", f"{pv_name}_{net.name}_error.png"))
    fig_scatter.savefig(os.path.join("out", f"{pv_name}_{net.name}_scatter.png"))

    # create a dataframe with the predictions and the actual
    df = pd.DataFrame(
        {"Predictions": test_predictions.flatten(), "Actual": test_actual.flatten()}
    )

    # save the dataframe in a csv file
