#  Copyright © Roberto Chiosa 2024.
#  Email: roberto.chiosa@polito.it
#  Last edited: 20/10/2024

# Standard library imports
import json
import logging
import os
from logging import getLogger

# Third party imports
import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.preprocessing import MinMaxScaler

# Project imports
from networks import LSTM, MLP
from utils.processing import dataset_dataloader, data_preparation
from utils.visualization import *

net = "MLP"

if __name__ == "__main__":

    # setup logging
    logger = getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Read configuration
    logger.info("Reading configuration")
    with open(os.path.join("utils", "config.json")) as f:
        config = json.load(f)

    if config["wandb"]["on"]:
        logger.info("Wandb is on")
        wandb.init(
            project=config["wandb"]["project_name"],
            entity=config["wandb"]["entity"],
            name=net + config["wandb"]["run_name"],
            config=config,
        )
    else:
        logger.info("Wandb is off")

    # 1. DATA PREPARATION
    data_df = data_preparation(filename=os.path.join("data", "dataset_final.csv"))

    # 2. DATA TRANSFORMATION
    # drop the Timestamp column
    data_df = data_df.drop(columns=["Timestamp"])

    # Subset the dataset in 4 portions. For each portion select the first 80% of the data as training set and the
    # remaining 20% as test set Then merge all in two datasets: train and test
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()

    # Calculate the size of each portion
    portion_size = len(data_df) // 4

    # Iterate over the 4 portions
    for i in range(4):
        # Calculate the start and end indices for the portion
        start_index = i * portion_size
        end_index = start_index + portion_size

        # Subset the portion from the dataset
        portion = data_df.iloc[start_index:end_index]

        # Split the portion into training and test data
        train_end_index = int(0.8 * len(portion))
        train_portion = portion.iloc[:train_end_index]
        test_portion = portion.iloc[train_end_index:]

        # Append the training and test data to train_df and test_df
        train_df = pd.concat([train_df, train_portion])
        test_df = pd.concat([test_df, test_portion])

    # train_df = data_df.iloc[round(len(data_df)*0.8):]
    # test_df = data_df.iloc[:round(len(data_df)*0.8)]

    # Convert the dataframes in numpy arrays
    train_df = train_df.to_numpy().astype(np.float32)
    test_df = test_df.to_numpy().astype(np.float32)

    # Normalize the data min max scaling
    scaler = MinMaxScaler()
    train_df = scaler.fit_transform(train_df)
    test_df = scaler.transform(test_df)

    # Split the data in input and output
    train_x, train_y = train_df[:, :-1], train_df[:, -1]
    test_x, test_y = test_df[:, :-1], test_df[:, -1]

    input_size = train_x.shape[-1]

    # Dataset and dataloader
    train_tensor, train_loader = dataset_dataloader(
        train_x, train_y, config[net]["batch_size"], shuffle=True
    )
    test_tensor, test_loader = dataset_dataloader(
        test_x, test_y, config[net]["batch_size"], shuffle=False
    )

    # 3. MODEL INITIALIZATION
    if net == "MLP":
        model = MLP(
            input_size=input_size,
            hidden_size=config[net]["hidden_size"],
            output_size=config[net]["output_size"],
            num_layers=config[net]["num_layers"],
            dropout_p=config[net]["dropout_p"],
        )

    elif net == "LSTM":
        model = LSTM(
            input_size=input_size,
            hidden_size=config[net]["hidden_size"],
            output_size=config[net]["output_size"],
            num_layers=config[net]["num_layers"],
            dropout_p=config[net]["dropout_p"],
        )

    # Initialize the optimizer and loss function criterion
    criterion = torch.nn.MSELoss()
    optimizer = getattr(torch.optim, config[net]["optimizer"])(
        model.parameters(), lr=config[net]["learning_rate"]
    )

    # 4. TRAINING
    loss_train = []

    for epoch in range(config[net]["epochs"]):
        model.train()
        if net == "LSTM":
            h = model.init_hidden(config[net]["batch_size"])

        for batch in train_loader:
            input, target = batch
            target = target.view(-1, 1)  # Reshape the target tensor
            # forward pass
            if net == "LSTM":
                h = model.init_hidden(config[net]["batch_size"])
                h = tuple([each.data for each in h])
                input = input.unsqueeze(1)
                output, h = model(input, h)
            else:
                output = model(input)
            loss = criterion(output, target)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train.append(loss.item())

        logger.info(f'Epoch {epoch}/{config[net]["epochs"]}, Loss: {loss_train[-1]}')

        if config["wandb"]["on"]:
            wandb.log({"Loss Train": loss_train[-1]})

    # 5. TESTING
    model.eval()
    if net == "LSTM":
        model.init_hidden(config[net]["batch_size"])

    with torch.no_grad():
        predictions = []
        actual = []
        for batch in test_loader:
            input_test, target_test = batch
            if net == "LSTM":
                input_test = input_test.unsqueeze(1)
                output, h = model(input_test, h)
                optimizer.zero_grad()
                loss_test = criterion(output, target_test)
            else:
                output = model(input_test)

            predictions.append(output.numpy())
            actual.append(target_test.numpy())

        predictions = np.concatenate(predictions, axis=0)
        actual = np.concatenate(actual, axis=0)

        # Rescale the predictions and actual
        predictions = scaler.inverse_transform(
            np.concatenate(
                (test_x[: len(predictions)], predictions.reshape(-1, 1)), axis=1
            )
        )[:, -1]
        actual = scaler.inverse_transform(
            np.concatenate((test_x[: len(actual)], actual.reshape(-1, 1)), axis=1)
        )[:, -1]

        # Calculate performance metrics
        rmse_test = np.sqrt(np.mean((predictions - actual) ** 2))
        r2_test = 1 - np.sum((actual - predictions) ** 2) / np.sum(
            (actual - np.mean(actual)) ** 2
        )

        try:
            mape_test = np.mean(np.abs((actual - predictions) / actual)) * 100
        except:
            mape_test = (
                    np.mean(np.abs((actual - predictions) / (actual + 1e-10))) * 100
            )

        logger.info(f"RMSE: {rmse_test:.2f}, MAPE: {mape_test:.3f}, R2: {r2_test:.2f}")

    # Plot the prediction and actual
    fig_test = plot_graph(ypred=predictions, ylab=actual, title="Test")
    error_dist = error_distribution(predictions, actual)
    scatter = plot_scatter(predictions, actual)

    if config["wandb"]["on"]:
        wandb.log({"RMSE Test": rmse_test, "MAPE Test": mape_test, "R2 Test": r2_test})
        wandb.log({"Test": fig_test})
        wandb.log({"Error Distribution": [wandb.Image(error_dist)]})
        wandb.log({"Scatter": [wandb.Image(scatter)]})
    else:
        logger.info({"RMSE Test": rmse_test, "MAPE Test": mape_test, "R2 Test": r2_test})
        fig_test.show()
        error_dist.show()
        scatter.show()

    # create a dataframe with the predictions and the actual
    df = pd.DataFrame(
        {"Predictions": predictions.flatten(), "Actual": actual.flatten()}
    )

    # save the dataframe in a csv file
    df.to_csv(os.path.join("data", f"predictions_{net}.csv"), index=False)
