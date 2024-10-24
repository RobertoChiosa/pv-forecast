#  Copyright © Roberto Chiosa 2024.
#  Email: roberto.chiosa@polito.it
#  Last edited: 24/10/2024

# Standard library imports
import json
import logging
import os
from logging import getLogger

# Third party imports
import pandas as pd
import torch
import wandb
from sklearn.preprocessing import MinMaxScaler

# Project imports
from networks import LSTM, MLP, Model
from utils.processing import dataset_dataloader, data_train_test_split, data_preparation_pv
from utils.visualization import *

if __name__ == "__main__":
    # net algorithm
    net_type = 'LSTM'

    # setup logging
    logger = getLogger(__name__)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s.%(funcName)s:%(lineno)d - %(message)s")

    # Read configuration
    logger.info("Reading configuration")
    with open(os.path.join("utils", "config.json")) as f:
        config = json.load(f)

    net = Model(name=net_type, config=config[net_type])

    if config["wandb"]["on"]:
        logger.info("Wandb is on")
        wandb.init(
            project=config["wandb"]["project_name"],
            entity=config["wandb"]["entity"],
            name=net.name + config["wandb"]["run_name"],
            config=config,
        )
    else:
        logger.info("Wandb is off")

    # 1. DATA PREPARATION
    # data_df = data_preparation_gim(filename=os.path.join("data", "dataset_final.csv"))
    data_df = data_preparation_pv(filename=os.path.join("data", "data_9000.csv"))

    # 2. DATA TRANSFORMATION
    # drop the Timestamp column
    data_df = data_df.drop(columns=["Timestamp"])
    train_df, test_df = data_train_test_split(data_df)

    # Normalize the data min max scaling
    scaler = MinMaxScaler()
    train_df = scaler.fit_transform(train_df)
    test_df = scaler.transform(test_df)

    # Split the data in input and output (nb. the output is the last column)
    train_x, train_y = train_df[:, :-1], train_df[:, 1]
    test_x, test_y = test_df[:, :-1], test_df[:, 1]

    input_size = train_x.shape[-1]

    # Dataset and dataloader
    train_tensor, train_loader = dataset_dataloader(
        train_x, train_y, net.batch_size, shuffle=True
    )
    test_tensor, test_loader = dataset_dataloader(
        test_x, test_y, net.batch_size, shuffle=False
    )

    # 3. MODEL INITIALIZATION
    if net.name == "MLP":
        model = MLP(
            input_size=input_size,
            hidden_size=net.hidden_size,
            output_size=net.output_size,
            num_layers=net.num_layers,
            dropout_p=net.dropout_p,
        )

    elif net.name == "LSTM":
        model = LSTM(
            input_size=input_size,
            hidden_size=net.hidden_size,
            output_size=net.output_size,
            num_layers=net.num_layers,
            dropout_p=net.dropout_p,
        )

    # Initialize the optimizer and loss function criterion
    criterion = torch.nn.MSELoss()
    optimizer = getattr(torch.optim, net.optimizer)(
        model.parameters(), lr=net.learning_rate
    )

    # 4. TRAINING
    loss_train = []

    for epoch in range(net.epochs):
        model.train()
        if net.name == "LSTM":
            h = model.init_hidden(net.batch_size)

        for batch in train_loader:
            input, target = batch
            target = target.view(-1, 1)  # Reshape the target tensor
            # forward pass
            if net.name == "LSTM":
                h = model.init_hidden(net.batch_size)
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

        logger.info(f'Epoch {epoch}/{net.epochs}, Loss: {loss_train[-1]}')

        if config["wandb"]["on"]:
            wandb.log({"Loss Train": loss_train[-1]})

    # 5. TESTING
    model.eval()
    if net.name == "LSTM":
        model.init_hidden(net.batch_size)

    with torch.no_grad():
        predictions = []
        actual = []
        for batch in test_loader:
            input_test, target_test = batch
            if net.name == "LSTM":
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
        except ZeroDivisionError:
            logger.warning("Actual values contain zero values, fixing MAPE calculation")
            mape_test = (
                    np.mean(np.abs((actual - predictions) / (actual + 1e-10))) * 100
            )

        logger.info(f"RMSE_test: {rmse_test:.4f}, MAPE_test: {mape_test:.4f}, R2_test: {r2_test:.4f}")

    # Plot the prediction and actual
    fig_line_plot = plot_graph(y_pred=predictions, y_real=actual, title="Test")
    fig_error_dist = error_distribution(y_pred=predictions, y_real=actual)
    fig_scatter = plot_scatter(y_pred=predictions, y_real=actual)

    if config["wandb"]["on"]:
        wandb.log({"RMSE Test": rmse_test, "MAPE Test": mape_test, "R2 Test": r2_test})
        wandb.log({"Test": fig_line_plot})
        wandb.log({"Error Distribution": [wandb.Image(fig_error_dist)]})
        wandb.log({"Scatter": [wandb.Image(fig_scatter)]})
    else:
        fig_line_plot.savefig(
            os.path.join('out', f"{net.name}_line_plot.png")
        )
        fig_error_dist.savefig(
            os.path.join('out', f"{net.name}_error.png")
        )
        fig_scatter.savefig(
            os.path.join('out', f"{net.name}_scatter.png")
        )

    # create a dataframe with the predictions and the actual
    df = pd.DataFrame(
        {"Predictions": predictions.flatten(), "Actual": actual.flatten()}
    )

    # save the dataframe in a csv file
    df.to_csv(os.path.join("data", f"predictions_{net.name}.csv"), index=False)
