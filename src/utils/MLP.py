#  Copyright © Roberto Chiosa 2024.
#  Email: roberto.chiosa@polito.it
#  Last edited: 29/10/2024
# Standard library imports
from logging import getLogger

# Third party imports
import numpy as np
import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)
from torch.utils.data import Dataset

# setup logging
logger = getLogger(__name__)

device = "cpu"


class MLP(torch.nn.Module):
    """
    Multi-layer perceptron (MLP) network class
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_p):
        super(MLP, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(input_size, hidden_size))
        for _ in range(num_layers - 1):
            self.layers.append(torch.nn.Linear(hidden_size, hidden_size))
        self.layers.append(torch.nn.Linear(hidden_size, output_size))
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, x):
        """
        Forward pass of the MLP network
        :param x:
        :return:
        """
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.relu(x)
            x = self.dropout(x)
        x = self.layers[-1](x)
        return x


class MLPTimeseriesDataset(Dataset):
    def __init__(self, x: np.array, y: np.array):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        # Return each sample and target as tensors
        return (
            torch.tensor(self.x[index], dtype=torch.float32),
            torch.tensor(self.y[index], dtype=torch.float32),
        )


def train_mlp(model, optimizer, criterion, data_loader, epochs):
    """
    Train the MLP model using a DataLoader.
    :param model:
    :param optimizer:
    :param criterion:
    :param data_loader:
    :param epochs:
    :return:
    """
    loss_list = []
    actual_values = []
    predicted_values = []

    for j, epoch in enumerate(range(epochs)):
        epoch_loss = 0.0
        for inputs, targets in data_loader:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.view(-1), targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            actual_values.extend(targets.cpu().numpy())
            predicted_values.extend(outputs.detach().numpy())

        avg_loss = epoch_loss / len(data_loader)
        loss_list.append(avg_loss)
        logger.info(f"[MLP Training] Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        # Stopping criterion
        if j > 0 and abs(loss_list[j - 1] - loss_list[j]) < 0.000001:
            break

    train_mae = mean_absolute_error(actual_values, predicted_values)
    train_r2 = r2_score(actual_values, predicted_values)
    train_rmse = root_mean_squared_error(actual_values, predicted_values)
    train_mape = mean_absolute_percentage_error(actual_values, predicted_values)
    logger.info(
        f"[MLP Training] MAE: {train_mae:.4f}, R2: {train_r2:.4f}, RMSE: {train_rmse:.4f}, MAPE: {train_mape:.4f}"
    )

    return loss_list, actual_values, predicted_values


def test_mlp(model, data_loader, criterion):
    """
    Test the MLP model using a DataLoader.
    :param model:
    :param data_loader:
    :param criterion:
    :return:
    """
    model.eval()
    loss_list = []
    actual_values = []
    predicted_values = []
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.view(-1), targets)

            loss_list.append(loss.item())

            actual_values.extend(targets.cpu().numpy())
            predicted_values.extend(outputs.detach().numpy())

    avg_loss = sum(loss_list) / len(data_loader)
    logger.info(f"[MLP Testing] Average loss: {avg_loss:.4f}")

    train_mae = mean_absolute_error(actual_values, predicted_values)
    train_r2 = r2_score(actual_values, predicted_values)
    train_rmse = root_mean_squared_error(actual_values, predicted_values)
    train_mape = mean_absolute_percentage_error(actual_values, predicted_values)
    logger.info(
        f"[MLP Testing] MAE: {train_mae:.4f}, R2: {train_r2:.4f}, RMSE: {train_rmse:.4f}, MAPE: {train_mape:.4f}"
    )

    return loss_list, actual_values, predicted_values
