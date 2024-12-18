#  Copyright © Roberto Chiosa 2024.
#  Email: roberto.chiosa@polito.it
#  Last edited: 18/12/2024
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


# Create a LSTM network class
class LSTM(torch.nn.Module):
    """
    Long Short-Term Memory (LSTM) network class
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_p):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.fc = torch.nn.Linear(hidden_size, output_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        """
        Forward pass of the LSTM network
        :param x:
        :return:
        """
        # Ensure input is 3D
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension: (batch_size, sequence_length=1, input_size)

        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))

        # Apply dropout and fully connected layer to the last time step
        out = self.dropout(out[:, -1, :])  # Take the output from the last time step
        out = self.fc(out)
        return out


class LSTMSeriesDataset(Dataset):
    def __init__(self, x: np.array, y: np.array, lookback=48):
        self.x = x
        self.y = y
        self.lookback = lookback

    def __len__(self):
        return len(self.x) - self.lookback + 1

    def __getitem__(self, index):
        # Get a sequence of `lookback` steps for each item (for LSTM)
        x_seq = self.x[index: index + self.lookback]
        y_seq = self.y[
            index + self.lookback - 1
            ]  # Target is the last step in the sequence

        # If using an MLP, you can flatten or directly return the final timestep
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(
            y_seq, dtype=torch.float32
        )


def train_lstm(model, optimizer, criterion, data_loader, epochs):
    """
    Train the LSTM model using a DataLoader.
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
        if j > 0 and abs(loss_list[j - 1] - loss_list[j]) < 0.00001:
            break

    train_mae = mean_absolute_error(actual_values, predicted_values)
    train_r2 = r2_score(actual_values, predicted_values)
    train_rmse = root_mean_squared_error(actual_values, predicted_values)
    train_mape = mean_absolute_percentage_error(actual_values, predicted_values)
    logger.info(
        f"[MLP Training] MAE: {train_mae:.4f}, R2: {train_r2:.4f}, RMSE: {train_rmse:.4f}, MAPE: {train_mape:.4f}"
    )

    return loss_list, actual_values, predicted_values


def test_lstm(model, data_loader, criterion):
    """
    Test the LSTM model using a DataLoader.
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
