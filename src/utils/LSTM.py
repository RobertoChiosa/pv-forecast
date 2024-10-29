#  Copyright © Roberto Chiosa 2024.
#  Email: roberto.chiosa@polito.it
#  Last edited: 29/10/2024
from logging import getLogger

import numpy as np
# Third party imports
import torch
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

    def init_hidden(self, batch_size):
        """
        Initialize hidden states and cell states
        :param batch_size:
        :return:
        """
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        hidden = (h0, c0)
        return hidden

    def forward(self, x, hidden_cell_tuple):
        """
        Forward pass of the LSTM network
        :param x:
        :param hidden_cell_tuple:
        :return:
        """
        batch_size, seq_len, _ = x.size()
        out, hidden_cell_tuple = self.lstm(x, hidden_cell_tuple)
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return out, hidden_cell_tuple


class LSTMSeriesDataset(Dataset):
    def __init__(self, x: np.array, y: np.array, lookback=1):
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
    """
    loss_list = []

    for j, epoch in enumerate(range(epochs)):
        epoch_loss = 0.0
        for inputs, targets in data_loader:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(data_loader)
        loss_list.append(avg_loss)
        logger.info(f"[LSTM Training]Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        # Stopping criterion
        if j > 0 and abs(loss_list[j - 1] - loss_list[j]) < 0.000001:
            break

    return loss_list
