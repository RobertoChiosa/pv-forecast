#  Copyright © Roberto Chiosa 2024.
#  Email: roberto.chiosa@polito.it
#  Last edited: 29/10/2024
# Standard library imports
from logging import getLogger

# Third party imports
import torch
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
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        # Return each sample and target as tensors
        return torch.tensor(self.x[index], dtype=torch.float32), torch.tensor(
            self.y[index], dtype=torch.float32
        )


def train_mlp(model, optimizer, criterion, data_loader, epochs):
    """
    Train the MLP model using a DataLoader.
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
        logger.info(f"[MLP Training] Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        # Stopping criterion
        if j > 0 and abs(loss_list[j - 1] - loss_list[j]) < 0.000001:
            break

    return loss_list
