# Third party imports
import torch

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
