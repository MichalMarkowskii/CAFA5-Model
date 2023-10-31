import torch
from torch import nn

class CNN1D(nn.Module):
    """
    Model klasyfikacyjny oparty na jednowymiarowej konwolucji.
    """

    def __init__(self, input_dim, num_classes):
        """
        Inicjalizuje model CNN1D.

        Parameters:
        ----------
        input_dim : int
            Wymiar danych wejściowych.
        num_classes : int
            Liczba klas wyjściowych.
        """

        super(CNN1D, self).__init__()
        # (batch_size, channels, embed_size)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, dilation=1, padding=1, stride=1)
        # (batch_size, 3, embed_size)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        # (batch_size, 3, embed_size/2 = 512)
        self.conv2 = nn.Conv1d(in_channels=3, out_channels=8, kernel_size=3, dilation=1, padding=1, stride=1)
        # (batch_size, 8, embed_size/2 = 512)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        # (batch_size, 8, embed_size/4 = 256)
        self.fc1 = nn.Linear(in_features=int(8 * input_dim/4), out_features=864)
        self.fc2 = nn.Linear(in_features=864, out_features=num_classes)

    def forward(self, x):
        """
        Przepuszcza dane wejściowe przez model.

        Parameters:
        ----------
        x : torch.Tensor
            Tensor danych wejściowych.

        Returns:
        -------
        torch.Tensor
            Tensor przewidywań modelu.
        """

        x = x.reshape(x.shape[0], 1, x.shape[1])
        x = self.pool1(nn.functional.tanh(self.conv1(x)))
        x = self.pool2(nn.functional.tanh(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = nn.functional.tanh(self.fc1(x))
        x = self.fc2(x)
        return x
