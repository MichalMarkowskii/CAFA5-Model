import torch


class MultiLayerPerceptron(torch.nn.Module):
    """
    Model klasyfikacyjny oparty na warstwach w pełni połączonych.
    """

    def __init__(self, input_dim, num_classes):
        """
        Inicjalizuje model MultiLayerPerceptron.

        Parameters:
        ----------
        input_dim : int
            Wymiar danych wejściowych.
        num_classes : int
            Liczba klas wyjściowych.
        """

        super(MultiLayerPerceptron, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, 864)
        self.activation1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(864, 712)
        self.activation2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(712, num_classes)

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

        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.linear3(x)
        return x
