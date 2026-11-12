import torch.nn as nn
import torch.nn.init as init

ACTIVATIONS = {
    "logistic": nn.Sigmoid,
    "tanh":     nn.Tanh,
    "softsign": nn.Softsign,
    "softplus": nn.Softplus,
    "linear":   nn.Identity,
}

class NeuralNetwork(nn.Module):
    def __init__(self, neurons_linear1=20, neurons_linear2=20, weight_init_method="xavier", sigma=0.1, func="logistic"):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(8, neurons_linear1)
        self.fc2 = nn.Linear(neurons_linear1,neurons_linear2)
        self.fc3 = nn.Linear(neurons_linear2, 1)
        linears = (self.fc1, self.fc2, self.fc3)

        if weight_init_method == "xavier":
            for m in linears:
                init.xavier_uniform_(m.weight)
        elif weight_init_method == "normal":
            for m in linears:
                init.normal_(m.weight, mean=0.0, std=sigma)
        elif weight_init_method == "truncated_normal":
            for m in linears:
                init.trunc_normal_(m.weight, mean=0.0, std=sigma, a=-2*sigma, b=2*sigma)
        else:
            raise ValueError(f"Unknown weight init {weight_init_method!r}")

        try:
            self.func = ACTIVATIONS[func]()
        except KeyError:
            raise ValueError(
                f"Unknown activation {func!r}. "
            ) from None

    def forward(self, x):
        x = self.func(self.fc1(x))
        x = self.func(self.fc2(x))
        x = self.fc3(x)
        return x

