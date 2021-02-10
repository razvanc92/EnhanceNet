import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, hiddens, input_size, activation_function, out_act):
        super(MLP, self).__init__()
        layers = []

        previous_h = input_size
        for i, h in enumerate(hiddens):
            activation = None if i == len(hiddens) - 1 and not out_act else activation_function
            layers.append(nn.Linear(previous_h, h))

            if activation is not None:
                layers.append(activation)

            previous_h = h
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)
