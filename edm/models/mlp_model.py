import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dims=256, num_inner_layers=2, dropout_rate=0.2, dropout=True, inner_dim=64):
        super(MLP, self).__init__()

        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        layers = []
        for l in range(num_inner_layers):
            input_d = in_dims if l == 0 else inner_dim
            output_d = 1 if l == (num_inner_layers - 1) else inner_dim
            layers.append(nn.Linear(input_d, output_d))
            
            if l != (num_inner_layers - 1):
                # ReLU should not be applied to the last layer
                layers.append(nn.ReLU())
                if dropout:
                    layers.append(nn.Dropout(dropout_rate))
        self.layers = nn.Sequential(*layers)
#         self.layers.apply(init_weights)

    def forward(self, x):
        out = self.layers(x)
        return out
