import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes=10, num_filters=32, filter_size=3, activation_fn=nn.ReLU, dense_neurons=128,
                 dropout=0.2, batchnorm=False, filter_organization='same'):
        super(CNN, self).__init__()
        layers = []
        in_channels = 3

        if filter_organization == "same":
            filter_counts = [num_filters] * 5
        elif filter_organization == "double":
            filter_counts = [num_filters * (2 ** i) for i in range(5)]
        elif filter_organization == "half":
            filter_counts = [max(4, num_filters // (2 ** i)) for i in range(5)]

        for filters in filter_counts:
            layers.append(nn.Conv2d(in_channels, filters, kernel_size=filter_size, padding=1))
            if batchnorm:
                layers.append(nn.BatchNorm2d(filters))
            layers.append(activation_fn())
            layers.append(nn.MaxPool2d(2))
            in_channels = filters

        self.conv = nn.Sequential(*layers)
        self.flattened_size = filter_counts[-1] * (128 // (2**5))**2

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, dense_neurons),
            activation_fn(),
            nn.Dropout(dropout),
            nn.Linear(dense_neurons, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
