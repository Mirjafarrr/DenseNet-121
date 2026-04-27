import torch
import torch.nn as nn

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, drop_rate=0.2):
        super().__init__()
        inter_channels = 4 * growth_rate
        self.drop_rate = drop_rate
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        
        self.dropout = nn.Dropout(p=self.drop_rate)

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
            
        out = self.conv2(self.relu(self.bn2(out)))
        if self.drop_rate > 0:
            out = self.dropout(out)
            
        return torch.cat([x, out], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, num_layers, growth_rate, drop_rate=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate, drop_rate))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransitionBlock(nn.Module):
    def __init__(self, in_channels, compression=0.5, drop_rate=0.2):
        super().__init__()
        out_channels = int(in_channels * compression)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.drop_rate = drop_rate # Store drop_rate
        self.dropout = nn.Dropout(p=self.drop_rate)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(self.relu(self.bn(x)))
        if self.drop_rate > 0:
            x = self.dropout(x)
        x = self.avgpool(x)
        return x


class DenseNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, growth_rate=32, block_layers=(6, 12, 24, 16), compression=0.5, init_features=64, drop_rate=0.2):
        super().__init__()

        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        num_features = init_features
        
        self.DenseBlock1 = DenseBlock(num_features, num_layers=block_layers[0], growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + block_layers[0] * growth_rate
        self.Transition1 = TransitionBlock(num_features, compression)
        num_features = int(num_features * compression)

        self.DenseBlock2 = DenseBlock(num_features, num_layers=block_layers[1], growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + block_layers[1] * growth_rate
        self.Transition2 = TransitionBlock(num_features, compression)
        num_features = int(num_features * compression)

        self.DenseBlock3 = DenseBlock(num_features, num_layers=block_layers[2], growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + block_layers[2] * growth_rate
        self.Transition3 = TransitionBlock(num_features, compression)
        num_features = int(num_features * compression)

        self.DenseBlock4 = DenseBlock(num_features, num_layers=block_layers[3], growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + block_layers[3] * growth_rate

        self.final_bn = nn.BatchNorm2d(num_features)
        self.relu = nn.ReLU(inplace=True)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.init_conv(x)

        x = self.DenseBlock1(x)
        x = self.Transition1(x)

        x = self.DenseBlock2(x)
        x = self.Transition2(x)

        x = self.DenseBlock3(x)
        x = self.Transition3(x)

        x = self.DenseBlock4(x)

        x = self.relu(self.final_bn(x))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x