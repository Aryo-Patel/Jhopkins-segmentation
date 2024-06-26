import torch
import torch.nn as nn

from constants import DEVICE


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()

        self.downs = nn.ParameterList()
        for feature in features:
            self.downs.append(DoubleConv(in_channels=in_channels, out_channels=feature))
            in_channels = feature

        self.last_dbl_conv = DoubleConv(
            in_channels=features[-1], out_channels=features[-1] * 2
        )

        self.ups = nn.ParameterList()
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    in_channels=feature * 2,
                    out_channels=feature,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.ups.append(DoubleConv(in_channels=feature * 2, out_channels=feature))

        self.final_conv = nn.Sequential(
            nn.Conv2d(features[0], out_channels, 3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):

        self.resid_connections = []

        # Channel expansion
        for down in self.downs:
            x = down(x)
            self.resid_connections.append(x)
            x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        x = self.last_dbl_conv(x)
        self.resid_connections = list(reversed(self.resid_connections))

        # channel compression
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            x = torch.cat([self.resid_connections[idx // 2], x], dim=1)
            x = self.ups[idx + 1](x)

        # prepare for output
        return torch.sigmoid(self.final_conv(x))
