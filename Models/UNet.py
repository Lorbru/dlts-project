import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF

class EncodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncodeBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,kernel_size = 5, stride = 2,padding =  2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope = 0.2, inplace=True),
        )
    def forward(self, x):
        return self.conv(x)
    
class DecodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_out=True):
        super(DecodeBlock, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        
        if drop_out: 
            layers.append(nn.Dropout(0.5))
            
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(
            self, in_channels=1, out_channels=1, features=[16, 32, 64, 128, 256],
    ):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        # Down part of UNET
        _in_channels = in_channels
        for feature in features:
            self.downs.append(EncodeBlock(_in_channels, feature))
            _in_channels = feature

        self.bottleneck = EncodeBlock(features[-1], features[-1]*2)
        
        # Up part of UNET
        self.ups.append(DecodeBlock(features[-1]*2, features[-1], drop_out = (features[-1] >= 64)))
        
        for feature in reversed(features[:-1]):
            self.ups.append(DecodeBlock(feature*4, feature, drop_out = (feature >= 64)))

        self.final_conv = DecodeBlock(features[0]*2, out_channels)

    def forward(self, x):
        input_x = x
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        

        for idx, up in enumerate(self.ups):
            x = up(x)
            skip_connection = skip_connections[idx]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            
            x = torch.cat((skip_connection, x), dim=1)
        
        x = self.final_conv(x)
        if x.shape != input_x.shape:
                x = TF.resize(x, size=input_x.shape[2:])
        return x * input_x