import torch
import torch.nn as nn

# Implement your ResNet34_UNet model here
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    
# Decoding path, deconvolution + double convolution
class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1) # x2 is from encoder
        return self.conv(x)
    
class ResidualBlock(nn.Module): 
    def __init__(self, in_channel, out_channel, stride=1, shortcut=None):         
        super(ResidualBlock, self).__init__() 
        self.left = nn.Sequential(   
            nn.Conv2d(in_channel, out_channel, 3, stride, 1, bias=False), # bias = False cause BN
            nn.BatchNorm2d(out_channel), 
            nn.ReLU(), 
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False), 
            nn.BatchNorm2d(out_channel), 
            nn.ReLU()
        ) 
        self.right = shortcut
    def forward(self, x): 
        out = self.left(x) 
        residual = x if self.right is None else self.right(x) 
        out = out + residual 
        return out
  
class resnet_34_unet(nn.Module): 
    def __init__(self, in_channels, num_classes): 
        super(resnet_34_unet, self).__init__() 
        # Conv1 = 7x7, 64, stride=2
        # 3x3 Max pool, stride = 2
        self.pre_layer = nn.Sequential( 
            nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(3, 2, 1)
        )
        # Conv2, 3 ResidualBlock
        self.layer1 = self.make_layer(64, 64, 3) 
        # Conv3, 1 stride=2 ResidualBlock and 3 ResidualBlock
        self.layer2 = self.make_layer(64, 128, 4, stride=2) 
        # Conv4, 1 stride=2 ResidualBlock and 5 ResidualBlock
        self.layer3 = self.make_layer(128, 256, 6, stride=2)
        # Conv3, 1 stride=2 ResidualBlock and 2 ResidualBlock
        self.layer4 = self.make_layer(256, 512, 3, stride=2)
        # BottleNeck, use a ResidualBlock to implement
        self.bottleneck = self.make_layer(512, 1024, 1, stride=2)

        self.up_convolution_1 = UpSample(1024, 512)
        self.up_convolution_2 = UpSample(512, 256)
        self.up_convolution_3 = UpSample(256, 128)
        self.up_convolution_4 = UpSample(128, 64)

        self.last = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1),
            nn.Sigmoid()
        )

    def make_layer(self, in_channel, out_channel, block_num, stride=1): 
        shortcut = nn.Sequential( 
            nn.Conv2d(in_channel, out_channel, 1, stride, bias=False), 
            nn.BatchNorm2d(out_channel), 
        ) 
        layers = [] 
        # first layer may need to down sample
        layers.append(ResidualBlock(in_channel, out_channel, stride, shortcut)) 
        for i in range(1, block_num): 
            layers.append(ResidualBlock(out_channel, out_channel)) 
        return nn.Sequential(*layers)
    
    def forward(self, x):
        output = self.pre_layer(x)
        output = self.layer1(output)
        down_1 = output
        output = self.layer2(output)
        down_2 = output
        output = self.layer3(output)
        down_3 = output
        output = self.layer4(output)
        down_4 = output

        output = self.bottleneck(output)

        up_1 = self.up_convolution_1(output, down_4)
        up_2 = self.up_convolution_2(up_1, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)
        
        output = self.last(up_4)
        
        return output




