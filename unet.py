import torch
from torch import nn


def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )


def upconv2x2(in_channels, out_channels, mode="transpose"):
    if mode == "transpose":
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
    else:
        return nn.Sequential(
            nn.Upsample(mode="bilinear", scale_factor=2),
            conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
        )


class UNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dropout=False):
        super(UNetDownBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout = dropout

        self.conv1 = conv(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )
        self.bn1 = nn.BatchNorm2d(self.out_channels, eps=1e-05)
        self.relu1 = nn.ReLU()

        self.conv2 = conv(self.out_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels, eps=1e-05)
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))

        if self.dropout:
            x = self.dropout1(x)

        return x


class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, merge_mode="concat", up_mode="transpose", dropout=False):
        super(UNetUpBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode
        self.dropout = dropout

        self.upconv = upconv2x2(self.in_channels, self.out_channels, mode=self.up_mode)

        if self.merge_mode == "concat":
            self.conv1 = conv(2 * self.out_channels, self.out_channels)
        else:
            self.conv1 = conv(self.out_channels, self.out_channels)
        self.bn1 = nn.BatchNorm2d(self.out_channels, eps=1e-05)
        self.relu1 = nn.ReLU()
        self.conv2 = conv(self.out_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels, eps=1e-05)
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)

    def forward(self, from_up, from_down):
        from_up = self.upconv(from_up)

        if self.merge_mode == "concat":
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))

        if self.dropout:
            x = self.dropout1(x)

        return x


class UNet(nn.Module):
    def __init__(self, n_channels=3, merge_mode="concat", up_mode="transpose"):
        super(UNet, self).__init__()
        self.n_chnnels = n_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.down1 = UNetDownBlock(self.n_chnnels, 64, 3, 1, 1, dropout = True)
        self.down2 = UNetDownBlock(64, 128, 4, 2, 1, dropout = True)
        self.down3 = UNetDownBlock(128, 256, 4, 2, 1, dropout = True)
        self.down4 = UNetDownBlock(256, 512, 4, 2, 1, dropout = True)
        self.down5 = UNetDownBlock(512, 512, 4, 2, 1, dropout = True)

        self.up1 = UNetUpBlock(512, 512, merge_mode=self.merge_mode, up_mode=self.up_mode, dropout = True)
        self.up2 = UNetUpBlock(512, 256, merge_mode=self.merge_mode, up_mode=self.up_mode, dropout = True)
        self.up3 = UNetUpBlock(256, 128, merge_mode=self.merge_mode, up_mode=self.up_mode, dropout = True)
        self.up4 = UNetUpBlock(128, 64, merge_mode=self.merge_mode, up_mode=self.up_mode, dropout = False)

        self.conv_final = nn.Sequential(conv(64, 3, 3, 1, 1))

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.conv_final(x)

        return x


class UNetAEUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(UNetAEUpBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout

        self.upconv = upconv2x2(self.in_channels, self.out_channels, mode="transpose")

        self.conv1 = conv(self.out_channels, self.out_channels)
        self.bn1 = nn.BatchNorm2d(self.out_channels, eps=1e-05)
        self.relu1 = nn.ReLU()
        self.conv2 = conv(self.out_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels, eps=1e-05)
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)

    def forward(self, from_up):
        x = self.upconv(from_up)

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))

        if self.dropout:
            x = self.dropout1(x)

        return x


class UNetAE(nn.Module):
    def __init__(self, n_channels=3, size = 64, latent_size = 8):
        super(UNetAE, self).__init__()
        self.n_chnnels = n_channels

        self.down1 = UNetDownBlock(self.n_chnnels, size, 3, 1, 1, dropout = True) # 256,256
        self.down2 = UNetDownBlock(size, 2*size, 4, 2, 1, dropout = True)   # 128, 128
        self.down3 = UNetDownBlock(2*size, 4*size, 4, 2, 1, dropout = True) # 64, 64
        self.down4 = UNetDownBlock(4*size, 8*size, 4, 2, 1, dropout = True) # 32,32
        self.down5 = UNetDownBlock(8*size, 8*size, 4, 2, 1, dropout = True) # 16,16

        self.down6 = nn.Sequential( conv(8*size, latent_size, kernel_size=3), 
                                    nn.ReLU(), 
                                    conv(latent_size, 8*size, kernel_size=3),
                                    nn.ReLU())


        self.up1 = UNetAEUpBlock(8*size, 8*size, dropout = True) # 32,32
        self.up2 = UNetAEUpBlock(8*size, 4*size, dropout = True) # 64,64
        self.up3 = UNetAEUpBlock(4*size, 2*size, dropout = True) # 128,128
        self.up4 = UNetAEUpBlock(2*size, size, dropout = False)  # 256,256

        self.conv_final = nn.Sequential(conv(size, 3, 3, 1, 1))

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)

        x = self.down6(x)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.conv_final(x)

        return x
