import torch
import torch.nn as nn
import torch.nn.functional as F
from torchstat import stat
from torchsummary import summary


# The Definition of Models : UXNet  UXNet4  UXNet5  UXNet6  UXNet7  UXNetCAM  UXNetSAM  UXNetCBAM  UXNet765CAM4SMALLSAM
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout


class BNREDWSCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(BNREDWSCONV, self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate, groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0,
                                    groups=1)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.point_conv(self.conv_s1(hx))))

        return xout


# upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src, tar):
    src = F.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=True)

    return src


class NUSUnit7S(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(NUSUnit7S, self).__init__()

        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv1 = REBNCONV(in_ch, out_ch, dirate=1)
        self.conv2 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.conv3 = REBNCONV(mid_ch, out_ch, dirate=1)
        self.conv4 = BNREDWSCONV(mid_ch, mid_ch, dirate=1)
        self.conv5 = BNREDWSCONV(mid_ch, mid_ch, dirate=2)
        self.conv6 = BNREDWSCONV(mid_ch, mid_ch, dirate=5)
        self.sa = SpatialAttention()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        hxin = self.conv1(hx)

        hx1 = self.conv2(hxin)
        hx = self.pool(hx1)

        hx2 = self.conv4(hx)
        hx = self.pool(hx2)

        hx3 = self.conv5(hx)
        hx = self.pool(hx3)

        hx4 = self.conv6(hx)
        hx = self.pool(hx4)

        hx5 = self.conv4(hx)
        hx = self.pool(hx5)

        hx6 = self.conv5(hx)

        hx7 = self.conv6(hx6)

        hx6d = self.conv4(torch.add(hx7, hx6))
        hx6dup = _upsample_like(hx6d, hx5)

        hx5d = self.conv4(torch.add(hx6dup, hx5))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.conv4(torch.add(hx5dup, hx4))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.conv4(torch.add(hx4dup, hx3))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.conv4(torch.add(hx3dup, hx2))
        hx2dup = _upsample_like(hx2d, hx1)

        hxinout = self.conv3(torch.add(hx2dup, hx1))

        hxinout = self.sa(hxinout) * hxinout

        return self.relu(hxin + hxinout)


class NUSUnit7C(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(NUSUnit7C, self).__init__()

        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv1 = REBNCONV(in_ch, out_ch, dirate=1)
        self.conv2 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.conv3 = REBNCONV(mid_ch, out_ch, dirate=1)
        self.conv4 = BNREDWSCONV(mid_ch, mid_ch, dirate=1)
        self.conv5 = BNREDWSCONV(mid_ch, mid_ch, dirate=2)
        self.conv6 = BNREDWSCONV(mid_ch, mid_ch, dirate=5)
        self.ca = ChannelAttention(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        hxin = self.conv1(hx)

        hx1 = self.conv2(hxin)
        hx = self.pool(hx1)

        hx2 = self.conv4(hx)
        hx = self.pool(hx2)

        hx3 = self.conv5(hx)
        hx = self.pool(hx3)

        hx4 = self.conv6(hx)
        hx = self.pool(hx4)

        hx5 = self.conv4(hx)
        hx = self.pool(hx5)

        hx6 = self.conv5(hx)

        hx7 = self.conv6(hx6)

        hx6d = self.conv4(torch.add(hx7, hx6))
        hx6dup = _upsample_like(hx6d, hx5)

        hx5d = self.conv4(torch.add(hx6dup, hx5))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.conv4(torch.add(hx5dup, hx4))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.conv4(torch.add(hx4dup, hx3))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.conv4(torch.add(hx3dup, hx2))
        hx2dup = _upsample_like(hx2d, hx1)

        hxinout = self.conv3(torch.add(hx2dup, hx1))

        hxinout = self.ca(hxinout) * hxinout

        return self.relu(hxin + hxinout)


class NUSUnit7CS(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(NUSUnit7CS, self).__init__()

        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv1 = REBNCONV(in_ch, out_ch, dirate=1)
        self.conv2 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.conv3 = REBNCONV(mid_ch, out_ch, dirate=1)
        self.conv4 = BNREDWSCONV(mid_ch, mid_ch, dirate=1)
        self.conv5 = BNREDWSCONV(mid_ch, mid_ch, dirate=2)
        self.conv6 = BNREDWSCONV(mid_ch, mid_ch, dirate=5)
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        hxin = self.conv1(hx)

        hx1 = self.conv2(hxin)
        hx = self.pool(hx1)

        hx2 = self.conv4(hx)
        hx = self.pool(hx2)

        hx3 = self.conv5(hx)
        hx = self.pool(hx3)

        hx4 = self.conv6(hx)
        hx = self.pool(hx4)

        hx5 = self.conv4(hx)
        hx = self.pool(hx5)

        hx6 = self.conv5(hx)

        hx7 = self.conv6(hx6)

        hx6d = self.conv4(torch.add(hx7, hx6))
        hx6dup = _upsample_like(hx6d, hx5)

        hx5d = self.conv4(torch.add(hx6dup, hx5))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.conv4(torch.add(hx5dup, hx4))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.conv4(torch.add(hx4dup, hx3))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.conv4(torch.add(hx3dup, hx2))
        hx2dup = _upsample_like(hx2d, hx1)

        hxinout = self.conv3(torch.add(hx2dup, hx1))

        hxinout = self.ca(hxinout) * hxinout
        hxinout = self.sa(hxinout) * hxinout

        return self.relu(hxin + hxinout)


class NUSUnit6S(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(NUSUnit6S, self).__init__()

        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv1 = REBNCONV(in_ch, out_ch, dirate=1)
        self.conv2 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.conv3 = REBNCONV(mid_ch, out_ch, dirate=1)
        self.conv4 = BNREDWSCONV(mid_ch, mid_ch, dirate=1)
        self.conv5 = BNREDWSCONV(mid_ch, mid_ch, dirate=2)
        self.conv6 = BNREDWSCONV(mid_ch, mid_ch, dirate=5)
        self.sa = SpatialAttention()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        hxin = self.conv1(hx)

        hx1 = self.conv2(hxin)
        hx = self.pool(hx1)

        hx2 = self.conv5(hx)
        hx = self.pool(hx2)

        hx3 = self.conv6(hx)
        hx = self.pool(hx3)

        hx4 = self.conv4(hx)
        hx = self.pool(hx4)

        hx5 = self.conv5(hx)

        hx6 = self.conv6(hx5)

        hx5d = self.conv4(torch.add(hx6, hx5))
        hx5up = _upsample_like(hx5d, hx4)

        hx4d = self.conv4(torch.add(hx5up, hx4))
        hx4up = _upsample_like(hx4d, hx3)

        hx3d = self.conv4(torch.add(hx4up, hx3))
        hx3up = _upsample_like(hx3d, hx2)

        hx2d = self.conv4(torch.add(hx3up, hx2))
        hx2up = _upsample_like(hx2d, hx1)

        hxinout = self.conv3(torch.add(hx2up, hx1))

        hxinout = self.sa(hxinout) * hxinout

        return self.relu(hxin + hxinout)


class NUSUnit6C(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(NUSUnit6C, self).__init__()

        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv1 = REBNCONV(in_ch, out_ch, dirate=1)
        self.conv2 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.conv3 = REBNCONV(mid_ch, out_ch, dirate=1)
        self.conv4 = BNREDWSCONV(mid_ch, mid_ch, dirate=1)
        self.conv5 = BNREDWSCONV(mid_ch, mid_ch, dirate=2)
        self.conv6 = BNREDWSCONV(mid_ch, mid_ch, dirate=5)
        self.ca = ChannelAttention(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        hxin = self.conv1(hx)

        hx1 = self.conv2(hxin)
        hx = self.pool(hx1)

        hx2 = self.conv5(hx)
        hx = self.pool(hx2)

        hx3 = self.conv6(hx)
        hx = self.pool(hx3)

        hx4 = self.conv4(hx)
        hx = self.pool(hx4)

        hx5 = self.conv5(hx)

        hx6 = self.conv6(hx5)

        hx5d = self.conv4(torch.add(hx6, hx5))
        hx5up = _upsample_like(hx5d, hx4)

        hx4d = self.conv4(torch.add(hx5up, hx4))
        hx4up = _upsample_like(hx4d, hx3)

        hx3d = self.conv4(torch.add(hx4up, hx3))
        hx3up = _upsample_like(hx3d, hx2)

        hx2d = self.conv4(torch.add(hx3up, hx2))
        hx2up = _upsample_like(hx2d, hx1)

        hxinout = self.conv3(torch.add(hx2up, hx1))

        hxinout = self.ca(hxinout) * hxinout

        return self.relu(hxin + hxinout)


class NUSUnit6CS(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(NUSUnit6CS, self).__init__()

        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv1 = REBNCONV(in_ch, out_ch, dirate=1)
        self.conv2 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.conv3 = REBNCONV(mid_ch, out_ch, dirate=1)
        self.conv4 = BNREDWSCONV(mid_ch, mid_ch, dirate=1)
        self.conv5 = BNREDWSCONV(mid_ch, mid_ch, dirate=2)
        self.conv6 = BNREDWSCONV(mid_ch, mid_ch, dirate=5)
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        hxin = self.conv1(hx)

        hx1 = self.conv2(hxin)
        hx = self.pool(hx1)

        hx2 = self.conv5(hx)
        hx = self.pool(hx2)

        hx3 = self.conv6(hx)
        hx = self.pool(hx3)

        hx4 = self.conv4(hx)
        hx = self.pool(hx4)

        hx5 = self.conv5(hx)

        hx6 = self.conv6(hx5)

        hx5d = self.conv4(torch.add(hx6, hx5))
        hx5up = _upsample_like(hx5d, hx4)

        hx4d = self.conv4(torch.add(hx5up, hx4))
        hx4up = _upsample_like(hx4d, hx3)

        hx3d = self.conv4(torch.add(hx4up, hx3))
        hx3up = _upsample_like(hx3d, hx2)

        hx2d = self.conv4(torch.add(hx3up, hx2))
        hx2up = _upsample_like(hx2d, hx1)

        hxinout = self.conv3(torch.add(hx2up, hx1))

        hxinout = self.ca(hxinout) * hxinout
        hxinout = self.sa(hxinout) * hxinout

        return self.relu(hxin + hxinout)


class NUSUnit5S(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(NUSUnit5S, self).__init__()

        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv1 = REBNCONV(in_ch, out_ch, dirate=1)
        self.conv2 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.conv3 = REBNCONV(mid_ch, out_ch, dirate=1)
        self.conv4 = BNREDWSCONV(mid_ch, mid_ch, dirate=1)
        self.conv5 = BNREDWSCONV(mid_ch, mid_ch, dirate=2)
        self.sa = SpatialAttention()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        hxin = self.conv1(hx)

        hx1 = self.conv2(hxin)
        hx = self.pool(hx1)

        hx2 = self.conv4(hx)
        hx = self.pool(hx2)

        hx3 = self.conv5(hx)
        hx = self.pool(hx3)

        hx4 = self.conv4(hx)

        hx5 = self.conv5(hx4)

        hx4d = self.conv4(torch.add(hx5, hx4))
        hx4up = _upsample_like(hx4d, hx3)

        hx3d = self.conv4(torch.add(hx4up, hx3))
        hx3up = _upsample_like(hx3d, hx2)

        hx2d = self.conv4(torch.add(hx3up, hx2))
        hx2up = _upsample_like(hx2d, hx1)

        hxinout = self.conv3(torch.add(hx2up, hx1))

        hxinout = self.sa(hxinout) * hxinout

        return self.relu(hxin + hxinout)


class NUSUnit5C(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(NUSUnit5C, self).__init__()

        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv1 = REBNCONV(in_ch, out_ch, dirate=1)
        self.conv2 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.conv3 = REBNCONV(mid_ch, out_ch, dirate=1)
        self.conv4 = BNREDWSCONV(mid_ch, mid_ch, dirate=1)
        self.conv5 = BNREDWSCONV(mid_ch, mid_ch, dirate=2)
        self.ca = ChannelAttention(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        hxin = self.conv1(hx)

        hx1 = self.conv2(hxin)
        hx = self.pool(hx1)

        hx2 = self.conv4(hx)
        hx = self.pool(hx2)

        hx3 = self.conv5(hx)
        hx = self.pool(hx3)

        hx4 = self.conv4(hx)

        hx5 = self.conv5(hx4)

        hx4d = self.conv4(torch.add(hx5, hx4))
        hx4up = _upsample_like(hx4d, hx3)

        hx3d = self.conv4(torch.add(hx4up, hx3))
        hx3up = _upsample_like(hx3d, hx2)

        hx2d = self.conv4(torch.add(hx3up, hx2))
        hx2up = _upsample_like(hx2d, hx1)

        hxinout = self.conv3(torch.add(hx2up, hx1))

        hxinout = self.ca(hxinout) * hxinout

        return self.relu(hxin + hxinout)


class NUSUnit5CS(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(NUSUnit5CS, self).__init__()

        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv1 = REBNCONV(in_ch, out_ch, dirate=1)
        self.conv2 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.conv3 = REBNCONV(mid_ch, out_ch, dirate=1)
        self.conv4 = BNREDWSCONV(mid_ch, mid_ch, dirate=1)
        self.conv5 = BNREDWSCONV(mid_ch, mid_ch, dirate=2)
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        hxin = self.conv1(hx)

        hx1 = self.conv2(hxin)
        hx = self.pool(hx1)

        hx2 = self.conv4(hx)
        hx = self.pool(hx2)

        hx3 = self.conv5(hx)
        hx = self.pool(hx3)

        hx4 = self.conv4(hx)

        hx5 = self.conv5(hx4)

        hx4d = self.conv4(torch.add(hx5, hx4))
        hx4up = _upsample_like(hx4d, hx3)

        hx3d = self.conv4(torch.add(hx4up, hx3))
        hx3up = _upsample_like(hx3d, hx2)

        hx2d = self.conv4(torch.add(hx3up, hx2))
        hx2up = _upsample_like(hx2d, hx1)

        hxinout = self.conv3(torch.add(hx2up, hx1))

        hxinout = self.ca(hxinout) * hxinout
        hxinout = self.sa(hxinout) * hxinout

        return self.relu(hxin + hxinout)


class NUSUnit4C(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(NUSUnit4C, self).__init__()

        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv1 = REBNCONV(in_ch, out_ch, dirate=1)
        self.conv2 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.conv3 = REBNCONV(mid_ch, out_ch, dirate=1)
        self.conv4 = BNREDWSCONV(mid_ch, mid_ch, dirate=1)
        self.conv5 = BNREDWSCONV(mid_ch, mid_ch, dirate=2)
        self.ca = ChannelAttention(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        hxin = self.conv1(hx)

        hx1 = self.conv2(hxin)
        hx = self.pool(hx1)

        hx2 = self.conv5(hx)
        hx = self.pool(hx2)

        hx3 = self.conv4(hx)

        hx4 = self.conv5(hx3)

        hx3d = self.conv4(torch.add(hx4, hx3))
        hx3up = _upsample_like(hx3d, hx2)

        hx2d = self.conv4(torch.add(hx3up, hx2))
        hx2up = _upsample_like(hx2d, hx1)

        hxinout = self.conv3(torch.add(hx2up, hx1))

        hxinout = self.ca(hxinout) * hxinout

        return self.relu(hxin + hxinout)


class NUSUnit4S(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(NUSUnit4S, self).__init__()

        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv1 = REBNCONV(in_ch, out_ch, dirate=1)
        self.conv2 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.conv3 = REBNCONV(mid_ch, out_ch, dirate=1)
        self.conv4 = BNREDWSCONV(mid_ch, mid_ch, dirate=1)
        self.conv5 = BNREDWSCONV(mid_ch, mid_ch, dirate=2)
        self.sa = SpatialAttention()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        hxin = self.conv1(hx)

        hx1 = self.conv2(hxin)
        hx = self.pool(hx1)

        hx2 = self.conv5(hx)
        hx = self.pool(hx2)

        hx3 = self.conv4(hx)

        hx4 = self.conv5(hx3)

        hx3d = self.conv4(torch.add(hx4, hx3))
        hx3up = _upsample_like(hx3d, hx2)

        hx2d = self.conv4(torch.add(hx3up, hx2))
        hx2up = _upsample_like(hx2d, hx1)

        hxinout = self.conv3(torch.add(hx2up, hx1))

        hxinout = self.sa(hxinout) * hxinout

        return self.relu(hxin + hxinout)


class NUSUnit4CS(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(NUSUnit4CS, self).__init__()

        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv1 = REBNCONV(in_ch, out_ch, dirate=1)
        self.conv2 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.conv3 = REBNCONV(mid_ch, out_ch, dirate=1)
        self.conv4 = BNREDWSCONV(mid_ch, mid_ch, dirate=1)
        self.conv5 = BNREDWSCONV(mid_ch, mid_ch, dirate=2)
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        hxin = self.conv1(hx)

        hx1 = self.conv2(hxin)
        hx = self.pool(hx1)

        hx2 = self.conv5(hx)
        hx = self.pool(hx2)

        hx3 = self.conv4(hx)

        hx4 = self.conv5(hx3)

        hx3d = self.conv4(torch.add(hx4, hx3))
        hx3up = _upsample_like(hx3d, hx2)

        hx2d = self.conv4(torch.add(hx3up, hx2))
        hx2up = _upsample_like(hx2d, hx1)

        hxinout = self.conv3(torch.add(hx2up, hx1))

        hxinout = self.ca(hxinout) * hxinout
        hxinout = self.sa(hxinout) * hxinout

        return self.relu(hxin + hxinout)


class NUSUnitSmallS(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(NUSUnitSmallS, self).__init__()

        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv1 = REBNCONV(in_ch, out_ch, dirate=1)
        self.conv2 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.conv3 = REBNCONV(mid_ch, out_ch, dirate=1)
        self.conv4 = BNREDWSCONV(mid_ch, mid_ch, dirate=1)
        self.conv5 = BNREDWSCONV(mid_ch, mid_ch, dirate=2)
        self.sa = SpatialAttention()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        hxin = self.conv1(hx)

        hx1 = self.conv2(hxin)

        hx2 = self.conv5(hx1)

        hx3 = self.conv4(hx2)

        hx4 = self.conv5(hx3)

        hx3d = self.conv4(torch.add(hx4, hx3))

        hx2d = self.conv4(torch.add(hx3d, hx2))

        hxinout = self.conv3(torch.add(hx2d, hx1))

        hxinout = self.sa(hxinout) * hxinout

        return self.relu(hxin + hxinout)


class NUSUnitSmallC(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(NUSUnitSmallC, self).__init__()

        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv1 = REBNCONV(in_ch, out_ch, dirate=1)
        self.conv2 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.conv3 = REBNCONV(mid_ch, out_ch, dirate=1)
        self.conv4 = BNREDWSCONV(mid_ch, mid_ch, dirate=1)
        self.conv5 = BNREDWSCONV(mid_ch, mid_ch, dirate=2)
        self.ca = ChannelAttention(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        hxin = self.conv1(hx)

        hx1 = self.conv2(hxin)

        hx2 = self.conv5(hx1)

        hx3 = self.conv4(hx2)

        hx4 = self.conv5(hx3)

        hx3d = self.conv4(torch.add(hx4, hx3))

        hx2d = self.conv4(torch.add(hx3d, hx2))

        hxinout = self.conv3(torch.add(hx2d, hx1))

        hxinout = self.ca(hxinout) * hxinout

        return self.relu(hxin + hxinout)


class NUSUnitSmallCS(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(NUSUnitSmallCS, self).__init__()

        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv1 = REBNCONV(in_ch, out_ch, dirate=1)
        self.conv2 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.conv3 = REBNCONV(mid_ch, out_ch, dirate=1)
        self.conv4 = BNREDWSCONV(mid_ch, mid_ch, dirate=1)
        self.conv5 = BNREDWSCONV(mid_ch, mid_ch, dirate=2)
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        hxin = self.conv1(hx)

        hx1 = self.conv2(hxin)

        hx2 = self.conv5(hx1)

        hx3 = self.conv4(hx2)

        hx4 = self.conv5(hx3)

        hx3d = self.conv4(torch.add(hx4, hx3))

        hx2d = self.conv4(torch.add(hx3d, hx2))

        hxinout = self.conv3(torch.add(hx2d, hx1))

        hxinout = self.ca(hxinout) * hxinout
        hxinout = self.sa(hxinout) * hxinout

        return self.relu(hxin + hxinout)


class NUSNet(nn.Module):

    def __init__(self, in_ch=3, out_ch=1):
        super(NUSNet, self).__init__()

        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.en1 = NUSUnit7S(in_ch, 32, 64)
        self.en2 = NUSUnit6S(64, 32, 128)
        self.en3 = NUSUnit5S(128, 64, 256)
        self.en4 = NUSUnit4C(256, 128, 512)
        self.en5 = NUSUnitSmallC(512, 256, 512)

        self.en6 = NUSUnitSmallC(512, 256, 512)

        self.de5 = NUSUnitSmallC(512, 256, 512)
        self.de4 = NUSUnit4C(512, 128, 256)
        self.de3 = NUSUnit5S(256, 64, 128)
        self.de2 = NUSUnit6S(128, 32, 64)
        self.de1 = NUSUnit7S(64, 32, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6, out_ch, 1)

    def forward(self, x):
        hx = x

        # encoder
        hx1 = self.en1(hx)
        hx = self.pool(hx1)

        hx2 = self.en2(hx)
        hx = self.pool(hx2)

        hx3 = self.en3(hx)
        hx = self.pool(hx3)

        hx4 = self.en4(hx)
        hx = self.pool(hx4)

        hx5 = self.en5(hx)

        hx6 = self.en6(hx5)

        # decoder
        hx5d = self.de5(torch.add(hx6, hx5))
        hx5up = _upsample_like(hx5d, hx4)

        hx4d = self.de4(torch.add(hx5up, hx4))
        hx4up = _upsample_like(hx4d, hx3)

        hx3d = self.de3(torch.add(hx4up, hx3))
        hx3up = _upsample_like(hx3d, hx2)

        hx2d = self.de2(torch.add(hx3up, hx2))
        hx2up = _upsample_like(hx2d, hx1)

        hx1d = self.de1(hx2up)

        # output
        sup1 = self.side1(hx1d)

        sup2 = self.side2(hx2d)
        sup2 = _upsample_like(sup2, sup1)

        sup3 = self.side3(hx3d)
        sup3 = _upsample_like(sup3, sup1)

        sup4 = self.side4(hx4d)
        sup4 = _upsample_like(sup4, sup1)

        sup5 = self.side5(hx5d)
        sup5 = _upsample_like(sup5, sup1)

        sup6 = self.side6(hx6)
        sup6 = _upsample_like(sup6, sup1)

        final_fusion_loss = self.outconv(torch.cat((sup1, sup2, sup3, sup4, sup5, sup6), 1))

        return F.torch.sigmoid(final_fusion_loss), F.torch.sigmoid(sup1), F.torch.sigmoid(sup2), F.torch.sigmoid(
            sup3), F.torch.sigmoid(sup4), F.torch.sigmoid(sup5), F.torch.sigmoid(sup6)


class NUSNetSAM(nn.Module):

    def __init__(self, in_ch=3, out_ch=1):
        super(NUSNetSAM, self).__init__()

        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.en1 = NUSUnit7S(in_ch, 32, 64)
        self.en2 = NUSUnit6S(64, 32, 128)
        self.en3 = NUSUnit5S(128, 64, 256)
        self.en4 = NUSUnit4S(256, 128, 512)
        self.en5 = NUSUnitSmallS(512, 256, 512)

        self.en6 = NUSUnitSmallS(512, 256, 512)

        self.de5 = NUSUnitSmallS(512, 256, 512)
        self.de4 = NUSUnit4S(512, 128, 256)
        self.de3 = NUSUnit5S(256, 64, 128)
        self.de2 = NUSUnit6S(128, 32, 64)
        self.de1 = NUSUnit7S(64, 32, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6, out_ch, 1)

    def forward(self, x):
        hx = x

        # encoder
        hx1 = self.en1(hx)
        hx = self.pool(hx1)

        hx2 = self.en2(hx)
        hx = self.pool(hx2)

        hx3 = self.en3(hx)
        hx = self.pool(hx3)

        hx4 = self.en4(hx)
        hx = self.pool(hx4)

        hx5 = self.en5(hx)

        hx6 = self.en6(hx5)

        # decoder
        hx5d = self.de5(torch.add(hx6, hx5))
        hx5up = _upsample_like(hx5d, hx4)

        hx4d = self.de4(torch.add(hx5up, hx4))
        hx4up = _upsample_like(hx4d, hx3)

        hx3d = self.de3(torch.add(hx4up, hx3))
        hx3up = _upsample_like(hx3d, hx2)

        hx2d = self.de2(torch.add(hx3up, hx2))
        hx2up = _upsample_like(hx2d, hx1)

        hx1d = self.de1(hx2up)

        # output
        sup1 = self.side1(hx1d)

        sup2 = self.side2(hx2d)
        sup2 = _upsample_like(sup2, sup1)

        sup3 = self.side3(hx3d)
        sup3 = _upsample_like(sup3, sup1)

        sup4 = self.side4(hx4d)
        sup4 = _upsample_like(sup4, sup1)

        sup5 = self.side5(hx5d)
        sup5 = _upsample_like(sup5, sup1)

        sup6 = self.side6(hx6)
        sup6 = _upsample_like(sup6, sup1)

        final_fusion_loss = self.outconv(torch.cat((sup1, sup2, sup3, sup4, sup5, sup6), 1))

        return F.torch.sigmoid(final_fusion_loss), F.torch.sigmoid(sup1), F.torch.sigmoid(sup2), F.torch.sigmoid(
            sup3), F.torch.sigmoid(sup4), F.torch.sigmoid(sup5), F.torch.sigmoid(sup6)


class NUSNetCAM(nn.Module):

    def __init__(self, in_ch=3, out_ch=1):
        super(NUSNetCAM, self).__init__()

        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.en1 = NUSUnit7C(in_ch, 32, 64)
        self.en2 = NUSUnit6C(64, 32, 128)
        self.en3 = NUSUnit5C(128, 64, 256)
        self.en4 = NUSUnit4C(256, 128, 512)
        self.en5 = NUSUnitSmallC(512, 256, 512)

        self.en6 = NUSUnitSmallC(512, 256, 512)

        self.de5 = NUSUnitSmallC(512, 256, 512)
        self.de4 = NUSUnit4C(512, 128, 256)
        self.de3 = NUSUnit5C(256, 64, 128)
        self.de2 = NUSUnit6C(128, 32, 64)
        self.de1 = NUSUnit7C(64, 32, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6, out_ch, 1)

    def forward(self, x):
        hx = x

        # encoder
        hx1 = self.en1(hx)
        hx = self.pool(hx1)

        hx2 = self.en2(hx)
        hx = self.pool(hx2)

        hx3 = self.en3(hx)
        hx = self.pool(hx3)

        hx4 = self.en4(hx)
        hx = self.pool(hx4)

        hx5 = self.en5(hx)

        hx6 = self.en6(hx5)

        # decoder
        hx5d = self.de5(torch.add(hx6, hx5))
        hx5up = _upsample_like(hx5d, hx4)

        hx4d = self.de4(torch.add(hx5up, hx4))
        hx4up = _upsample_like(hx4d, hx3)

        hx3d = self.de3(torch.add(hx4up, hx3))
        hx3up = _upsample_like(hx3d, hx2)

        hx2d = self.de2(torch.add(hx3up, hx2))
        hx2up = _upsample_like(hx2d, hx1)

        hx1d = self.de1(hx2up)

        # output
        sup1 = self.side1(hx1d)

        sup2 = self.side2(hx2d)
        sup2 = _upsample_like(sup2, sup1)

        sup3 = self.side3(hx3d)
        sup3 = _upsample_like(sup3, sup1)

        sup4 = self.side4(hx4d)
        sup4 = _upsample_like(sup4, sup1)

        sup5 = self.side5(hx5d)
        sup5 = _upsample_like(sup5, sup1)

        sup6 = self.side6(hx6)
        sup6 = _upsample_like(sup6, sup1)

        final_fusion_loss = self.outconv(torch.cat((sup1, sup2, sup3, sup4, sup5, sup6), 1))

        return F.torch.sigmoid(final_fusion_loss), F.torch.sigmoid(sup1), F.torch.sigmoid(sup2), F.torch.sigmoid(
            sup3), F.torch.sigmoid(sup4), F.torch.sigmoid(sup5), F.torch.sigmoid(sup6)


class NUSNet4(nn.Module):

    def __init__(self, in_ch=3, out_ch=1):
        super(NUSNet4, self).__init__()

        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.en1 = NUSUnit4S(in_ch, 32, 64)
        self.en2 = NUSUnit4S(64, 32, 128)
        self.en3 = NUSUnit4S(128, 64, 256)
        self.en4 = NUSUnit4C(256, 128, 512)
        self.en5 = NUSUnit4C(512, 256, 512)

        self.en6 = NUSUnit4C(512, 256, 512)

        self.de5 = NUSUnit4C(512, 256, 512)
        self.de4 = NUSUnit4C(512, 128, 256)
        self.de3 = NUSUnit4S(256, 64, 128)
        self.de2 = NUSUnit4S(128, 32, 64)
        self.de1 = NUSUnit4S(64, 32, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6, out_ch, 1)

    def forward(self, x):
        hx = x

        # encoder
        hx1 = self.en1(hx)
        hx = self.pool(hx1)

        hx2 = self.en2(hx)
        hx = self.pool(hx2)

        hx3 = self.en3(hx)
        hx = self.pool(hx3)

        hx4 = self.en4(hx)
        hx = self.pool(hx4)

        hx5 = self.en5(hx)

        hx6 = self.en6(hx5)

        # decoder
        hx5d = self.de5(torch.add(hx6, hx5))
        hx5up = _upsample_like(hx5d, hx4)

        hx4d = self.de4(torch.add(hx5up, hx4))
        hx4up = _upsample_like(hx4d, hx3)

        hx3d = self.de3(torch.add(hx4up, hx3))
        hx3up = _upsample_like(hx3d, hx2)

        hx2d = self.de2(torch.add(hx3up, hx2))
        hx2up = _upsample_like(hx2d, hx1)

        hx1d = self.de1(hx2up)

        # output
        sup1 = self.side1(hx1d)

        sup2 = self.side2(hx2d)
        sup2 = _upsample_like(sup2, sup1)

        sup3 = self.side3(hx3d)
        sup3 = _upsample_like(sup3, sup1)

        sup4 = self.side4(hx4d)
        sup4 = _upsample_like(sup4, sup1)

        sup5 = self.side5(hx5d)
        sup5 = _upsample_like(sup5, sup1)

        sup6 = self.side6(hx6)
        sup6 = _upsample_like(sup6, sup1)

        final_fusion_loss = self.outconv(torch.cat((sup1, sup2, sup3, sup4, sup5, sup6), 1))

        return F.torch.sigmoid(final_fusion_loss), F.torch.sigmoid(sup1), F.torch.sigmoid(sup2), F.torch.sigmoid(
            sup3), F.torch.sigmoid(sup4), F.torch.sigmoid(sup5), F.torch.sigmoid(sup6)


class NUSNet5(nn.Module):

    def __init__(self, in_ch=3, out_ch=1):
        super(NUSNet5, self).__init__()

        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.en1 = NUSUnit5S(in_ch, 32, 64)
        self.en2 = NUSUnit5S(64, 32, 128)
        self.en3 = NUSUnit5S(128, 64, 256)
        self.en4 = NUSUnit5C(256, 128, 512)
        self.en5 = NUSUnit5C(512, 256, 512)

        self.en6 = NUSUnit5C(512, 256, 512)

        self.de5 = NUSUnit5C(512, 256, 512)
        self.de4 = NUSUnit5C(512, 128, 256)
        self.de3 = NUSUnit5S(256, 64, 128)
        self.de2 = NUSUnit5S(128, 32, 64)
        self.de1 = NUSUnit5S(64, 32, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6, out_ch, 1)

    def forward(self, x):
        hx = x

        # encoder
        hx1 = self.en1(hx)
        hx = self.pool(hx1)

        hx2 = self.en2(hx)
        hx = self.pool(hx2)

        hx3 = self.en3(hx)
        hx = self.pool(hx3)

        hx4 = self.en4(hx)
        hx = self.pool(hx4)

        hx5 = self.en5(hx)

        hx6 = self.en6(hx5)

        # decoder
        hx5d = self.de5(torch.add(hx6, hx5))
        hx5up = _upsample_like(hx5d, hx4)

        hx4d = self.de4(torch.add(hx5up, hx4))
        hx4up = _upsample_like(hx4d, hx3)

        hx3d = self.de3(torch.add(hx4up, hx3))
        hx3up = _upsample_like(hx3d, hx2)

        hx2d = self.de2(torch.add(hx3up, hx2))
        hx2up = _upsample_like(hx2d, hx1)

        hx1d = self.de1(hx2up)

        # output
        sup1 = self.side1(hx1d)

        sup2 = self.side2(hx2d)
        sup2 = _upsample_like(sup2, sup1)

        sup3 = self.side3(hx3d)
        sup3 = _upsample_like(sup3, sup1)

        sup4 = self.side4(hx4d)
        sup4 = _upsample_like(sup4, sup1)

        sup5 = self.side5(hx5d)
        sup5 = _upsample_like(sup5, sup1)

        sup6 = self.side6(hx6)
        sup6 = _upsample_like(sup6, sup1)

        final_fusion_loss = self.outconv(torch.cat((sup1, sup2, sup3, sup4, sup5, sup6), 1))

        return F.torch.sigmoid(final_fusion_loss), F.torch.sigmoid(sup1), F.torch.sigmoid(sup2), F.torch.sigmoid(
            sup3), F.torch.sigmoid(sup4), F.torch.sigmoid(sup5), F.torch.sigmoid(sup6)


class NUSNet6(nn.Module):

    def __init__(self, in_ch=3, out_ch=1):
        super(NUSNet6, self).__init__()

        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.en1 = NUSUnit6S(in_ch, 32, 64)
        self.en2 = NUSUnit6S(64, 32, 128)
        self.en3 = NUSUnit6S(128, 64, 256)
        self.en4 = NUSUnit6C(256, 128, 512)
        self.en5 = NUSUnit6C(512, 256, 512)

        self.en6 = NUSUnit6C(512, 256, 512)

        self.de5 = NUSUnit6C(512, 256, 512)
        self.de4 = NUSUnit6C(512, 128, 256)
        self.de3 = NUSUnit6S(256, 64, 128)
        self.de2 = NUSUnit6S(128, 32, 64)
        self.de1 = NUSUnit6S(64, 32, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6, out_ch, 1)

    def forward(self, x):
        hx = x

        # encoder
        hx1 = self.en1(hx)
        hx = self.pool(hx1)

        hx2 = self.en2(hx)
        hx = self.pool(hx2)

        hx3 = self.en3(hx)
        hx = self.pool(hx3)

        hx4 = self.en4(hx)
        hx = self.pool(hx4)

        hx5 = self.en5(hx)

        hx6 = self.en6(hx5)

        # decoder
        hx5d = self.de5(torch.add(hx6, hx5))
        hx5up = _upsample_like(hx5d, hx4)

        hx4d = self.de4(torch.add(hx5up, hx4))
        hx4up = _upsample_like(hx4d, hx3)

        hx3d = self.de3(torch.add(hx4up, hx3))
        hx3up = _upsample_like(hx3d, hx2)

        hx2d = self.de2(torch.add(hx3up, hx2))
        hx2up = _upsample_like(hx2d, hx1)

        hx1d = self.de1(hx2up)

        # output
        sup1 = self.side1(hx1d)

        sup2 = self.side2(hx2d)
        sup2 = _upsample_like(sup2, sup1)

        sup3 = self.side3(hx3d)
        sup3 = _upsample_like(sup3, sup1)

        sup4 = self.side4(hx4d)
        sup4 = _upsample_like(sup4, sup1)

        sup5 = self.side5(hx5d)
        sup5 = _upsample_like(sup5, sup1)

        sup6 = self.side6(hx6)
        sup6 = _upsample_like(sup6, sup1)

        final_fusion_loss = self.outconv(torch.cat((sup1, sup2, sup3, sup4, sup5, sup6), 1))

        return F.torch.sigmoid(final_fusion_loss), F.torch.sigmoid(sup1), F.torch.sigmoid(sup2), F.torch.sigmoid(
            sup3), F.torch.sigmoid(sup4), F.torch.sigmoid(sup5), F.torch.sigmoid(sup6)


class NUSNet7(nn.Module):

    def __init__(self, in_ch=3, out_ch=1):
        super(NUSNet7, self).__init__()

        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.en1 = NUSUnit7S(in_ch, 32, 64)
        self.en2 = NUSUnit7S(64, 32, 128)
        self.en3 = NUSUnit7S(128, 64, 256)
        self.en4 = NUSUnit7C(256, 128, 512)
        self.en5 = NUSUnit7C(512, 256, 512)

        self.en6 = NUSUnit7C(512, 256, 512)

        self.de5 = NUSUnit7C(512, 256, 512)
        self.de4 = NUSUnit7C(512, 128, 256)
        self.de3 = NUSUnit7S(256, 64, 128)
        self.de2 = NUSUnit7S(128, 32, 64)
        self.de1 = NUSUnit7S(64, 32, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6, out_ch, 1)

    def forward(self, x):
        hx = x

        # encoder
        hx1 = self.en1(hx)
        hx = self.pool(hx1)

        hx2 = self.en2(hx)
        hx = self.pool(hx2)

        hx3 = self.en3(hx)
        hx = self.pool(hx3)

        hx4 = self.en4(hx)
        hx = self.pool(hx4)

        hx5 = self.en5(hx)

        hx6 = self.en6(hx5)

        # decoder
        hx5d = self.de5(torch.add(hx6, hx5))
        hx5up = _upsample_like(hx5d, hx4)

        hx4d = self.de4(torch.add(hx5up, hx4))
        hx4up = _upsample_like(hx4d, hx3)

        hx3d = self.de3(torch.add(hx4up, hx3))
        hx3up = _upsample_like(hx3d, hx2)

        hx2d = self.de2(torch.add(hx3up, hx2))
        hx2up = _upsample_like(hx2d, hx1)

        hx1d = self.de1(hx2up)

        # output
        sup1 = self.side1(hx1d)

        sup2 = self.side2(hx2d)
        sup2 = _upsample_like(sup2, sup1)

        sup3 = self.side3(hx3d)
        sup3 = _upsample_like(sup3, sup1)

        sup4 = self.side4(hx4d)
        sup4 = _upsample_like(sup4, sup1)

        sup5 = self.side5(hx5d)
        sup5 = _upsample_like(sup5, sup1)

        sup6 = self.side6(hx6)
        sup6 = _upsample_like(sup6, sup1)

        final_fusion_loss = self.outconv(torch.cat((sup1, sup2, sup3, sup4, sup5, sup6), 1))

        return F.torch.sigmoid(final_fusion_loss), F.torch.sigmoid(sup1), F.torch.sigmoid(sup2), F.torch.sigmoid(
            sup3), F.torch.sigmoid(sup4), F.torch.sigmoid(sup5), F.torch.sigmoid(sup6)


class NUSNetCBAM(nn.Module):

    def __init__(self, in_ch=3, out_ch=1):
        super(NUSNetCBAM, self).__init__()

        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.en1 = NUSUnit7CS(in_ch, 32, 64)
        self.en2 = NUSUnit6CS(64, 32, 128)
        self.en3 = NUSUnit5CS(128, 64, 256)
        self.en4 = NUSUnit4CS(256, 128, 512)
        self.en5 = NUSUnitSmallCS(512, 256, 512)

        self.en6 = NUSUnitSmallCS(512, 256, 512)

        self.de5 = NUSUnitSmallCS(512, 256, 512)
        self.de4 = NUSUnit4CS(512, 128, 256)
        self.de3 = NUSUnit5CS(256, 64, 128)
        self.de2 = NUSUnit6CS(128, 32, 64)
        self.de1 = NUSUnit7CS(64, 32, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6, out_ch, 1)

    def forward(self, x):
        hx = x

        # encoder
        hx1 = self.en1(hx)
        hx = self.pool(hx1)

        hx2 = self.en2(hx)
        hx = self.pool(hx2)

        hx3 = self.en3(hx)
        hx = self.pool(hx3)

        hx4 = self.en4(hx)
        hx = self.pool(hx4)

        hx5 = self.en5(hx)

        hx6 = self.en6(hx5)

        # decoder
        hx5d = self.de5(torch.add(hx6, hx5))
        hx5up = _upsample_like(hx5d, hx4)

        hx4d = self.de4(torch.add(hx5up, hx4))
        hx4up = _upsample_like(hx4d, hx3)

        hx3d = self.de3(torch.add(hx4up, hx3))
        hx3up = _upsample_like(hx3d, hx2)

        hx2d = self.de2(torch.add(hx3up, hx2))
        hx2up = _upsample_like(hx2d, hx1)

        hx1d = self.de1(hx2up)

        # output
        sup1 = self.side1(hx1d)

        sup2 = self.side2(hx2d)
        sup2 = _upsample_like(sup2, sup1)

        sup3 = self.side3(hx3d)
        sup3 = _upsample_like(sup3, sup1)

        sup4 = self.side4(hx4d)
        sup4 = _upsample_like(sup4, sup1)

        sup5 = self.side5(hx5d)
        sup5 = _upsample_like(sup5, sup1)

        sup6 = self.side6(hx6)
        sup6 = _upsample_like(sup6, sup1)

        final_fusion_loss = self.outconv(torch.cat((sup1, sup2, sup3, sup4, sup5, sup6), 1))

        return F.torch.sigmoid(final_fusion_loss), F.torch.sigmoid(sup1), F.torch.sigmoid(sup2), F.torch.sigmoid(
            sup3), F.torch.sigmoid(sup4), F.torch.sigmoid(sup5), F.torch.sigmoid(sup6)


class NUSNet765CAM4SMALLSAM(nn.Module):

    def __init__(self, in_ch=3, out_ch=1):
        super(NUSNet765CAM4SMALLSAM, self).__init__()

        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.en1 = NUSUnit7C(in_ch, 32, 64)
        self.en2 = NUSUnit6C(64, 32, 128)
        self.en3 = NUSUnit5C(128, 64, 256)
        self.en4 = NUSUnit4S(256, 128, 512)
        self.en5 = NUSUnitSmallS(512, 256, 512)

        self.en6 = NUSUnitSmallS(512, 256, 512)

        self.de5 = NUSUnitSmallS(512, 256, 512)
        self.de4 = NUSUnit4S(512, 128, 256)
        self.de3 = NUSUnit5C(256, 64, 128)
        self.de2 = NUSUnit6C(128, 32, 64)
        self.de1 = NUSUnit7C(64, 32, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6, out_ch, 1)

    def forward(self, x):
        hx = x

        # encoder
        hx1 = self.en1(hx)
        hx = self.pool(hx1)

        hx2 = self.en2(hx)
        hx = self.pool(hx2)

        hx3 = self.en3(hx)
        hx = self.pool(hx3)

        hx4 = self.en4(hx)
        hx = self.pool(hx4)

        hx5 = self.en5(hx)

        hx6 = self.en6(hx5)

        # decoder
        hx5d = self.de5(torch.add(hx6, hx5))
        hx5up = _upsample_like(hx5d, hx4)

        hx4d = self.de4(torch.add(hx5up, hx4))
        hx4up = _upsample_like(hx4d, hx3)

        hx3d = self.de3(torch.add(hx4up, hx3))
        hx3up = _upsample_like(hx3d, hx2)

        hx2d = self.de2(torch.add(hx3up, hx2))
        hx2up = _upsample_like(hx2d, hx1)

        hx1d = self.de1(hx2up)

        # output
        sup1 = self.side1(hx1d)

        sup2 = self.side2(hx2d)
        sup2 = _upsample_like(sup2, sup1)

        sup3 = self.side3(hx3d)
        sup3 = _upsample_like(sup3, sup1)

        sup4 = self.side4(hx4d)
        sup4 = _upsample_like(sup4, sup1)

        sup5 = self.side5(hx5d)
        sup5 = _upsample_like(sup5, sup1)

        sup6 = self.side6(hx6)
        sup6 = _upsample_like(sup6, sup1)

        final_fusion_loss = self.outconv(torch.cat((sup1, sup2, sup3, sup4, sup5, sup6), 1))

        return F.torch.sigmoid(final_fusion_loss), F.torch.sigmoid(sup1), F.torch.sigmoid(sup2), F.torch.sigmoid(
            sup3), F.torch.sigmoid(sup4), F.torch.sigmoid(sup5), F.torch.sigmoid(sup6)


# torchsummary
# model = NUSNetCBAM()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)
# print(summary(model, (3, 320, 320)))

# # torchstat
# model = NUSNet()
# print(stat(model, (3, 320, 320)))
