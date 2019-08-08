class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
        

### ChannelAtten 通道注意力 ###
### 先在每个通道做 avg & max pool,然后通过一个MLP学习 pool后的 refined values ###

class ChannelAtten(nn.Module):           
    def __init__(self, teacher_channels, reduction_ratio=16, pool_type = 'avg'):
        super(ChannelAtten, self).__init__()
        self.teacher_channels = teacher_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(teacher_channels, teacher_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(teacher_channels // reduction_ratio, teacher_channels)
        )
        self.pool_type = pool_type

    def forward(self, student, teacher):
        avg_pool = F.avg_pool2d(teacher, (teacher.size(2), teacher.size(3)), stride=(teacher.size(2), teacher.size(3)))
        channel_atten = self.mlp(avg_pool)
        scale = torch.sigmoid(channel_atten).unsqueeze(2).unsqueeze(3).expand_as(student)

        return student * scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
### SpatialAtten 空间注意力（在ChannelAtten后的基础上做的） ###
### 先Channel-refined features做 avg & max pool,然后通过一个7*7的卷积核学习 pool后的 values ###

class SpatialAtten(nn.Module):                  # 
    def __init__(self):
        super(SpatialAtten, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(in_planes=2, out_planes=1, kernel_size=kernel_size, stride=1,
                                 padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, student, teacher):
        x_compress = self.compress(teacher)  # 压缩 teacher feature 至2 channels with max and avg pool
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)

        return student * scale


class Atten(nn.Module):  # 定义 dual self attention network ___ channel_atten with spatial_atten
    def __init__(self, teacher_channels, student_channels):
        super(Atten, self).__init__()
        self.ChannelAtten = ChannelAtten(teacher_channels)
        self.SpatialAtten = SpatialAtten()

    def forward(self, x):
        x = self.ChannelAtten(x, x)
        output = self.SpatialAtten(x, x)

        return output
