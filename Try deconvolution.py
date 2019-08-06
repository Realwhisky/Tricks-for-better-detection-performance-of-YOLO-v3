class _router_v3(nn.Module):
    def __init__(self, inp, oup, stride=1, bilinear=True):
        super(_router_v3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.LeakyReLU(0.1, inplace=True),
        )
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        else:
            self.up = nn.ConvTranspose2d(oup, oup, 2, stride=2)   
			# 可尝试反卷积（转置卷积）的方式，效果不错
       # self.conv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)

    def forward(self, x1, x2):
        # prune channel
        x1 = self.conv(x1)
        # up
        x1 = self.up(x1)
        # ideally the following is not needed
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        return torch.cat((x1, x2), dim=1)