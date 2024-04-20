import torch
import torch.nn as nn
import torch.nn.functional as F
class invertedBlock(nn.Module):
    def __init__(self, in_channel, out_channel,ratio=2):
        super(invertedBlock, self).__init__()
        internal_channel = in_channel * ratio
        self.relu = nn.GELU()
        ## 7*7卷积，并行3*3卷积
        self.conv1 = nn.Conv2d(internal_channel, internal_channel, 7, 1, 3, groups=in_channel,bias=False)

        self.convFFN = ConvFFN(in_channels=in_channel, out_channels=in_channel)
        self.layer_norm = nn.LayerNorm(in_channel)
        self.pw1 = nn.Conv2d(in_channels=in_channel, out_channels=internal_channel, kernel_size=1, stride=1,
                             padding=0, groups=1,bias=False)
        self.pw2 = nn.Conv2d(in_channels=internal_channel, out_channels=in_channel, kernel_size=1, stride=1,
                             padding=0, groups=1,bias=False)


    def hifi(self,x):

        x1=self.pw1(x)
        x1=self.relu(x1)
        x1=self.conv1(x1)
        x1=self.relu(x1)
        x1=self.pw2(x1)
        x1=self.relu(x1)
        # x2 = self.conv2(x)
        x3 = x1+x

        x3 = x3.permute(0, 2, 3, 1).contiguous()
        x3 = self.layer_norm(x3)
        x3 = x3.permute(0, 3, 1, 2).contiguous()
        x4 = self.convFFN(x3)

        return x4

    def forward(self, x):
        return self.hifi(x)+x
class ConvFFN(nn.Module):

    def __init__(self, in_channels, out_channels, expend_ratio=4):
        super().__init__()

        internal_channels = in_channels * expend_ratio
        self.pw1 = nn.Conv2d(in_channels=in_channels, out_channels=internal_channels, kernel_size=1, stride=1,
                             padding=0, groups=1,bias=False)
        self.pw2 = nn.Conv2d(in_channels=internal_channels, out_channels=out_channels, kernel_size=1, stride=1,
                             padding=0, groups=1,bias=False)
        self.nonlinear = nn.GELU()

    def forward(self, x):
        x1 = self.pw1(x)
        x2 = self.nonlinear(x1)
        x3 = self.pw2(x2)
        x4 = self.nonlinear(x3)
        return x4 + x

class mixblock(nn.Module):
    def __init__(self, n_feats):
        super(mixblock, self).__init__()
        self.conv1=nn.Sequential(nn.Conv2d(n_feats,n_feats,3,1,1,bias=False),nn.GELU())
        self.conv2=nn.Sequential(nn.Conv2d(n_feats,n_feats,3,1,1,bias=False),nn.GELU(),nn.Conv2d(n_feats,n_feats,3,1,1,bias=False),nn.GELU(),nn.Conv2d(n_feats,n_feats,3,1,1,bias=False),nn.GELU())
        self.alpha=nn.Parameter(torch.ones(1))
        self.beta=nn.Parameter(torch.ones(1))
    def forward(self,x):
        return self.alpha*self.conv1(x)+self.beta*self.conv2(x)
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
class Downupblock(nn.Module):
    def __init__(self, n_feats):
        super(Downupblock, self).__init__()
        self.encoder = mixblock(n_feats)
        self.decoder_high = mixblock(n_feats)  # nn.Sequential(one_module(n_feats),

        self.decoder_low = nn.Sequential(mixblock(n_feats), mixblock(n_feats), mixblock(n_feats))
        self.alise = nn.Conv2d(n_feats,n_feats,1,1,0,bias=False)  # one_module(n_feats)
        self.alise2 = nn.Conv2d(n_feats*2,n_feats,3,1,1,bias=False)  # one_module(n_feats)
        self.down = nn.AvgPool2d(kernel_size=2)
        self.att = CALayer(n_feats)
        self.raw_alpha=nn.Parameter(torch.ones(1))

        self.raw_alpha.data.fill_(0)
        self.ega=selfAttention(n_feats, n_feats)

    def forward(self, x,raw):
        x1 = self.encoder(x)
        x2 = self.down(x1)
        high = x1 - F.interpolate(x2, size=x.size()[-2:], mode='bilinear', align_corners=True)

        high=high+self.ega(high,high)*self.raw_alpha
        x2=self.decoder_low(x2)
        x3 = x2
        # x3 = self.decoder_low(x2)
        high1 = self.decoder_high(high)
        x4 = F.interpolate(x3, size=x.size()[-2:], mode='bilinear', align_corners=True)
        return self.alise(self.att(self.alise2(torch.cat([x4, high1], dim=1)))) + x
class Updownblock(nn.Module):
    def __init__(self, n_feats):
        super(Updownblock, self).__init__()
        self.encoder = mixblock(n_feats)
        self.decoder_high = mixblock(n_feats)  # nn.Sequential(one_module(n_feats),
        #                     one_module(n_feats),
        #                     one_module(n_feats))
        self.decoder_low = nn.Sequential(mixblock(n_feats), mixblock(n_feats), mixblock(n_feats))

        self.alise = nn.Conv2d(n_feats,n_feats,1,1,0,bias=False)  # one_module(n_feats)
        self.alise2 = nn.Conv2d(n_feats*2,n_feats,3,1,1,bias=False)  # one_module(n_feats)
        self.down = nn.AvgPool2d(kernel_size=2)
        self.att = CALayer(n_feats)
        self.raw_alpha=nn.Parameter(torch.ones(1))
        # fill 0
        self.raw_alpha.data.fill_(0)
        self.ega=selfAttention(n_feats, n_feats)

    def forward(self, x,raw):
        x1 = self.encoder(x)
        x2 = self.down(x1)
        high = x1 - F.interpolate(x2, size=x.size()[-2:], mode='bilinear', align_corners=True)
        high=high+self.ega(high,high)*self.raw_alpha
        x2=self.decoder_low(x2)
        x3 = x2
        high1 = self.decoder_high(high)
        x4 = F.interpolate(x3, size=x.size()[-2:], mode='bilinear', align_corners=True)
        return self.alise(self.att(self.alise2(torch.cat([x4, high1], dim=1)))) + x
class basic_block(nn.Module):
    ## 双并行分支，通道分支和空间分支
    def __init__(self, in_channel, out_channel, depth,ratio=1):
        super(basic_block, self).__init__()




        # 个数为depth个

        self.rep1 = nn.Sequential(*[invertedBlock(in_channel=in_channel, out_channel=in_channel,ratio=ratio) for i in range(depth)])


        self.relu=nn.GELU()
        # 一部分做3个3*3卷积，一部分做1个

        self.updown=Updownblock(in_channel)
        self.downup=Downupblock(in_channel)
    def forward(self, x,raw=None):


        x1 = self.rep1(x)


        x1=self.updown(x1,raw)
        x1=self.downup(x1,raw)
        return x1+x

import torchvision
class VGG_aware(nn.Module):
    def __init__(self,outFeature):
        super(VGG_aware, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())

        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)


    def forward(self, x):
        return self.blocks[0](x)

import torch.nn.functional as f
class selfAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(selfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.scale = 1.0 / (out_channels ** 0.5)

    def forward(self, feature, feature_map):
        query = self.query_conv(feature)
        key = self.key_conv(feature)
        value = self.value_conv(feature)
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores * self.scale

        attention_weights = f.softmax(attention_scores, dim=-1)

        attended_values = torch.matmul(attention_weights, value)

        output_feature_map = (feature_map + attended_values)

        return output_feature_map