import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import math
from typing import Type
from timm.models.efficientnet_blocks import SqueezeExcite, DepthwiseSeparableConv



class MBConv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()

        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding,
                               stride=stride, dilation=dilation, groups=dim_in)
        self.norm_layer = nn.GroupNorm(4, dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))


class ResECA(nn.Module):
    def __init__(self, in_c, out_c):
        super(ResECA, self).__init__()
        self.conv1 = MBConv(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = MBConv(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.conv3 = MBConv(in_c, out_c, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_c)

        self.eca = ECALayer(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)

        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x3 = self.eca(x3)

        x4 = x2 + x3
        x4 = self.relu(x4)
        return x4

class ECALayer(nn.Module):
    def __init__(self, inp):
        super(ECALayer, self).__init__()
        gamma = 2
        b = 1
        t = int(abs((math.log2(inp) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.conv(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(out)
        return x * out.expand_as(x)




# 8,16,24,32,48,64
# GFLOPs :0.06, Params : 0.06
# 8,16,32,48,64,96
# GFLOPs :0.09, Params : 0.12
# 16,24,32,48,64,128
# GFLOPs :0.11, Params : 0.15

class DHR_Net(nn.Module):
    def  __init__(self, n_classes=1, n_channels=3, c_list=[8,16,24,32,48,64], bridge=True, gt_ds=True):
        super().__init__()
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.bridge = bridge
        self.gt_ds = gt_ds

        self.encoder1 = nn.Sequential(
            nn.Conv2d(n_channels, c_list[0], 3, stride=1, padding=1),
        )

        self.encoder2 =nn.Sequential(
            MBConv(c_list[0], c_list[1], 3, stride=1, padding=1),
        )

        self.encoder3 = nn.Sequential(
            DSDCBlock(c_list[1], c_list[2]),

        )

        self.encoder4 = nn.Sequential(
            DSDCBlock(c_list[2], c_list[3]),
        )


        self.encoder5 = nn.Sequential(
            MBConv(c_list[3], c_list[4], 3, stride=1, padding=1),
            HA(c_list[4]),
        )


        self.encoder6 = nn.Sequential(
            MBConv(c_list[4], c_list[5], 3, stride=1, padding=1),
            HA(c_list[5]),
        )

        # Bottleneck layers
        self.ca = ResECA(c_list[5],c_list[5])


        self.decoder1 = nn.Sequential(
            HA(c_list[5]),
            MBConv(c_list[5], c_list[4], 3, stride=1, padding=1),
        )

        self.decoder2 = nn.Sequential(
            HA(c_list[4]),
            MBConv(c_list[4], c_list[3],3, stride=1, padding=1),
        )

        self.decoder3 = nn.Sequential(
            DSDCBlock(c_list[3], c_list[2]),
        )

        self.decoder4 = nn.Sequential(
            DSDCBlock(c_list[2], c_list[1]),
        )


        self.decoder5 = nn.Sequential(
            MBConv(c_list[1], c_list[0], 3, stride=1, padding=1),
        )

        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])

        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0], n_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        t1 = out # b, c0, H/2, W/2

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        t2 = out # b, c1, H/4, W/4 

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        # out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3_1(self.encoder3(out))),2,2))
        t3 = out # b, c2, H/8, W/8

        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
        # out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4_1(self.encoder4(out))),2,2))
        t4 = out # b, c3, H/16, W/16

        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        # out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5_1(self.encoder5(out))),2,2))
        t5 = out # b, c4, H/32, W/32

        out = F.gelu(self.encoder6(out))  # b, c5, H/32, W/32
        # out = F.gelu(self.encoder6_1(self.encoder6(out))) # b, c5, H/32, W/32
        # t6 = out

        out_bottleneck = self.ca(out)

        out5 = F.gelu(self.dbn1(self.decoder1(out_bottleneck)))  # b, c4, H/32, W/32

        out5 = torch.add(out5, t5) # b, c4, H/32, W/32

        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)), scale_factor=(2, 2), mode='bilinear',align_corners=True))  # b, c3, H/16, W/16

        out4 = torch.add(out4, t4) # b, c3, H/16, W/16

        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)), scale_factor=(2, 2), mode='bilinear',align_corners=True))  # b, c2, H/8, W/8

        out3 = torch.add(out3, t3) # b, c2, H/8, W/8

        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)), scale_factor=(2, 2), mode='bilinear',align_corners=True))  # b, c1, H/4, W/4


        out2 = torch.add(out2, t2) # b, c1, H/4, W/4

        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c0, H/2, W/2

        out1 = torch.add(out1, t1) # b, c0, H/2, W/2
        out0 = F.interpolate(self.final(out1),scale_factor=(2,2),mode ='bilinear',align_corners=True) # b, num_class, H, W
        return torch.sigmoid(out0)


def make_divisible(v, divisor=8, min_value=None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class DSBlock(nn.Module):
    def __init__(self, in_chs, ds_ratio):
        super(DSBlock, self).__init__()
        out_chs = make_divisible(in_chs * ds_ratio)
        self.conv_reduce = nn.Conv2d(in_chs, out_chs, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv_expand = nn.Conv2d(out_chs, in_chs, 1, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x_ = x.mean((2, 3), keepdim=True)
        x_ = self.conv_reduce(x_)
        x_ = self.relu(x_)
        x_ = self.conv_expand(x_)
        return x * self.act(x_)

class DCConv(nn.Module):
    def __init__(self, channels):
        super(DCConv, self).__init__()
        self.conv3 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.relu1 = nn.ReLU(inplace=True)
    def forward(self, x):
        x3 = self.conv3(x)
        x3 = self.relu3(x3)
        x1 = self.conv1(x)
        x1 = self.relu1(x1)
        out = x3 + x1
        return out

class DSDCBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            drop_path: float = 0.,
            debug=False
    ) -> None:
        super(DSDCBlock, self).__init__()
        self.debug = debug
        self.drop_path_rate: float = drop_path
        if act_layer == nn.GELU:
            act_layer = _gelu_ignore_parameters
        self.main_path = nn.Sequential(
            norm_layer(in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1,1)),
            DepthwiseSeparableConv(in_chs=in_channels, out_chs=out_channels, stride=1,
                                   act_layer=act_layer, norm_layer=norm_layer, drop_path_rate=drop_path),
            DSBlock(in_chs=out_channels, ds_ratio=0.25),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1,1)),
        )
        self.dc = DCConv(out_channels)
        # Make skip path
        self.skip_path = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,1)) \
                            if (in_channels != out_channels) else nn.Identity()
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.main_path(input)
        if self.drop_path_rate > 0.:
            output = drop_path(output, self.drop_path_rate, self.training)
        output = output + self.skip_path(input)
        output = self.dc(output)
        return output

def _gelu_ignore_parameters(*args, **kwargs) -> nn.Module:
        activation = nn.GELU()
        return activation

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class CA(nn.Module):
    def __init__(self, Channel_nums):
        super(CA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.alpha = nn.Parameter(data=torch.FloatTensor([0.5]), requires_grad=True)
        self.beta = nn.Parameter(data=torch.FloatTensor([0.5]), requires_grad=True)
        self.gamma = 2
        self.b = 1
        self.k = self.get_kernel_num(Channel_nums)
        self.conv1d = nn.Conv1d(kernel_size=self.k, in_channels=1, out_channels=1, padding=self.k // 2)
        self.sigmoid = nn.Sigmoid()
    def get_kernel_num(self, C):
        t = math.log2(C) / self.gamma + self.b / self.gamma
        floor = math.floor(t)
        k = floor + (1 - floor % 2)
        return k
    def forward(self, x):
        F_avg = self.avg_pool(x)
        F_max = self.max_pool(x)
        F_add = 0.5 * (F_avg + F_max) + self.alpha * F_avg + self.beta * F_max
        F_add_ = F_add.squeeze(-1).permute(0, 2, 1)
        F_add_ = self.conv1d(F_add_).permute(0, 2, 1).unsqueeze(-1)
        out = self.sigmoid(F_add_)
        return out


class SA(nn.Module):
    def __init__(self, Channel_num):
        super(SA, self).__init__()
        self.channel = Channel_num
        self.delta = 0.6  # split rate
        self.C_relevant = self.get_relevant_channelNum(Channel_num)
        self.C_irelevant = Channel_num - self.C_relevant
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.norm_active = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Sigmoid()
        )
    def get_relevant_channelNum(self, C):
        t = self.delta * C
        floor = math.floor(t)
        C_r = floor + floor % 2
        return C_r
    def get_r_ir_channels(self, C_r, M):
        _, topk = torch.topk(M, dim=1, k=C_r)
        relevant_channels = torch.zeros_like(M)
        irelevant_channels = torch.ones_like(M)
        relevant_channels = relevant_channels.scatter(1, topk, 1)
        irelevant_channels = irelevant_channels.scatter(1, topk, 0)
        return relevant_channels, irelevant_channels
    def get_features(self, r_channels, ir_channels, channel_refined_feature):
        relevant_features = r_channels * channel_refined_feature
        irelevant_features = ir_channels * channel_refined_feature
        return relevant_features, irelevant_features
    def forward(self, x, M):
        relevant_channels, irelevant_channels = self.get_r_ir_channels(self.C_relevant, M)
        relevant_features, irelevant_features = self.get_features(relevant_channels, irelevant_channels, x)
        r_AvgPool = torch.mean(relevant_features, dim=1, keepdim=True) * (self.channel / self.C_relevant)
        r_MaxPool, _ = torch.max(relevant_features, dim=1, keepdim=True)
        ir_AvgPool = torch.mean(irelevant_features, dim=1, keepdim=True) * (self.channel / self.C_irelevant)
        ir_MaxPool, _ = torch.max(irelevant_features, dim=1, keepdim=True)
        r_x = torch.cat([r_AvgPool, r_MaxPool], dim=1)
        ir_x = torch.cat([ir_AvgPool, ir_MaxPool], dim=1)
        A_S1 = self.norm_active(self.conv(r_x))
        A_S2 = self.norm_active(self.conv(ir_x))
        F1 = relevant_features * A_S1
        F2 = irelevant_features * A_S2
        refined_feature = F1 + F2
        return refined_feature

class HA(nn.Module):
    def __init__(self, n_channels):
        super(HA, self).__init__()
        self.channel = n_channels
        self.CA = CA(self.channel)
        self.SA = SA(self.channel)
        self.relu = nn.ReLU()
    def forward(self, x):
        residual = x
        channel_attention_map = self.CA(x)
        channel_refined_feature = channel_attention_map * x
        final_refined_feature = self.SA(channel_refined_feature, channel_attention_map)
        out = self.relu(final_refined_feature + residual)
        return out


# from thop import profile
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = DHR_Net().to(device)
# input = torch.randn(1, 3, 224, 224).to(device)
# flops, params = profile(model, inputs=(input, ))
# print("GFLOPs :{:.2f}, Params : {:.2f}".format(flops/1e9,params/1e6)) #flops单位G，para单位M
# # GFLOPs :0.06, Params : 0.06




