####################################################################################################
# DGL: Device Generic Latency model for Neural Architecture Search
####################################################################################################
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

def Hswish(x,inplace=True):
    return x * F.relu6(x + 3., inplace=inplace) / 6.

def Hsigmoid(x,inplace=True):
    return F.relu6(x + 3., inplace=inplace) / 6.


# Squeeze-And-Excite 
class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y=self.avg_pool(x).view(b, c)
        y=self.se(y)
        y = Hsigmoid(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Bottleneck(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,exp,stride,se='True',nl='HS'):
        super(Bottleneck, self).__init__()
        exp_channels=exp*in_channels
        #self.exp_channels=exp_channels
        padding = (kernel_size - 1) // 2
        if nl == 'RE':
            self.nlin_layer = F.relu6
        elif nl == 'HS':
            self.nlin_layer = Hswish
        self.stride=stride
        if se:
            self.se=SEModule(exp_channels)
        else:
            self.se=None
        
        self.conv1=nn.Conv2d(in_channels,exp_channels,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn1 = nn.BatchNorm2d(exp_channels)
        self.conv2=nn.Conv2d(exp_channels,exp_channels,kernel_size=kernel_size,stride=stride,
                             padding=padding,groups=exp_channels,bias=False)
        self.bn2=nn.BatchNorm2d(exp_channels)
        self.conv3=nn.Conv2d(exp_channels,out_channels,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn3=nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride == 1 and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self,x):
        out=self.nlin_layer(self.bn1(self.conv1(x)))
        if self.se is not None:
            out=self.bn2(self.conv2(out))
            out=self.nlin_layer(self.se(out))
        else:
            out = self.nlin_layer(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class FBNetV3(nn.Module):
    # (out_channels,kernel_size,exp_channels,stride,se,nl)
    cfg1=[ 
         (1,False,'HS'),
         (2,False,'HS'),
         (2,True,'HS'),
         (2,False,'HS'),
         (1,True,'HS'),
         (2,True,'HS'),
         (1,True,'HS')]
    def __init__(self,c1,cfg,num_classes=1000):
        super(FBNetV3,self).__init__()
        self.c1 = c1
        self.cfg = cfg
        last=cfg[6,0]
        self.conv1=nn.Conv2d(3,c1,3,2,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(c1)
        self.layers = self._make_layers(in_channels=c1)
        self.conv2=nn.Conv2d(last,last*6,1,stride=1,bias=False)
        self.bn2=nn.BatchNorm2d(last*6)
        self.conv3=nn.Conv2d(last*6,1984,1,1,padding=0,bias=True)
        #self.conv4=nn.Conv2d(1280,num_classes,1,stride=1,padding=0,bias=True)
        self.linear = nn.Linear(1984, num_classes)

    def _make_layers(self,in_channels):
        #cfg2=[(i[0]+i[1]) for i in zip(self.cfg, self.cfg1)]
        layers=[]
        #for out_channels,kernel_size,exp_channels,blocknumber,strides,se,nl in cfg2:
        for i in range(7):
            out_channels=self.cfg[i,0]
            kernel_size=self.cfg[i,1]
            exp1=self.cfg[i,2]
            exp2 =self.cfg[i,3]
            blocknumber=self.cfg[i,4]
            strides=self.cfg1[i][0]
            se=self.cfg1[i][1]
            nl=self.cfg1[i][2]
            for j in range(blocknumber):
                stride=strides if j==0 else 1
                exp=exp1 if j==0 else exp2
                layers.append(
                    Bottleneck(in_channels,out_channels,kernel_size,exp,stride,se,nl)
                 )
                in_channels=out_channels
        return nn.Sequential(*layers)

    def forward(self,x):
        out=Hswish(self.bn1(self.conv1(x)))
        out=self.layers(out)
        out=Hswish(self.bn2(self.conv2(out)))
        out=F.avg_pool2d(out,7)
        out=Hswish(self.conv3(out))
        out=out.view(out.size(0), -1)
        out=self.linear(out)
        #out = self.conv4(out)
       #a, b = out.size(0), out.size(1)
        #out = out.view(a, b)
        
        return out





#Using simple MLP as the network parameters predictor.
class MLP(nn.Module):
    def __init__(self,search_space):
        super(MLP, self).__init__()
        self.search_space=search_space
        if self.search_space=='nasbench201':
            self.fc1 = nn.Linear(6,512)
        if self.search_space=='fbnetv3':
            self.fc1 = nn.Linear(28,512)
        self.tan=torch.nn.Tanh()
        self.fc2 = nn.Linear(512,2048)
        self.fc3 = nn.Linear(2048,2048)
        self.fc31 = nn.Linear(2048,2048)
        self.fc4 = nn.Linear(2048,256)
        self.fc5 = nn.Linear(256,28)
        self.fc6 = nn.Linear(28,1)
   
    def forward(self, x):
        x = self.fc1(x)
        x = self.tan(x) 
        x = self.fc2(x)
        x = self.tan(x)
        x = self.fc3(x)
        x = self.tan(x)
        x = self.fc31(x)
        x = self.tan(x)
        x = self.fc4(x)
        x = self.tan(x)
        x = self.fc5(x)
        x = self.tan(x)      
        x = self.fc6(x)
        return x

class Latencytrain(nn.Module):
    def __init__(self):
        super(Latencytrain, self).__init__()
        self.fc1 = nn.Linear(28,512)
        self.tan=torch.nn.Tanh()
        self.fc2 = nn.Linear(512,2048)
        self.fc3 = nn.Linear(2048,2048)
        self.fc31 = nn.Linear(2048,2048)
        self.fc4 = nn.Linear(2048,256)
        self.fc5 = nn.Linear(256,28)
        self.fc6 = nn.Linear(28,2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.tan(x) 
        x = self.fc2(x)
        x = self.tan(x)
        x = self.fc3(x)
        x = self.tan(x)
        x = self.fc31(x)
        x = self.tan(x)     
        x = self.fc4(x)
        x = self.tan(x)
        x = self.fc5(x)
        x = self.tan(x)      
        x = self.fc6(x)
        return x

class Acctrain(nn.Module):
    def __init__(self):
        super(Acctrain, self).__init__()
        self.fc1 = nn.Linear(28,512)
        self.tan=torch.nn.Tanh()
        self.fc2 = nn.Linear(512,2048)
        self.fc3 = nn.Linear(2048,2048)
        self.fc31 = nn.Linear(2048,2048)
        self.fc4 = nn.Linear(2048,256)
        self.fc5 = nn.Linear(256,28)
        self.fc6 = nn.Linear(28,2)
          
    def forward(self, x):
        x = self.fc1(x)
        x = self.tan(x) 
        x = self.fc2(x)
        x = self.tan(x)
        x = self.fc3(x)
        x = self.tan(x)
        x = self.fc31(x)
        x = self.tan(x)
        x = self.fc4(x)
        x = self.tan(x)
        x = self.fc5(x)
        x = self.tan(x)      
        x = self.fc6(x)
        return x

class ReLUConvBN(nn.Module):
    def __init__(
        self,
        C_in,
        C_out,
        kernel_size,
        stride,
        padding,
        dilation,
        affine,
        track_running_stats=True,
    ):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_out,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=not affine,
            ),
            nn.BatchNorm2d(
                C_out, affine=affine, track_running_stats=track_running_stats
            ),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):
    def __init__(
        self,
        C_in,
        C_out,
        kernel_size,
        stride,
        padding,
        dilation,
        affine,
        track_running_stats=True,
    ):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=C_in,
                bias=False,
            ),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=not affine),
            nn.BatchNorm2d(
                C_out, affine=affine, track_running_stats=track_running_stats
            ),
        )

    def forward(self, x):
        return self.op(x)


class DualSepConv(nn.Module):
    def __init__(
        self,
        C_in,
        C_out,
        kernel_size,
        stride,
        padding,
        dilation,
        affine,
        track_running_stats=True,
    ):
        super(DualSepConv, self).__init__()
        self.op_a = SepConv(
            C_in,
            C_in,
            kernel_size,
            stride,
            padding,
            dilation,
            affine,
            track_running_stats,
        )
        self.op_b = SepConv(
            C_in, C_out, kernel_size, 1, padding, dilation, affine, track_running_stats
        )

    def forward(self, x):
        x = self.op_a(x)
        x = self.op_b(x)
        return x


class ResNetBasicblock(nn.Module):
    def __init__(self, inplanes, planes, stride, affine=True, track_running_stats=True):
        super(ResNetBasicblock, self).__init__()
        assert stride == 1 or stride == 2, "invalid stride {:}".format(stride)
        self.conv_a = ReLUConvBN(
            inplanes, planes, 3, stride, 1, 1, affine, track_running_stats
        )
        self.conv_b = ReLUConvBN(
            planes, planes, 3, 1, 1, 1, affine, track_running_stats
        )
        if stride == 2:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(
                    inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False
                ),
            )
        elif inplanes != planes:
            self.downsample = ReLUConvBN(
                inplanes, planes, 1, 1, 0, 1, affine, track_running_stats
            )
        else:
            self.downsample = None
        self.in_dim = inplanes
        self.out_dim = planes
        self.stride = stride
        self.num_conv = 2

    def extra_repr(self):
        string = "{name}(inC={in_dim}, outC={out_dim}, stride={stride})".format(
            name=self.__class__.__name__, **self.__dict__
        )
        return string

    def forward(self, inputs):

        basicblock = self.conv_a(inputs)
        basicblock = self.conv_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        return residual + basicblock


class POOLING(nn.Module):
    def __init__(
        self, C_in, C_out, stride, mode, affine=True, track_running_stats=True
    ):
        super(POOLING, self).__init__()
        if C_in == C_out:
            self.preprocess = None
        else:
            self.preprocess = ReLUConvBN(
                C_in, C_out, 1, 1, 0, 1, affine, track_running_stats
            )
        if mode == "avg":
            self.op = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
        elif mode == "max":
            self.op = nn.MaxPool2d(3, stride=stride, padding=1)
        else:
            raise ValueError("Invalid mode={:} in POOLING".format(mode))

    def forward(self, inputs):
        if self.preprocess:
            x = self.preprocess(inputs)
        else:
            x = inputs
        return self.op(x)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    def __init__(self, C_in, C_out, stride):
        super(Zero, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        self.is_zero = True

    def forward(self, x):
        if self.C_in == self.C_out:
            if self.stride == 1:
                return x.mul(0.0)
            else:
                return x[:, :, :: self.stride, :: self.stride].mul(0.0)
        else:
            shape = list(x.shape)
            shape[1] = self.C_out
            zeros = x.new_zeros(shape, dtype=x.dtype, device=x.device)
            return zeros

    def extra_repr(self):
        return "C_in={C_in}, C_out={C_out}, stride={stride}".format(**self.__dict__)


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, stride, affine, track_running_stats):
        super(FactorizedReduce, self).__init__()
        self.stride = stride
        self.C_in = C_in
        self.C_out = C_out
        self.relu = nn.ReLU(inplace=False)
        if stride == 2:
            # assert C_out % 2 == 0, 'C_out : {:}'.format(C_out)
            C_outs = [C_out // 2, C_out - C_out // 2]
            self.convs = nn.ModuleList()
            for i in range(2):
                self.convs.append(
                    nn.Conv2d(
                        C_in, C_outs[i], 1, stride=stride, padding=0, bias=not affine
                    )
                )
            self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)
        elif stride == 1:
            self.conv = nn.Conv2d(
                C_in, C_out, 1, stride=stride, padding=0, bias=not affine
            )
        else:
            raise ValueError("Invalid stride : {:}".format(stride))
        self.bn = nn.BatchNorm2d(
            C_out, affine=affine, track_running_stats=track_running_stats
        )

    def forward(self, x):
        if self.stride == 2:
            x = self.relu(x)
            y = self.pad(x)
            out = torch.cat([self.convs[0](x), self.convs[1](y[:, :, 1:, 1:])], dim=1)
        else:
            out = self.conv(x)
        out = self.bn(out)
        return out

    def extra_repr(self):
        return "C_in={C_in}, C_out={C_out}, stride={stride}".format(**self.__dict__)


# Auto-ReID: Searching for a Part-Aware ConvNet for Person Re-Identification, ICCV 2019
class PartAwareOp(nn.Module):
    def __init__(self, C_in, C_out, stride, part=4):
        super().__init__()
        self.part = 4
        self.hidden = C_in // 3
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.local_conv_list = nn.ModuleList()
        for i in range(self.part):
            self.local_conv_list.append(
                nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(C_in, self.hidden, 1),
                    nn.BatchNorm2d(self.hidden, affine=True),
                )
            )
        self.W_K = nn.Linear(self.hidden, self.hidden)
        self.W_Q = nn.Linear(self.hidden, self.hidden)

        if stride == 2:
            self.last = FactorizedReduce(C_in + self.hidden, C_out, 2)
        elif stride == 1:
            self.last = FactorizedReduce(C_in + self.hidden, C_out, 1)
        else:
            raise ValueError("Invalid Stride : {:}".format(stride))

    def forward(self, x):
        batch, C, H, W = x.size()
        assert H >= self.part, "input size too small : {:} vs {:}".format(
            x.shape, self.part
        )
        IHs = [0]
        for i in range(self.part):
            IHs.append(min(H, int((i + 1) * (float(H) / self.part))))
        local_feat_list = []
        for i in range(self.part):
            feature = x[:, :, IHs[i] : IHs[i + 1], :]
            xfeax = self.avg_pool(feature)
            xfea = self.local_conv_list[i](xfeax)
            local_feat_list.append(xfea)
        part_feature = torch.cat(local_feat_list, dim=2).view(batch, -1, self.part)
        part_feature = part_feature.transpose(1, 2).contiguous()
        part_K = self.W_K(part_feature)
        part_Q = self.W_Q(part_feature).transpose(1, 2).contiguous()
        weight_att = torch.bmm(part_K, part_Q)
        attention = torch.softmax(weight_att, dim=2)
        aggreateF = torch.bmm(attention, part_feature).transpose(1, 2).contiguous()
        features = []
        for i in range(self.part):
            feature = aggreateF[:, :, i : i + 1].expand(
                batch, self.hidden, IHs[i + 1] - IHs[i]
            )
            feature = feature.view(batch, self.hidden, IHs[i + 1] - IHs[i], 1)
            features.append(feature)
        features = torch.cat(features, dim=2).expand(batch, self.hidden, H, W)
        final_fea = torch.cat((x, features), dim=1)
        outputs = self.last(final_fea)
        return outputs


def drop_path(x, drop_prob):
    if drop_prob > 0.0:
        keep_prob = 1.0 - drop_prob
        mask = x.new_zeros(x.size(0), 1, 1, 1)
        mask = mask.bernoulli_(keep_prob)
        x = torch.div(x, keep_prob)
        x.mul_(mask)
    return x


    
    
    
    
    
    
    
    
    

    
OPS = {
    "none": lambda C_in, C_out, stride, affine, track_running_stats: Zero(
        C_in, C_out, stride
    ),
    "avg_pool_3x3": lambda C_in, C_out, stride, affine, track_running_stats: POOLING(
        C_in, C_out, stride, "avg", affine, track_running_stats
    ),
    "max_pool_3x3": lambda C_in, C_out, stride, affine, track_running_stats: POOLING(
        C_in, C_out, stride, "max", affine, track_running_stats
    ),
    "nor_conv_7x7": lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(
        C_in,
        C_out,
        (7, 7),
        (stride, stride),
        (3, 3),
        (1, 1),
        affine,
        track_running_stats,
    ),
    "nor_conv_3x3": lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(
        C_in,
        C_out,
        (3, 3),
        (stride, stride),
        (1, 1),
        (1, 1),
        affine,
        track_running_stats,
    ),
    "nor_conv_1x1": lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(
        C_in,
        C_out,
        (1, 1),
        (stride, stride),
        (0, 0),
        (1, 1),
        affine,
        track_running_stats,
    ),
    "dua_sepc_3x3": lambda C_in, C_out, stride, affine, track_running_stats: DualSepConv(
        C_in,
        C_out,
        (3, 3),
        (stride, stride),
        (1, 1),
        (1, 1),
        affine,
        track_running_stats,
    ),
    "dua_sepc_5x5": lambda C_in, C_out, stride, affine, track_running_stats: DualSepConv(
        C_in,
        C_out,
        (5, 5),
        (stride, stride),
        (2, 2),
        (1, 1),
        affine,
        track_running_stats,
    ),
    "dil_sepc_3x3": lambda C_in, C_out, stride, affine, track_running_stats: SepConv(
        C_in,
        C_out,
        (3, 3),
        (stride, stride),
        (2, 2),
        (2, 2),
        affine,
        track_running_stats,
    ),
    "dil_sepc_5x5": lambda C_in, C_out, stride, affine, track_running_stats: SepConv(
        C_in,
        C_out,
        (5, 5),
        (stride, stride),
        (4, 4),
        (2, 2),
        affine,
        track_running_stats,
    ),
    "skip_connect": lambda C_in, C_out, stride, affine, track_running_stats: Identity()
    if stride == 1 and C_in == C_out
    else FactorizedReduce(C_in, C_out, stride, affine, track_running_stats),
}




class ResNetBasicblock(nn.Module):
    def __init__(self, inplanes, planes, stride, affine=True, track_running_stats=True):
        super(ResNetBasicblock, self).__init__()
        assert stride == 1 or stride == 2, "invalid stride {:}".format(stride)
        self.conv_a = ReLUConvBN(
            inplanes, planes, 3, stride, 1, 1, affine, track_running_stats
        )
        self.conv_b = ReLUConvBN(
            planes, planes, 3, 1, 1, 1, affine, track_running_stats
        )
        if stride == 2:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(
                    inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False
                ),
            )
        elif inplanes != planes:
            self.downsample = ReLUConvBN(
                inplanes, planes, 1, 1, 0, 1, affine, track_running_stats
            )
        else:
            self.downsample = None
        self.in_dim = inplanes
        self.out_dim = planes
        self.stride = stride
        self.num_conv = 2

    
    def forward(self, inputs):

        basicblock = self.conv_a(inputs)
        basicblock = self.conv_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        return residual + basicblock

 






 
class InferCell(nn.Module):
    def __init__(
        self, genotype, C_in, C_out, stride, affine=True, track_running_stats=True
    ):
        super(InferCell, self).__init__()

        self.layers = nn.ModuleList()
        self.node_IN = []
        self.node_IX = []
        self.genotype = deepcopy(genotype)
        for i in range(1, len(genotype)):
            node_info = genotype[i - 1]
            cur_index = []
            cur_innod = []
            for (op_name, op_in) in node_info:
                if op_in == 0:
                    layer = OPS[op_name](
                        C_in, C_out, stride, affine, track_running_stats
                    )
                else:
                    layer = OPS[op_name](C_out, C_out, 1, affine, track_running_stats)
                cur_index.append(len(self.layers))
                cur_innod.append(op_in)
                self.layers.append(layer)
            self.node_IX.append(cur_index)
            self.node_IN.append(cur_innod)
        self.nodes = len(genotype)
        self.in_dim = C_in
        self.out_dim = C_out

   



    def forward(self, inputs):
        nodes = [inputs]
        for i, (node_layers, node_innods) in enumerate(zip(self.node_IX, self.node_IN)):
            node_feature = sum(
                self.layers[_il](nodes[_ii])
                for _il, _ii in zip(node_layers, node_innods)
            )
            nodes.append(node_feature)
        return nodes[-1]

    
 
    

# The macro structure for architectures in NAS-Bench-201
class TinyNetwork(nn.Module):
    def __init__(self, C, N, genotype, num_classes):
        super(TinyNetwork, self).__init__()
        self._C = C
        self._layerN = N

        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(C)
        )

        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

        C_prev = C
        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(
            zip(layer_channels, layer_reductions)
        ):
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2, True)
            else:
                cell = InferCell(genotype, C_prev, C_curr, 1)
            self.cells.append(cell)
            C_prev = cell.out_dim
        self._Layer = len(self.cells)

        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    

    def forward(self, inputs):
        feature = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            feature = cell(feature)

        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return logits

