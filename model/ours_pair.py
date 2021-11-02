import torch
import torch.nn as nn
import torch.nn.functional as F

from model.SPP import ASPP_simple, ASPP

def softmax_2d(x):
    return torch.exp(x) / torch.sum(torch.sum(torch.exp(x), dim=-1, keepdim=True), dim=-2, keepdim=True)

class Bottleneck(nn.Module):
    
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, rate=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=rate, padding=rate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.rate = rate

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResBlock(nn.Module):
    def __init__(self, in_channel):
        super(ResBlock, self).__init__()
        self.body = nn.Sequential(
                nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, stride=1, bias=True),
                nn.PReLU(),
                nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, stride=1, bias=True)
            )

    def forward(self, x):
        res = self.body(x)
        res += x

        return res

class PredBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PredBlock, self).__init__()
        self.seg = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(out_channel, 1, kernel_size=1, stride=1)
            )

        self.fcn = nn.Sequential(
                nn.Conv2d(in_channel+1, in_channel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(out_channel, 1, kernel_size=3, stride=1, padding=1, bias=False),
                nn.AdaptiveAvgPool2d(1)
            )


    def forward(self, x):
        pred = self.seg(x)
        feat = torch.cat((x, pred), 1)
        feat = self.fcn(feat)
        feat = feat.view(feat.size(0), -1)
        score = torch.sigmoid(feat)
        return pred, score

class DualRefine(nn.Module):
    def __init__(self, in_channel):
        super(DualRefine, self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PReLU()
            )
        self.conv2 = nn.Sequential(
                nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PReLU()
            )
        self.fusion = nn.Conv2d(in_channel*2, in_channel, kernel_size=1)

    def forward(self, x, y, score_x, score_y):
        # attention
        x = score_x.view(-1,1,1,1) * x
        y = score_y.view(-1,1,1,1) * y

        # refine
        err_x = self.conv1(y-x)
        err_y = self.conv2(x-y)
        f_x = x + err_x
        f_y = y + err_y

        # fusion
        f = torch.cat((f_x,f_y), dim=1)
        f = self.fusion(f)

        return f

class RefineBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RefineBlock, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU()
            )

    def forward(self, x, y):
        x = F.interpolate(x, y.size()[2:], mode='bilinear', align_corners=True)
        f = torch.cat([x, y], 1)
        f = self.conv(f)
        return f

class SkipBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SkipBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x

class MyNet(nn.Module):
    def __init__(self, n_classes, os=16):
        super(MyNet, self).__init__()

        self.inplanes = 64

        aspp_rates = [1, 6, 12, 18]
        
        if os == 16:
            strides = [1, 2, 2, 1]
            rates = [1, 1, 1, 2]
        elif os == 8:
            strides = [1, 2, 1, 1]
            rates = [1, 1, 2, 4]

        # Modules
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        layers = [3, 4, 23, 3] # resnet101
        # layers = [3, 4, 6, 3] # resnet50

        self.layer1 = self._make_layer(64, layers[0], stride=strides[0], rate=rates[0])
        self.layer2 = self._make_layer(128, layers[1], stride=strides[1], rate=rates[1])
        self.layer3 = self._make_layer(256, layers[2], stride=strides[2], rate=rates[2])
        self.layer4 = self._make_layer(512, layers[3], stride=strides[3], rate=rates[3])
        
        highInputChannels = 2048
        highOutputChannels = 256
        lowInputChannels = [64,256,512,1024]
        lowOutputChannels = 48
        asppInputChannels = 256
        asppOutputChannels = 256

        self.skipblock1 = SkipBlock(lowInputChannels[0], lowOutputChannels)
        self.skipblock2 = SkipBlock(lowInputChannels[1], lowOutputChannels)
        self.skipblock3 = SkipBlock(lowInputChannels[2], lowOutputChannels)
        self.skipblock4 = SkipBlock(lowInputChannels[3], lowOutputChannels)
        self.skipblock5 = SkipBlock(highInputChannels, highOutputChannels)

        self.pred1 = PredBlock(lowOutputChannels, lowOutputChannels//2)
        self.pred2 = PredBlock(lowOutputChannels, lowOutputChannels//2)
        self.pred3 = PredBlock(lowOutputChannels, lowOutputChannels//2)
        self.pred4 = PredBlock(lowOutputChannels, lowOutputChannels//2)
        self.pred5 = PredBlock(highOutputChannels, highOutputChannels//2)

        # self.pred = PredBlock(lowOutputChannels, lowOutputChannels//2)

        self.aspp = ASPP(asppInputChannels, asppOutputChannels, aspp_rates)

        self.last_conv = nn.Sequential(
                nn.Conv2d(asppOutputChannels+lowOutputChannels, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, n_classes, kernel_size=1, stride=1)
            )

        # for fusion
        self.adp1 = DualRefine(lowOutputChannels)
        self.adp2 = DualRefine(lowOutputChannels)
        self.adp3 = DualRefine(lowOutputChannels)
        self.adp4 = DualRefine(lowOutputChannels)
        self.adp5 = DualRefine(highOutputChannels)

        self.refine4 = RefineBlock(asppOutputChannels+highOutputChannels, asppOutputChannels)
        self.refine3 = RefineBlock(asppOutputChannels+lowOutputChannels, asppOutputChannels)
        self.refine2 = RefineBlock(asppOutputChannels+lowOutputChannels, asppOutputChannels)
        self.refine1 = RefineBlock(asppOutputChannels+lowOutputChannels, asppOutputChannels)


    def _make_layer(self, planes, blocks, stride=1, rate=1):
        
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )

        layers = []
        layers.append(Bottleneck(self.inplanes, planes, stride, rate, downsample))
        self.inplanes = planes * Bottleneck.expansion
        for i in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, img, flo):
        # encoder for image
        x = self.conv1(img)
        x = self.bn1(x)
        x = self.relu(x)
        conv1_feat_x = self.skipblock1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        layer1_feat_x = self.skipblock2(x)
        x = self.layer2(x)
        layer2_feat_x = self.skipblock3(x)
        x = self.layer3(x)
        layer3_feat_x = self.skipblock4(x)
        x = self.layer4(x)
        layer4_feat_x = self.skipblock5(x)

        # encoderfor flow
        y = self.conv1(flo)
        y = self.bn1(y)
        y = self.relu(y)
        conv1_feat_y = self.skipblock1(y)
        y = self.maxpool(y)
        y = self.layer1(y)
        layer1_feat_y = self.skipblock2(y)
        y = self.layer2(y)
        layer2_feat_y = self.skipblock3(y)
        y = self.layer3(y)
        layer3_feat_y = self.skipblock4(y)
        y = self.layer4(y)
        layer4_feat_y = self.skipblock5(y)


        p1_x, score1_x = self.pred1(conv1_feat_x)
        p1_y, score1_y = self.pred1(conv1_feat_y)
        p2_x, score2_x = self.pred2(layer1_feat_x)
        p2_y, score2_y = self.pred2(layer1_feat_y)
        p3_x, score3_x = self.pred3(layer2_feat_x)
        p3_y, score3_y = self.pred3(layer2_feat_y)
        p4_x, score4_x = self.pred4(layer3_feat_x)
        p4_y, score4_y = self.pred4(layer3_feat_y)
        p5_x, score5_x = self.pred5(layer4_feat_x)
        p5_y, score5_y = self.pred5(layer4_feat_y)

        # decoder
        f1 = self.adp1(conv1_feat_x,conv1_feat_y,score1_x,score1_y)
        f2 = self.adp2(layer1_feat_x,layer1_feat_y,score2_x,score2_y)
        f3 = self.adp3(layer2_feat_x,layer2_feat_y,score3_x,score3_y)
        f4 = self.adp4(layer3_feat_x,layer3_feat_y,score4_x,score4_y)
        f5 = self.adp5(layer4_feat_x,layer4_feat_y,score5_x,score5_y)

        f = self.aspp(f5)

        f = self.refine4(f, f5)
        f = self.refine3(f, f4)
        f = self.refine2(f, f3)
        f = self.refine1(f, f2)
        

        f = F.interpolate(f, f1.size()[2:], mode='bilinear', align_corners=True)
        f = torch.cat((f, f1), dim=1)
        f = self.last_conv(f)

        f = F.interpolate(f, img.size()[2:], mode='bilinear', align_corners=True)

        p = {
            'layer5':[p5_x,p5_y,score5_x,score5_y],
            'layer4':[p4_x,p4_y,score4_x,score4_y],
            'layer3':[p3_x,p3_y,score3_x,score3_y],
            'layer2':[p2_x,p2_y,score2_x,score2_y],
            'layer1':[p1_x,p1_y,score1_x,score1_y],
            'final': f
            }

        return p

def init_conv1x1(net):
    for k, v in net.state_dict().items():
        if 'conv1x1' in k:
            if 'weight' in k:
                nn.init.kaiming_normal_(v)
            elif 'bias' in k:
                nn.init.constant_(v, 0)
    return net

def get_params(model, lr):
    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        if 'conv1x1' in key:
            params += [{'params':[value], 'lr':lr*10}]
        else:
            params += [{'params':[value], 'lr':lr}]
    return params

if __name__ == '__main__':
    num_classes = 10
    in_batch, inchannel, in_h, in_w = 4, 3, 448, 448
    x = torch.randn(in_batch, inchannel, in_h, in_w)
    y = torch.randn(in_batch, inchannel, in_h, in_w)
    model = MyNet(num_classes)
    out = model(x,y)
    print(out.shape)
    torch.save(model.state_dict(), 'test.pth')