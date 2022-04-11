import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import mmd
import torch.nn.functional as F
import torch


N = 8192

__all__ = ['ResNet', 'resnet50']


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


# --------------一维信号的特征处理器---------------
def conv3x1(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    # 3x3 kernel
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

# get BasicBlock which layers < 50(18, 34)
class MyBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None): #inplanes输入通道数，planes输出通道数
        super(MyBlock, self).__init__()
        self.conv1 = conv3x1(inplanes, planes, stride)   # 第一个卷积层，3x1，stride为1，不改变维度。
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x1(planes, planes)   # 第二个卷积层，3x1，输入通道数和输出通道数使用planes，也就是不需要降采样和改变通道数。
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:         # 需要对x降维
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ADDneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):

        super(ADDneck, self).__init__()
        self.conv1 = conv3x1(inplanes, planes, stride)   # 第一个卷积层，3x1，stride为1，不改变维度。
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x1(planes, planes)   # 第二个卷积层，3x3，输入通道数和输出通道数使用planes，也就是不需要降采样和改变通道数。
        self.bn2 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)



        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):   #block代表Bottleneck，layers代表[3,4,6,3]
        self.inplanes = 20
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 20, kernel_size=127, stride=1, padding=64,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(20)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.layer1 = self._make_layer(block, 20, layers[0], stride=1)    # 20代表输出维度！
        self.layer2 = self._make_layer(block, 20, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 20, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 20, layers[3], stride=1)
        # self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        # self.baselayer = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        # self.fc = nn.Linear(40960, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # 这里是为了结局两个残差块之间可能维度不匹配无法直接相加的问题，相同类型的残差块只需要改变第一个输入的维数就好，后面的输入维数都等于输出维数
    def _make_layer(self, block, planes, blocks, stride=1):   #block代表Bottleneck，blocks就是layer[]的参数
        downsample = None
        # 扩维
        if stride != 1 or self.inplanes != planes * block.expansion:    # stride等于1 或者 输入输出维度不相等---降维
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        # 特判第一残差块
        layers.append(block(self.inplanes, planes, stride, downsample))   # 加入 block， 参数：输入，输出维度，步长，是否降维。
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.maxpool(x)

        return x


class MSDAN(nn.Module):

    def __init__(self, num_classes=3):
        super(MSDAN, self).__init__()
        self.sharedNet = resnet_1d(True)
        self.sonnet1 = ADDneck(20, 20)
        self.sonnet2 = ADDneck(20, 20)
        self.sonnet3 = ADDneck(20, 20)
        #加的fc层
        self.cls_fc1_son1 = nn.Linear(5 * N, 256)
        self.cls_fc1_son2 = nn.Linear(5 * N, 256)
        self.cls_fc1_son3 = nn.Linear(5 * N, 256)
        #
        self.cls_fc_son1 = nn.Linear(256, num_classes)
        self.cls_fc_son2 = nn.Linear(256, num_classes)
        self.cls_fc_son3 = nn.Linear(256, num_classes)

        self.avgpool = nn.AvgPool1d(kernel_size=2, stride=2, padding=0)

    def forward(self, data_src, data_tgt = 0, label_src = 0, mark = 1):
        lmmd_loss = 0
        if self.training == True:
            data_src = self.sharedNet(data_src)
            data_tgt = self.sharedNet(data_tgt)

            data_tgt_son1 = self.sonnet1(data_tgt)
            data_tgt_son1 = self.avgpool(data_tgt_son1)
            data_tgt_son1 = data_tgt_son1.view(data_tgt_son1.size(0), -1)
            data_tgt_son1 = self.cls_fc1_son1(data_tgt_son1)   #add
            pred_tgt_son1 = self.cls_fc_son1(data_tgt_son1)

            data_tgt_son2 = self.sonnet2(data_tgt)
            data_tgt_son2 = self.avgpool(data_tgt_son2)
            data_tgt_son2 = data_tgt_son2.view(data_tgt_son2.size(0), -1)      #得到 “目标域” 的三个预测值！
            data_tgt_son2 = self.cls_fc1_son2(data_tgt_son2)  # add
            pred_tgt_son2 = self.cls_fc_son2(data_tgt_son2)

            data_tgt_son3 = self.sonnet3(data_tgt)
            data_tgt_son3 = self.avgpool(data_tgt_son3)
            data_tgt_son3 = data_tgt_son3.view(data_tgt_son3.size(0), -1)
            data_tgt_son3 = self.cls_fc1_son3(data_tgt_son3)  # add
            pred_tgt_son3 = self.cls_fc_son3(data_tgt_son3)

            if mark == 1:

                data_src = self.sonnet1(data_src)
                data_src = self.avgpool(data_src)
                data_src = data_src.view(data_src.size(0), -1)
                data_src = self.cls_fc1_son1(data_src)  #add
                #mmd_loss += mmd.mmd(data_src, data_tgt_son1)

                lmmd_loss = mmd.lmmd(data_src, data_tgt_son1, label_src, torch.nn.functional.softmax(pred_tgt_son1, dim=1))

                l1_loss = torch.mean( torch.abs(torch.nn.functional.softmax(data_tgt_son1, dim=1)
                                                - torch.nn.functional.softmax(data_tgt_son2, dim=1)) )
                l1_loss += torch.mean( torch.abs(torch.nn.functional.softmax(data_tgt_son1, dim=1)
                                                - torch.nn.functional.softmax(data_tgt_son3, dim=1)) )
                pred_src = self.cls_fc_son1(data_src)

                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)

                return cls_loss, lmmd_loss, l1_loss / 2

            if mark == 2:

                data_src = self.sonnet2(data_src)
                data_src = self.avgpool(data_src)
                data_src = data_src.view(data_src.size(0), -1)
                data_src = self.cls_fc1_son2(data_src)  # add
                # mmd_loss += mmd.mmd(data_src, data_tgt_son2)

                lmmd_loss = mmd.lmmd(data_src, data_tgt_son2, label_src, torch.nn.functional.softmax(pred_tgt_son2, dim=1))

                l1_loss = torch.mean( torch.abs(torch.nn.functional.softmax(data_tgt_son2, dim=1)
                                                - torch.nn.functional.softmax(data_tgt_son1, dim=1)) )
                l1_loss += torch.mean( torch.abs(torch.nn.functional.softmax(data_tgt_son2, dim=1)
                                                - torch.nn.functional.softmax(data_tgt_son3, dim=1)) )
                pred_src = self.cls_fc_son2(data_src)

                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)

                return cls_loss, lmmd_loss, l1_loss / 2

            if mark == 3:

                data_src = self.sonnet3(data_src)
                data_src = self.avgpool(data_src)
                data_src = data_src.view(data_src.size(0), -1)
                data_src = self.cls_fc1_son3(data_src)  # add
                # mmd_loss += mmd.mmd(data_src, data_tgt_son3)
                lmmd_loss = mmd.lmmd(data_src, data_tgt_son3, label_src, torch.nn.functional.softmax(pred_tgt_son3, dim=1))

                l1_loss = torch.mean( torch.abs(torch.nn.functional.softmax(data_tgt_son3, dim=1)
                                                - torch.nn.functional.softmax(data_tgt_son1, dim=1)) )
                l1_loss += torch.mean( torch.abs(torch.nn.functional.softmax(data_tgt_son3, dim=1)
                                                - torch.nn.functional.softmax(data_tgt_son2, dim=1)) )
                pred_src = self.cls_fc_son3(data_src)

                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)

                return cls_loss, lmmd_loss, l1_loss / 2

        else:
            data = self.sharedNet(data_src)

            fea_son1 = self.sonnet1(data)
            fea_son1 = self.avgpool(fea_son1)
            fea_son1 = fea_son1.view(fea_son1.size(0), -1)
            fea_son1 = self.cls_fc1_son1(fea_son1)  # add
            pred1 = self.cls_fc_son1(fea_son1)

            fea_son2 = self.sonnet2(data)
            fea_son2 = self.avgpool(fea_son2)
            fea_son2 = fea_son2.view(fea_son2.size(0), -1)
            fea_son2 = self.cls_fc1_son2(fea_son2)  # add
            pred2 = self.cls_fc_son2(fea_son2)

            fea_son3 = self.sonnet3(data)
            fea_son3 = self.avgpool(fea_son3)
            fea_son3 = fea_son3.view(fea_son3.size(0), -1)
            fea_son3 = self.cls_fc1_son3(fea_son3)  # add
            pred3 = self.cls_fc_son3(fea_son3)

            if mark == 1:
                return pred1, pred2, pred3, fea_son1
            if mark == 2:
                return pred1, pred2, pred3, fea_son2
            if mark == 3:
                return pred1, pred2, pred3, fea_son3

            #return pred1, pred2, pred3

def resnet_1d(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(MyBlock, [2, 2, 2, 2], **kwargs)
    return model
