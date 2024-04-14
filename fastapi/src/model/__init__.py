from torchvision import models
from .FrEIA import framework as Ff
from .FrEIA import modules as Fm
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetWideFeat(torch.nn.Module):
    def __init__(self, pretrained=False):
        super(ResNetWideFeat, self).__init__()

        if pretrained:
            ResnetWide = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.DEFAULT)
        else:
            ResnetWide = models.wide_resnet50_2()
        
        self.conv1 = ResnetWide.conv1
        self.bn1 = ResnetWide.bn1
        self.relu = ResnetWide.relu
        self.maxpool = ResnetWide.maxpool
        
        self.layer1 = ResnetWide.layer1
        self.layer2 = ResnetWide.layer2
        self.layer3 = ResnetWide.layer3
        
        self.channels = [256, 512, 1024]
        self.reduction = [4, 8, 16]
        
    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        return [x1, x2, x3]

def subnet_conv_func(kernel_size, hidden_ratio):
    def subnet_conv(in_channels, out_channels):
        hidden_channels = int(in_channels * hidden_ratio)
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size, padding="same"),
        )

    return subnet_conv

def nf_fast_flow(input_chw, conv3x3_only, hidden_ratio, flow_steps, clamp=2.0):
    nodes = Ff.SequenceINN(*input_chw)
    for i in range(flow_steps):
        if i % 2 == 1 and not conv3x3_only:
            kernel_size = 1
        else:
            kernel_size = 3
        nodes.append(
            Fm.AllInOneBlock,
            subnet_constructor=subnet_conv_func(kernel_size, hidden_ratio),
            affine_clamping=clamp,
            permute_soft=False,
        )
    return nodes

class FastFlow(nn.Module):
    def __init__(
        self,
        input_size,
        flow_steps=8,
        conv3x3_only=False,
        hidden_ratio=1.0,
        pretrained=False
    ):
        super(FastFlow, self).__init__()
        
        self.feature_extractor = ResNetWideFeat(pretrained)
        channels = self.feature_extractor.channels
        scales = self.feature_extractor.reduction

        # for transformers, use their pretrained norm w/o grad
        # for resnets, self.norms are trainable LayerNorm
        self.norms = nn.ModuleList()
        for in_channels, scale in zip(channels, scales):
            self.norms.append(
                nn.LayerNorm(
                    [in_channels, int(input_size[1] / scale), int(input_size[0] / scale)],
                    elementwise_affine=True,
                )
            )

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.nf_flows = nn.ModuleList()
        for in_channels, scale in zip(channels, scales):
            self.nf_flows.append(
                nf_fast_flow(
                    [in_channels, int(input_size[1] / scale), int(input_size[0] / scale)],
                    conv3x3_only=conv3x3_only,
                    hidden_ratio=hidden_ratio,
                    flow_steps=flow_steps,
                )
            )
        self.input_size = input_size

    def forward(self, x):
        self.feature_extractor.eval()
        features = self.feature_extractor(x)
        features = [self.norms[i](feature) for i, feature in enumerate(features)]

        loss = 0
        outputs = []
        for i, feature in enumerate(features):
            output, log_jac_dets = self.nf_flows[i](feature)
            loss += torch.mean(
                0.5 * torch.sum(output**2, dim=(1, 2, 3)) - log_jac_dets
            )
            outputs.append(output)
        ret = {"loss": loss}

        if not self.training:
            anomaly_map_list = []
            for output in outputs:
                log_prob = -torch.mean(output**2, dim=1, keepdim=True) * 0.5
                prob = torch.exp(log_prob)
                a_map = F.interpolate(
                    -prob,
                    size=[self.input_size[1], self.input_size[0]],
                    mode="bilinear",
                    align_corners=False,
                )
                anomaly_map_list.append(a_map)
            anomaly_map_list = torch.stack(anomaly_map_list, dim=-1)
            anomaly_map = torch.mean(anomaly_map_list, dim=-1)
            ret["anomaly_map"] = anomaly_map
        return ret
