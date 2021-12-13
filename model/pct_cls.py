import torch
import torch.nn as nn
import torch.nn.functional as F
from util import sample_and_group
from model.pct_module import *

"""
    pct分割网络完整版,本文件的pct是分割版pct
"""

class Pct(nn.Module):
    """Pct网络

    网络结构（从上到下，从左到右）：
        Input Embedding module: Neighborhood Embedding i.e. LBR-->SG
        Attention module: four stacked offset-attention layer
        Stack to 1024 channel
        Classification: LBRD-->LBRD-->Linear-->Score
        
    Returns:
        Tensor: 提取到的特征
    """

    def __init__(self, args, output_channels=40):
        super(Pct, self).__init__()
        self.args = args

        # Input Embedding module, here is Neighborhood Embedding
        ## Point Embedding
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        ## SG
        self.gather_local_0 = SG_Layer(in_channels=128, out_channels=128)
        self.gather_local_1 = SG_Layer(in_channels=256, out_channels=256)

        # Attention module, here is Offset-Attention
        self.pt_last = Offset_Attention_Position_Embedding(args)

        # Stack to 1024 channel
        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))

        # Classification
        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        # B, D, N
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=512, radius=0.15, nsample=32, xyz=xyz, points=x)         
        feature_0 = self.gather_local_0(new_feature)
        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=new_xyz, points=feature) 
        feature_1 = self.gather_local_1(new_feature)

        x = self.pt_last(feature_1, new_xyz)
        x = torch.cat([x, feature_1], dim=1)
        x = self.conv_fuse(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)

        return x


class SPct(nn.Module):
    """SPct网络

    网络结构（从上到下，从左到右）：
        Input Embedding module: no Neighborhood Embedding i.e. LBR
        Attention module: four stacked offset-attention layer
        Stack to 1024 channel
        Classification: LBRD-->LBRD-->Linear-->Score
        
    Returns:
        Tensor: 提取到的特征
    """

    def __init__(self, args, output_channels=40):
        super(SPct, self).__init__()
        self.args = args

        # Input Embedding module
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        # Attention module
        self.pt_last = Offset_Attention(args, channels=64)

        # Stack to 1024 channel
        self.conv_fuse = nn.Sequential(nn.Conv1d(256, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))

        # Classification
        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size, _, _ = x.size()
        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        # B, D, N
        x = F.relu(self.bn2(self.conv2(x)))

        x = self.pt_last(x)
        x = self.conv_fuse(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)

        return x

class NPct(nn.Module):
    """NPct网络

    网络结构（从上到下，从左到右）：
        Input Embedding module: No Neighborhood Embedding i.e. LBR
        Attention module: four stacked self-attention layer
        Stack to 1024 channel
        Classification: LBRD-->LBRD-->Linear-->Score
        
    Returns:
        Tensor: 提取到的特征
    """

    def __init__(self, args, output_channels=40):
        super(NPct, self).__init__()

        self.args = args

        # Input Embedding module, no Neighborhood Embedding, Point Embedding i.e. two LBR
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        # Attention module
        self.pt_last = Self_Attention(args, channels=64)

        # Stack to 1024 channel
        self.conv_fuse = nn.Sequential(nn.Conv1d(256, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))

        # Classification
        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size, _, _ = x.size()
        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        # B, D, N
        x = F.relu(self.bn2(self.conv2(x)))

        x = self.pt_last(x)
        x = self.conv_fuse(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)

        return x