import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    Pct中各个模块函数
"""

class SG_Layer(nn.Module):
    """
    SG module 后半部分，前半部分在sample_and_group中实现了
    """

    def __init__(self, in_channels, out_channels):
        super(SG_Layer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6]) 
        x = x.permute(0, 1, 3, 2) # B, N, D, S
        x = x.reshape(-1, d, s) # B*N, D, S

        batch_size_new, _, _ = x.size() # batch_size_new = B*N
        x = F.relu(self.bn1(self.conv1(x))) # B*N, D, S
        x = F.relu(self.bn2(self.conv2(x))) # B*N, D, S

        # adaptive_max_pool1d(x, 1)将S池化为1
        # .view(batch_size, -1)将最后一个维度删去
        x = F.adaptive_max_pool1d(x, 1).view(batch_size_new, -1) # B*N, D
        x = x.reshape(b, n, -1).permute(0, 2, 1) # B, D, N
        return x

class Offset_Attention_Position_Embedding(nn.Module):
    """
    Attention Modules: four stacked attention layer
    """

    def __init__(self, args, channels=256):
        super(Offset_Attention_Position_Embedding, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.pos_xyz = nn.Conv1d(3, channels, 1) # position embedding
        self.bn1 = nn.BatchNorm1d(channels)

        self.oa1 = OA_Layer_Point_Embedding(channels)
        self.oa2 = OA_Layer_Point_Embedding(channels)
        self.oa3 = OA_Layer_Point_Embedding(channels)
        self.oa4 = OA_Layer_Point_Embedding(channels)

    def forward(self, x, xyz):
        # 
        # b, 3, npoint, nsample  
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample 
        # permute reshape
        batch_size, _, N = x.size()
        xyz = xyz.permute(0, 2, 1)
        xyz = self.pos_xyz(xyz)
        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x1 = self.oa1(x, xyz)
        x2 = self.oa2(x1, xyz)
        x3 = self.oa3(x2, xyz)
        x4 = self.oa4(x3, xyz)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x

class OA_Layer_Point_Embedding(nn.Module):
    """
    Offset-Attention Layer
    """

    def __init__(self, channels):
        super(OA_Layer_Point_Embedding, self).__init__()

        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, xyz):
        # b, n, c
        x = x + xyz
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k)

        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True)) # l1 Norm
        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x

class Offset_Attention(nn.Module):
    """
    Attention Modules: four stacked attention layer
    """

    def __init__(self, args, channels=256):
        super(Offset_Attention, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.oa1 = OA_Layer(channels)
        self.oa2 = OA_Layer(channels)
        self.oa3 = OA_Layer(channels)
        self.oa4 = OA_Layer(channels)

    def forward(self, x):
        # 
        # b, 3, npoint, nsample  
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample 
        # permute reshape
        batch_size, _, N = x.size()
        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x1 = self.oa1(x)
        x2 = self.oa2(x1)
        x3 = self.oa3(x2)
        x4 = self.oa4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x

class OA_Layer(nn.Module):
    """
    Offset-Attention Layer
    """

    def __init__(self, channels):
        super(OA_Layer, self).__init__()

        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # b, n, c
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k)

        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True)) # l1 Norm
        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x

class Self_Attention(nn.Module):
    """
    Attention Modules: four stacked attention layer
    Note: no Position Embedding
    """

    def __init__(self, args, channels=256):
        super(Self_Attention, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)


        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

    def forward(self, x):
        # 
        # b, 3, npoint, nsample  
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample 
        # permute reshape
        batch_size, _, N = x.size()

        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x

class SA_Layer(nn.Module):
    """
    Self-Attention Layer
    """

    def __init__(self, channels):
        super(SA_Layer, self).__init__()

        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # b, n, c
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k) # bmm计算两个诸如(b,m,n)和(b,n,l)，得到(b,m,l)

        # Scale
        _, n, _ = energy.size()
        attention = energy / (n ** 0.5)
        # Softmax
        attention = self.softmax(energy)

        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x_r)))
        x = x + x_r
        return x