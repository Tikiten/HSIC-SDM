import math
from functools import partial
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Reduce

# from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.fft
from torch import cosine_similarity

from torchsummary import summary
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from thop import clever_format
from thop import profile

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x

class DWMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x.flatten(2).transpose(1, 2)
        return x

def get_frequency_modes(seq_len,modes=32, mode_select_method='random'):
    """
    get modes on frequency domain:
    'random' means sampling randomly;
    'else' means sampling the lowest modes;
    """
    if mode_select_method == 'random':
        index = list(range(0, seq_len))
        np.random.shuffle(index)
        index = index[:modes]
    else:
        index = list(range(0, modes))
    index.sort()
    return index

def run_sobel(conv_x, conv_y, input):
    g_x = conv_x(input)
    g_y = conv_y(input)
    g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2))
    return torch.sigmoid(g) * input

def get_sobel(in_chan, out_chan):
    '''
    filter_x = np.array([
        [3, 0, -3],
        [10, 0, -10],
        [3, 0, -3],
    ]).astype(np.float32)
    filter_y = np.array([
        [3, 10, 3],
        [0, 0, 0],
        [-3, -10, -3],
    ]).astype(np.float32)
    '''
    filter_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ]).astype(np.float32)
    filter_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ]).astype(np.float32)
    filter_x = filter_x.reshape((1, 1, 3, 3))
    filter_x = np.repeat(filter_x, in_chan, axis=1)
    filter_x = np.repeat(filter_x, out_chan, axis=0)

    filter_y = filter_y.reshape((1, 1, 3, 3))
    filter_y = np.repeat(filter_y, in_chan, axis=1)
    filter_y = np.repeat(filter_y, out_chan, axis=0)

    filter_x = torch.from_numpy(filter_x)
    filter_y = torch.from_numpy(filter_y)
    filter_x = nn.Parameter(filter_x, requires_grad=False)
    filter_y = nn.Parameter(filter_y, requires_grad=False)
    conv_x = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_x.weight = filter_x
    conv_y = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_y.weight = filter_y
    sobel_x = nn.Sequential(conv_x, nn.BatchNorm2d(out_chan))
    sobel_y = nn.Sequential(conv_y, nn.BatchNorm2d(out_chan))
    return sobel_x, sobel_y

#中心像素强化1，Q用原始的中心
class CenterAttention(nn.Module):
    def __init__(self, dim, spe_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.spe_dim = spe_dim
        self.scale = head_dim**-0.5
        # self.norm = nn.LayerNorm(dim)
        self.q = nn.Linear(spe_dim, dim)
        # self.q = nn.Linear(dim, dim)
        self.kv = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x_spe, x_spa):
        B, C, H, W = x_spa.shape
        N = H * W
        kv_spa = x_spa.view(B, C, N).permute(0, 2, 1)
        q_spe = x_spe.repeat(1, N, 1)
        # print('q_spe:',q_spe.shape)
        kv = self.kv(kv_spa).reshape(B, -1, 2, C).permute(2, 0, 1, 3) # 3, B, HW, C
        q = self.q(q_spe).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = kv[0].reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = kv[1].reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # print('q:',q.shape)
        # print('k:',k.shape)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = (attn @ v).transpose(2, 3).reshape(B, C, H, W)
        out = x + x_spa

        return out

#中心像素强化2，Q用特征图的中心
class CenterAttention_v2(nn.Module):
    def __init__(self, dim, spe_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # self.spe_dim = spe_dim
        self.scale = head_dim**-0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x_spe, x_spa):
        B, C, H, W = x_spa.shape
        N = H * W
        kv_spa = x_spa.view(B, C, N).permute(0, 2, 1)
        cent_spec_vector = kv_spa[:, int((N - 1) / 2)].unsqueeze(1) #B, 1, C
        # print('cent_spec_vector:',cent_spec_vector.shape)
        q_spe = cent_spec_vector.repeat(1, N, 1)
        # print('q_spe:',q_spe.shape)
        kv = self.kv(kv_spa).reshape(B, -1, 2, C).permute(2, 0, 1, 3) # 3, B, HW, C
        q = self.q(q_spe).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = kv[0].reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = kv[1].reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # print('q:',q.shape)
        # print('k:',k.shape)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = (attn @ v).transpose(2, 3).reshape(B, C, H, W)
        out = x + x_spa

        return out

#中心像素强化3, 余弦距离
class CenterAttention_v3(nn.Module):
    def __init__(self, dim, spe_dim, num_heads):
        super().__init__()
        # self.spe_dim = spe_dim
        # self.norm = nn.LayerNorm(dim)
        # self.q = nn.Linear(dim, dim)
        # self.kv = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, dim * 2)
        # )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x_spe, x_spa):
        B, C, H, W = x_spa.shape
        N = H * W
        kv_spa = x_spa.view(B, C, N).permute(0, 2, 1)
        cent_spec_vector = kv_spa[:, int((N - 1) / 2)].unsqueeze(1) #B, 1, C
        # print('cent_spec_vector:',cent_spec_vector.shape)
        # print('q_spe:',q_spe.shape)
        # kv = self.kv(kv_spa).reshape(B, -1, 2, C).permute(2, 0, 1, 3) # 2, B, HW, C
        k = kv_spa
        v = kv_spa
        # csv_expand = cent_spec_vector.expand(B,N,C)
        # print('csv_expand:',csv_expand.size())

        # ED_sim
        # E_dist = torch.norm(csv_expand - q, dim=2, p=2)
        # sim_E_dist = 1 / (1 + E_dist)
        
        # Cos_sim
        sim_cos = cosine_similarity(cent_spec_vector, k, dim=2)  # include negative
        atten_s  = nn.Softmax(dim=-1)(sim_cos)
        atten_s = torch.unsqueeze(atten_s, 2)
        # print('atten_s:',atten_s.shape)
        q_attened = torch.mul(atten_s, v)
        # print('q_attened:',q_attened.size())
        x = q_attened.contiguous().view(B, C, H, W)
        out = x + x_spa

        return out

#中心像素强化4, 欧式距离
class CenterAttention_v4(nn.Module):
    def __init__(self, dim, spe_dim, num_heads):
        super().__init__()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x_spe, x_spa):
        B, C, H, W = x_spa.shape
        N = H * W
        k = x_spa.view(B, C, N).permute(0, 2, 1)
        v = k
        cent_spec_vector = k[:, int((N - 1) / 2)].unsqueeze(1) #B, 1, C
        # print('cent_spec_vector:',cent_spec_vector.shape)
        # print('q_spe:',q_spe.shape)
        # kv = self.kv(kv_spa).reshape(B, -1, 2, C).permute(2, 0, 1, 3) # 2, B, HW, C
        csv_expand = cent_spec_vector.expand(B,N,C)
        # print('csv_expand:',csv_expand.size())

        # ED_sim
        E_dist = torch.norm(csv_expand - k, dim=2, p=2)
        sim_E_dist = 1 / (1 + E_dist)
        atten_ED = nn.Softmax(dim=-1)(sim_E_dist)
        atten_ED = torch.unsqueeze(atten_ED, 2)
        q_attened = torch.mul(atten_ED, v)
        x = q_attened.contiguous().view(B, C, H, W)
        out = x + x_spa
        
        # Cos_sim
        # sim_cos = cosine_similarity(cent_spec_vector, k, dim=2)  # include negative
        # atten_s  = nn.Softmax(dim=-1)(sim_cos)
        # atten_s = torch.unsqueeze(atten_s, 2)
        # # print('atten_s:',atten_s.shape)
        # q_attened = torch.mul(atten_s, v)
        # x = q_attened.contiguous().view(B, C, H, W)
        # out = x + x_spa

        return out

#边缘信息强化
class EdgeGuidedStripAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.sobel_x1, self.sobel_y1 = get_sobel(dim, 1)
        
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 3), padding=(0, 1), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (3, 1), padding=(1, 0), groups=dim)
        
        self.conv1_1 = nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (5, 1), padding=(2, 0), groups=dim)
        
        self.conv2_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        edge = run_sobel(self.sobel_x1, self.sobel_y1, x)
        edge_att = nn.GELU()(edge)
        # x = edge_att * x
        x = edge_att + x
  
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)

        return attn * u

#边缘信息强化
class EdgeGuidedStripAttention_v2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.upscale = nn.Conv2d(dim,dim*9,1)
        self.sobel_x1, self.sobel_y1 = get_sobel(dim, 1)
        
        # self.convdown = nn.Conv2d(dim, dim, 3, stride=3, padding=1, groups=dim)
        self.conv0 = nn.Conv2d(dim, dim, 3, stride=3, padding=1, groups=dim)
        # self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 3), padding=(0, 1), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (3, 1), padding=(1, 0), groups=dim)
        
        self.conv1_1 = nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (5, 1), padding=(2, 0), groups=dim)
        
        self.conv2_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        pixel_shuffle = nn.PixelShuffle(3)
        x_upscale = self.upscale(x)
        x_subpixel = pixel_shuffle(x_upscale)
        edge = run_sobel(self.sobel_x1, self.sobel_y1, x_subpixel)
        edge_att = nn.GELU()(edge)
        # x = edge_att * x
        x_subpixel = edge_att + x_subpixel
        x = self.convdown(x_subpixel)
  
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)

        return attn * u

class CenterEdgeEnhancedFilter(nn.Module):
    def __init__(self, dim, spe_dim, num_heads, center_choice):
        super().__init__()
        # print('center_choice:',center_choice)
        if center_choice == 0:
            self.center = CenterAttention(dim, spe_dim, num_heads)
        elif center_choice == 1:
            self.center = CenterAttention_v2(dim, spe_dim, num_heads)
        elif center_choice == 2:
            self.center = CenterAttention_v3(dim, spe_dim, num_heads)
        elif center_choice == 3:
            self.center = CenterAttention_v4(dim, spe_dim, num_heads)
        self.edge = EdgeGuidedStripAttention(dim)
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = nn.GELU()
        self.proj_2 = nn.Conv2d(dim, dim, 1)
    
    def forward(self, x_spe, x_spa):
        x = x_spa
        shortcut = x_spa.clone()
        
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = self.proj_1(x)
        x = self.activation(x)
        # x = x.view(B, C, N).permute(0, 2, 1)
        xc = self.center(x_spe, x)
        # x = x.permute(0, 2, 1).view(B, C, H, W)
        xe = self.edge(x)
        x = xc + xe
        x = self.proj_2(x)
        x = x.view(B, C, N).permute(0, 2, 1)
        x = x + shortcut
        return x

#center edge 四种融合方式
class CenterEdgeEnhancedFilter_v2(nn.Module):
    def __init__(self, dim, spe_dim, num_heads, center_choice, ce_choice):
        super().__init__()
        # print('center_choice:',center_choice)
        self.ce_choice = ce_choice
        if self.ce_choice == 1:
            self.proj_cat = nn.Conv2d(dim*2, dim, 1)
            
        if center_choice == 0:
            self.center = CenterAttention(dim, spe_dim, num_heads)
        elif center_choice == 1:
            self.center = CenterAttention_v2(dim, spe_dim, num_heads)
        elif center_choice == 2:
            self.center = CenterAttention_v3(dim, spe_dim, num_heads)
        elif center_choice == 3:
            self.center = CenterAttention_v4(dim, spe_dim, num_heads)
        self.edge = EdgeGuidedStripAttention(dim)
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = nn.GELU()
        self.proj_2 = nn.Conv2d(dim, dim, 1)
    
    def forward(self, x_spe, x_spa):
        x = x_spa
        shortcut = x_spa.clone()
        
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = self.proj_1(x)
        x = self.activation(x)
        if self.ce_choice == 0: #并联相加
            # x = x.view(B, C, N).permute(0, 2, 1)
            xc = self.center(x_spe, x)
            # x = x.permute(0, 2, 1).view(B, C, H, W)
            xe = self.edge(x)
            x = xc + xe
        elif self.ce_choice == 1: #并联cat
            xc = self.center(x_spe, x)
            xe = self.edge(x)
            x = torch.cat([xc,xe],dim=1)
            x = self.proj_cat(x)
        elif self.ce_choice == 2: #顺序串联center-edge
            xc = self.center(x_spe, x)
            xe = self.edge(xc)
            x = xe
        elif self.ce_choice == 3: #顺序串联edge-center
            xe = self.edge(x)
            xc = self.center(x_spe, xe)
            x = xc
        x = self.proj_2(x)
        x = x.view(B, C, N).permute(0, 2, 1)
        x = x + shortcut
        return x

class SpatialBlock(nn.Module):
    def __init__(self, dim, spe_dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_heads=2, init_values=1e-5, center_choice=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm_spe = norm_layer(spe_dim)
        self.attn = CenterEdgeEnhancedFilter(dim, spe_dim, num_heads, center_choice)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = DWMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.scale1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.scale2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)

    def forward(self, input):
        x_spe, x_spa = input
        x = x_spa.clone()
        x = x + self.drop_path(self.scale1 * self.attn(self.norm_spe(x_spe), self.norm1(x_spa)))
        x = x + self.drop_path(self.scale2 * self.mlp(self.norm2(x)))
        return x

class SpatialBlock_v2(nn.Module):
    def __init__(self, dim, spe_dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_heads=2, init_values=1e-5, center_choice=0, ce_choice=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm_spe = norm_layer(spe_dim)
        self.attn = CenterEdgeEnhancedFilter_v2(dim, spe_dim, num_heads, center_choice, ce_choice)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = DWMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.scale1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.scale2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)

    def forward(self, input):
        x_spe, x_spa = input
        x = x_spa.clone()
        x = x + self.drop_path(self.scale1 * self.attn(self.norm_spe(x_spe), self.norm1(x_spa)))
        x = x + self.drop_path(self.scale2 * self.mlp(self.norm2(x)))
        return x

class GlobalFilter(nn.Module):
    def __init__(self, dim, img_size=9, sample_ratio=0.5, channel_fft=True):
        super().__init__()
        N = img_size * img_size
        N_half = N//2 + 1
        N_qut = int(sample_ratio * N_half)
        # print('N:',N)
        # get modes on frequency domain
        self.index = get_frequency_modes(N_half, modes=N_qut, mode_select_method='random')
        # print('modes={}, index={}'.format(N_qut, self.index))
        self.complex_weight = nn.Parameter(torch.randn(N_half, dim, 2, dtype=torch.float32) * 0.02)
        
        self.channel_fft = channel_fft
        if channel_fft:
            self.complex_weightc = nn.Parameter(torch.randn(N, dim//2+1, 2, dtype=torch.float32) * 0.02)
            # print('self.complex_weightc:',self.complex_weightc.shape)
    
    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bh,h->bh", input, weights)

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape
        x = x.to(torch.float32)
        x_ft = torch.fft.rfft(x, dim=1, norm='ortho')
        # print('x_ft:',x_ft.shape)
        weight = torch.view_as_complex(self.complex_weight)
        out_ft = torch.zeros(B, N//2+1, C, device=x.device, dtype=torch.cfloat)
        for wi, i in enumerate(self.index):
            # print('wi i:',wi,i)
            out_ft[:, wi, :] = self.compl_mul1d(x_ft[:, i, :], weight[wi, :])
        # Return to time domain
        x1 = torch.fft.irfft(out_ft, n=N, dim=1, norm='ortho')
        
        if self.channel_fft:
            x_ftC = torch.fft.rfft(x, dim=2, norm='ortho')
            weightc = torch.view_as_complex(self.complex_weightc)
            x_ftC = x_ftC * weightc
            x2 = torch.fft.irfft(x_ftC, n=C, dim=2, norm='ortho')
            out = x1 + x2
        else:
            out = x1
        return out

class Block(nn.Module):

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=14, w=8, flag='Learn'):
        super().__init__()
        self.norm1 = norm_layer(dim)
        # self.filter = GlobalFilter(dim, h=h, w=w)
        self.filter = F3Filter(dim, h=h, w=w, flag=flag)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))
        return x

class BlockLayerScale(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, 
                norm_layer=nn.LayerNorm, img_size=9, init_values=1e-5,
                sample_ration=0.5, channel_fft=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = GlobalFilter(dim, img_size=img_size, sample_ratio=sample_ration, channel_fft=channel_fft)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = DWMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # self.gamma = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.scale1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.scale2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.scale1 * self.filter(self.norm1(x)))
        x = x + self.drop_path(self.scale2 * self.mlp(self.norm2(x)))
        # x = x + self.drop_path(self.gamma * self.mlp(self.norm2(self.filter(self.norm1(x)))))
        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        # print('num_patches:',num_patches)
        # print('img_size:',img_size)
        # print('patch_size:',patch_size)
        # print('inchans:',in_chans)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # print('x:',x.shape)
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class DownLayer(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=13, dim_in=64, dim_out=128):
        super().__init__()
        self.img_size = img_size
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.proj = nn.Conv2d(dim_in, dim_out, kernel_size=5, stride=1, padding=1) #13-9-5则把这里的padding改成0,还有网络里size改成-4i
        self.num_patches = img_size * img_size // 4

    def forward(self, x):
        B, N, C = x.size()
        # print('DownLayer x:',x.size())
        x = x.view(B, self.img_size, self.img_size, C).permute(0, 3, 1, 2)
        # print('DownLayer x:',x.size())
        x = self.proj(x).permute(0, 2, 3, 1)
        # print('DownLayer x:',x.size())
        x = x.reshape(B, -1, self.dim_out)
        # print('DownLayer x:',x.size())
        return x

class StemConv(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=15, embed_dim=64):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size

        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=patch_size, stride=patch_size),  
        )
        
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # print('x:',x.shape)
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=13, dim_in=64, dim_out=128, donwsample=True):
        super().__init__()
        self.img_size = img_size
        self.dim_in = dim_in
        self.dim_out = dim_out
        if donwsample:
            self.proj = nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=0) #13-9-5则把这里的padding改成0,还有网络里size改成-4i
        else:
            self.proj = nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        B, _, C = x.size()
        # print('DownLayer x:',x.size())
        x = x.view(B, self.img_size, self.img_size, C).permute(0, 3, 1, 2)
        # print('DownLayer x:',x.size())
        x = self.proj(x).permute(0, 2, 3, 1)
        # print('DownLayer x:',x.size())
        x = x.reshape(B, -1, self.dim_out)
        # print('DownLayer x:',x.size())
        return x

#金字塔家族
class FEDFormerPyramid(nn.Module):
    def __init__(self, img_size=13, in_chans=30, num_classes=16, embed_dim_init=16, depth_init= 2,
                 mlp_ratio_init=4, drop_rate=0., drop_path_rate=0., norm_layer=None, init_values=0.001, 
                 no_layerscale=False, dropcls=0.1, sample_ration=0.5, channel_fft=True,
                 dataset_min_square=225, org_spe_dim = 200, num_heads=[2,4,8]):
        super().__init__()
        # print('flag:',flag)
        self.name = 'FEDFormerPyramid'
        self.img_size = img_size
        self.org_spe_dim = org_spe_dim
        total_depth = 3
        embed_dim=[]
        depth = []
        mlp_ratio = []
        sizes = []
        for i in range(total_depth):
            embed_dim.append(embed_dim_init*(2**i))
            depth.append(depth_init)
            mlp_ratio.append(mlp_ratio_init)
            sizes.append(img_size - i*2) 
            # sizes.append(img_size - i*4)   #########
        
        # self.num_classes = num_classes
        self.num_features = embed_dim[-1]  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = nn.ModuleList()
        
        self.dataset_min_square = dataset_min_square
        self.fft = GlobalFilter(1, img_size=int(math.sqrt(dataset_min_square)), sample_ratio=1, channel_fft=False)
        
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, padding=(1,1,1),kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(8),
            nn.GELU(),
        )

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=8*in_chans, out_channels=64, padding=(1,1), kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        
        p_size = 1
        patch_embed = PatchEmbed(
                img_size=img_size, patch_size=p_size, in_chans=64, embed_dim=embed_dim[0])
        num_patches = patch_embed.num_patches
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim[0]))

        self.patch_embed.append(patch_embed)

        for i in range(total_depth-1):
            patch_embed = DownLayer(sizes[i], embed_dim[i], embed_dim[i+1])
            num_patches = patch_embed.num_patches
            self.patch_embed.append(patch_embed)

        # print('self.patch_embed:',self.patch_embed)
        # self.pos_drop = nn.Dropout(p=drop_rate)
        self.blocks = nn.ModuleList()
        self.spablocks = nn.ModuleList()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        # print('dpr:',dpr)
        cur = 0
        for i in range(total_depth):
            h = sizes[i]
            w = h // 2 + 1

            if no_layerscale:
                print('using standard block')
                blk = nn.Sequential(*[
                    Block(
                    dim=embed_dim[i], mlp_ratio=mlp_ratio[i],
                    drop=drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer, h=h, w=w, flag=flag)
                for j in range(depth[i])
                ])
            else:
                print('using layerscale block')
                blk = nn.Sequential(*[
                    BlockLayerScale(
                    dim=embed_dim[i], mlp_ratio=mlp_ratio[i],
                    drop=drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer, img_size=sizes[i], 
                    init_values=init_values, sample_ration=sample_ration, channel_fft=channel_fft)
                for j in range(depth[i])
                ])
                
                spablk = nn.Sequential(*[
                    SpatialBlock(
                    dim=embed_dim[i], spe_dim=org_spe_dim,
                    mlp_ratio=mlp_ratio[i],
                    drop=drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer, num_heads=num_heads[i], init_values=init_values)
                for j in range(depth[i])
                ])
                
            self.blocks.append(blk)
            self.spablocks.append(spablk)
            cur += depth[i]

        # Classifier head
        self.norm = norm_layer(self.num_features)
        self.head = nn.Linear(self.num_features, num_classes)
            
        self.batch_norm = nn.Sequential(
            nn.BatchNorm1d(dataset_min_square, eps=0.001, momentum=0.1, affine=True),
            nn.GELU(),
        )
         
        self.to_logits_spe = nn.Sequential(
            Reduce('b c -> b c', 'mean'),
            nn.LayerNorm(dataset_min_square),
            nn.Linear(dataset_min_square, num_classes)
        )    
        
        self.to_logits_main = nn.Sequential(
            nn.LayerNorm(self.num_features),
            nn.Linear(self.num_features, num_classes)
        )

        self.to_logits_spa = nn.Sequential(
            nn.LayerNorm(self.num_features),
            nn.Linear(self.num_features, num_classes)
        )
        
        # trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x_spa, x_spe):
        # print('x_spe:',x_spe.shape) #64，1，1,200
        # print('x_spa:',x_spa.shape) #64，9，9，15
        # 我的hyper需要的标准输入为b,n,h,w,c，一般输入都是B H W C

        b, h, w, c = x_spa.size()
        pad_len = self.dataset_min_square - self.org_spe_dim
        pad = torch.nn.ReflectionPad2d(padding=(0,pad_len, 0, 0))
        x_spe_paded = pad(x_spe).view(b, self.dataset_min_square, -1)
        print('x_spe_paded:',x_spe_paded.shape)
        x_spe_fft = self.fft(x_spe_paded).squeeze(-1)
        # print('x_spe_fft:',x_spe_fft.shape)
        output_spe = self.batch_norm(x_spe_fft)
        # print('output_spe:',output_spe.shape)
        
        x_spe = x_spe.squeeze(1)
        x = x_spa.unsqueeze(1)
        x = rearrange(x, 'b n h w c -> b n c h w')
        x = self.conv3d_features(x)
        x = rearrange(x, 'b c h w y -> b (c h) w y')
        x = self.conv2d_features(x)
        x1 = x.clone()
        x2 = x.clone()
        for i in range(3):
            # print('i:',i)
            x1 = self.patch_embed[i](x1)
            # print('x1:',x1.size())
            # if i == 0:
            #     x = x + self.pos_embed
            x1 = self.blocks[i](x1)
            
            x2 = self.patch_embed[i](x2)
            # print('x2:',x2.size())
            # if i == 0:
            #     x = x + self.pos_embed
            x2 = self.spablocks[i]((x_spe, x2))

        # print('tokens:',x.size())
        output_main = self.norm(x1).mean(1)
        output_spa = self.norm(x2).mean(1)
        # print('output_main:',output_main.shape)
        # print('output_spa:',output_spa.shape)
        
        return output_main, output_spe, output_spa

    def forward(self, x_spa, x_spe):
        fea_main, fea_spe, fea_spa = self.forward_features(x_spa, x_spe)
        output_main = self.to_logits_main(fea_main)
        output_spe = self.to_logits_spe(fea_spe)
        output_spa = self.to_logits_spa(fea_spa)
        
        return output_main, output_spe, output_spa

#版本2
class FEDFormerPyramid_v2(nn.Module):
    def __init__(self, img_size=13, 
                 in_chans=30, 
                 num_classes=16, 
                 embed_dim=[32, 32, 32], 
                 depth=[2,2,2],
                 mlp_ratio=[1,1,1], 
                 num_heads=[2,4,8], 
                 drop_rate=0., 
                 drop_path_rate=0., 
                 dropcls=0.1, 
                 norm_layer=None, 
                 uniform_drop = False,
                 num_stages=3,
                 init_values=0.001, 
                 sample_ration=0.5, 
                 channel_fft=True,
                 dataset_min_square=225, 
                 org_spe_dim = 200):
        super().__init__()
        self.name = 'FEDFormerPyramid_v2'
        self.num_classes = num_classes
        self.num_stages = num_stages
        self.img_size = img_size
        self.org_spe_dim = org_spe_dim
        self.dataset_min_square = dataset_min_square
        self.num_features = embed_dim[-1]  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        
        self.conv3d_dim = 8
        self.conv2d_indim = self.conv3d_dim * in_chans
        self.conv2d_outdim = 64
        
        sizes = []
        for i in range(num_stages):
            sizes.append(img_size - i*2) 

        self.fft = GlobalFilter(1, img_size=int(math.sqrt(dataset_min_square)), sample_ratio=1, channel_fft=False)
        
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=self.conv3d_dim, padding=(1,1,1),kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(self.conv3d_dim),
            nn.GELU(),
        )

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=self.conv2d_indim, out_channels=self.conv2d_outdim, padding=(1,1), kernel_size=(3, 3)),
            nn.BatchNorm2d(self.conv2d_outdim),
            nn.GELU(),
        )

            
        if uniform_drop:
            print('using uniform droppath with expect rate', drop_path_rate)
            dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        else:
            print('using linear droppath with expect rate', drop_path_rate)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        print('dpr is:',dpr)
        # dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = StemConv(img_size=img_size, patch_size=1, 
                                       in_chans=self.conv2d_outdim, embed_dim=embed_dim[0])
            else:
                patch_embed = OverlapPatchEmbed(sizes[i-1], embed_dim[i-1], embed_dim[i])
                
            blk = nn.Sequential(*[
                BlockLayerScale(
                dim=embed_dim[i], mlp_ratio=mlp_ratio[i],
                drop=drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer, img_size=sizes[i], 
                init_values=init_values, sample_ration=sample_ration, channel_fft=channel_fft)
            for j in range(depth[i])
            ])
            
            spablk = nn.Sequential(*[
                SpatialBlock(
                dim=embed_dim[i], spe_dim=org_spe_dim,
                mlp_ratio=mlp_ratio[i],
                drop=drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer, num_heads=num_heads[i], init_values=init_values)
            for j in range(depth[i])
            ])
            
            norm = norm_layer(embed_dim[i])
            cur += depth[i]
            
            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", blk)
            setattr(self, f"spablock{i + 1}", spablk)
            setattr(self, f"norm{i + 1}", norm)

            
        self.spe_norm = nn.Sequential(
            nn.LayerNorm(dataset_min_square),
            # nn.GELU(),
        )
         
        if dropcls > 0:
            print('dropout %.2f before classifier' % dropcls)
            self.final_dropout = nn.Dropout(p=dropcls)
            self.to_logits_spe = nn.Sequential(
                # nn.LayerNorm(dataset_min_square),
                nn.Dropout(p=dropcls),
                nn.Linear(dataset_min_square, num_classes)
            )    
            
            self.to_logits_main = nn.Sequential(
                # nn.LayerNorm(self.num_features),
                nn.Dropout(p=dropcls),
                nn.Linear(self.num_features, num_classes)
            )

            self.to_logits_spa = nn.Sequential(
                # nn.LayerNorm(self.num_features),
                nn.Dropout(p=dropcls),
                nn.Linear(self.num_features, num_classes)
            ) 
        else:
            self.to_logits_spe = nn.Sequential(
                # nn.LayerNorm(dataset_min_square),
                nn.Linear(dataset_min_square, num_classes)
            )    
            
            self.to_logits_main = nn.Sequential(
                # nn.LayerNorm(self.num_features),
                nn.Linear(self.num_features, num_classes)
            )

            self.to_logits_spa = nn.Sequential(
                # nn.LayerNorm(self.num_features),
                nn.Linear(self.num_features, num_classes)
            )
        
        
        # trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x_spa, x_spe):
        # print('x_spe:',x_spe.shape) #64，1，1,200
        # print('x_spa:',x_spa.shape) #64，9，9，15
        # 我的hyper需要的标准输入为b,n,h,w,c，一般输入都是B H W C

        b, _, _, _ = x_spa.size()
        pad_len = self.dataset_min_square - self.org_spe_dim
        pad = torch.nn.ReflectionPad2d(padding=(0,pad_len, 0, 0))
        x_spe_paded = pad(x_spe).view(b, self.dataset_min_square, -1)
        # print('x_spe_paded:',x_spe_paded.shape)
        x_spe_fft = self.fft(x_spe_paded).squeeze(-1)
        # print('x_spe_fft:',x_spe_fft.shape)
        output_spe = self.spe_norm(x_spe_fft)
        # print('output_spe:',output_spe.shape)
        
        x_spe = x_spe.squeeze(1)
        
        x = x_spa.unsqueeze(1)
        x = rearrange(x, 'b n h w c -> b n c h w')
        x = self.conv3d_features(x)
        x = rearrange(x, 'b c h w y -> b (c h) w y')
        x = self.conv2d_features(x)
        x1 = x
        x2 = x
        # print('x:', x.shape)
        for i in range(3):
            # print('i:',i)
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            spablock = getattr(self, f"spablock{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x1 = patch_embed(x1)
            # print('x1:', x1.shape)
            for blk in block:
                x1 = blk(x1)
            x1 = norm(x1)

            x2 = patch_embed(x2)
            # if i == 0:
            #     x2 = patch_embed(x2)
            # print('x2:', x2.shape)
            for spablk in spablock:
                x2 = spablk((x_spe,x2))
            x2 = norm(x2)

        output_main = x1
        output_spa = x2
        # print('output_main:',output_main.shape)
        # print('output_spa:',output_spa.shape)
        
        return output_main, output_spe, output_spa

    def forward(self, x_spa, x_spe):
        fea_main, fea_spe, fea_spa = self.forward_features(x_spa, x_spe)
        output_main = self.to_logits_main(fea_main.mean(1))
        output_spe = self.to_logits_spe(fea_spe)
        output_spa = self.to_logits_spa(fea_spa.mean(1))
        
        return output_main, output_spe, output_spa

class ChannelFilter(nn.Module):
    def __init__(self, dim, img_size=1):
        super().__init__()
        N = img_size * img_size
        self.complex_weightc = nn.Parameter(torch.randn(N, dim//2+1, 2, dtype=torch.float32) * 0.02)
       
    def forward(self, x):
        _, _, C = x.shape
        x = x.to(torch.float32)
        x_ftC = torch.fft.rfft(x, dim=2, norm='ortho')
        weightc = torch.view_as_complex(self.complex_weightc)
        x_ftC = x_ftC * weightc
        x2 = torch.fft.irfft(x_ftC, n=C, dim=2, norm='ortho')
        out = x2
        return out

#版本3
class FEDFormerPyramid_v3(nn.Module):
    def __init__(self, img_size=13, 
                 in_chans=30, 
                 num_classes=16, 
                 embed_dim=[32, 32, 32], 
                 depth=[2,2,2],
                 mlp_ratio=[1,1,1], 
                 num_heads=[2,4,8], 
                 drop_rate=0., 
                 drop_path_rate=0., 
                 dropcls=0.1, 
                 norm_layer=None, 
                 uniform_drop = False,
                 num_stages=3,
                 init_values=0.001, 
                 sample_ration=0.5, 
                 channel_fft=True,
                 org_spe_dim = 200,
                 center_choice = 0):
        super().__init__()
        self.name = 'FEDFormerPyramid_v3'
        self.num_classes = num_classes
        self.num_stages = num_stages
        self.img_size = img_size
        self.org_spe_dim = org_spe_dim
        self.num_features = embed_dim[-1]  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        
        self.conv3d_dim = 8
        self.conv2d_indim = self.conv3d_dim * in_chans
        self.conv2d_outdim = 64
        
        sizes = []
        for i in range(num_stages):
            sizes.append(img_size - i*2) 

        self.fft = ChannelFilter(org_spe_dim, img_size=1)
        
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=self.conv3d_dim, padding=(1,1,1),kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(self.conv3d_dim),
            nn.GELU(),
        )

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=self.conv2d_indim, out_channels=self.conv2d_outdim, padding=(1,1), kernel_size=(3, 3)),
            nn.BatchNorm2d(self.conv2d_outdim),
            nn.GELU(),
        )

            
        if uniform_drop:
            print('using uniform droppath with expect rate', drop_path_rate)
            dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        else:
            print('using linear droppath with expect rate', drop_path_rate)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        print('dpr is:',dpr)
        # dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = StemConv(img_size=img_size, patch_size=1, 
                                       in_chans=self.conv2d_outdim, embed_dim=embed_dim[0])
            else:
                patch_embed = OverlapPatchEmbed(sizes[i-1], embed_dim[i-1], embed_dim[i])
                
            blk = nn.Sequential(*[
                BlockLayerScale(
                dim=embed_dim[i], mlp_ratio=mlp_ratio[i],
                drop=drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer, img_size=sizes[i], 
                init_values=init_values, sample_ration=sample_ration, channel_fft=channel_fft)
            for j in range(depth[i])
            ])
            
            spablk = nn.Sequential(*[
                SpatialBlock(
                dim=embed_dim[i], spe_dim=org_spe_dim,
                mlp_ratio=mlp_ratio[i],
                drop=drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer, num_heads=num_heads[i], init_values=init_values,center_choice=center_choice)
            for j in range(depth[i])
            ])
            
            norm = norm_layer(embed_dim[i])
            cur += depth[i]
            
            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", blk)
            setattr(self, f"spablock{i + 1}", spablk)
            setattr(self, f"norm{i + 1}", norm)

            
        self.spe_norm = nn.Sequential(
            nn.LayerNorm(org_spe_dim),
            # nn.GELU(),
        )
         
        if dropcls > 0:
            print('dropout %.2f before classifier' % dropcls)
            self.final_dropout = nn.Dropout(p=dropcls)
            self.to_logits_spe = nn.Sequential(
                # nn.LayerNorm(dataset_min_square),
                nn.Dropout(p=dropcls),
                nn.Linear(org_spe_dim, num_classes)
            )    
            
            self.to_logits_main = nn.Sequential(
                # nn.LayerNorm(self.num_features),
                nn.Dropout(p=dropcls),
                nn.Linear(self.num_features, num_classes)
            )

            self.to_logits_spa = nn.Sequential(
                # nn.LayerNorm(self.num_features),
                nn.Dropout(p=dropcls),
                nn.Linear(self.num_features, num_classes)
            ) 
        else:
            self.to_logits_spe = nn.Sequential(
                # nn.LayerNorm(dataset_min_square),
                nn.Linear(org_spe_dim, num_classes)
            )    
            
            self.to_logits_main = nn.Sequential(
                # nn.LayerNorm(self.num_features),
                nn.Linear(self.num_features, num_classes)
            )

            self.to_logits_spa = nn.Sequential(
                # nn.LayerNorm(self.num_features),
                nn.Linear(self.num_features, num_classes)
            )
        
        # trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x_spa, x_spe):
        # print('x_spe:',x_spe.shape) #64，1，1,200
        # print('x_spa:',x_spa.shape) #64，9，9，15
        # 我的hyper需要的标准输入为b,n,h,w,c，一般输入都是B H W C
        fre = []
        # b, _, _, _ = x_spa.size()
        x_spe = x_spe.squeeze(1) #64，1,200
        # print('x_spe_paded:',x_spe_paded.shape)
        x_spe_fft = self.fft(x_spe).squeeze(1)
        # print('x_spe_fft:',x_spe_fft.shape)
        output_spe = self.spe_norm(x_spe_fft)
        # print('output_spe:',output_spe.shape)
        
        x = x_spa.unsqueeze(1) #64，1, 9，9，15
        x = rearrange(x, 'b n h w c -> b n c h w')
        x = self.conv3d_features(x)
        x = rearrange(x, 'b c h w y -> b (c h) w y')
        x = self.conv2d_features(x)
        x1 = x
        x2 = x
        # print('x:', x.shape)
        for i in range(3):
            # print('i:',i)
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            spablock = getattr(self, f"spablock{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            
            x1 = patch_embed(x1)
            for blk in block:
                x1 = blk(x1)
            x1 = norm(x1)
            fre.append(x1)

            x2 = patch_embed(x2)
            for spablk in spablock:
                x2 = spablk((x_spe,x2))
            x2 = x2 + fre[i]
            x2 = norm(x2)
            
        output_main = x2.mean(1)
        # output_main = torch.cat([fre[1].mean(1),fre[2].mean(1)],dim=1)
        output_fre = x1.mean(1) #fre对应spa loss
        # print('output_main:',output_main.shape)
        # print('output_fre:',output_fre.shape)
        
        return output_main, output_spe, output_fre

    def forward(self, x_spa, x_spe):
        fea_main, fea_spe, fea_spa = self.forward_features(x_spa, x_spe)
        output_main = self.to_logits_main(fea_main)
        output_spe = self.to_logits_spe(fea_spe)
        output_spa = self.to_logits_spa(fea_spa)
        
        return output_main, output_spe, output_spa

#版本4
class FEDFormerPyramid_v4(nn.Module):
    def __init__(self, img_size=13, 
                 in_chans=30, 
                 num_classes=16, 
                 embed_dim=[32, 32, 32], 
                 depth=[2,2,2],
                 mlp_ratio=[1,1,1], 
                 num_heads=[2,4,8], 
                 drop_rate=0., 
                 drop_path_rate=0., 
                 dropcls=0.1, 
                 norm_layer=None, 
                 uniform_drop = False,
                 num_stages=3,
                 init_values=0.001, 
                 sample_ration=0.5, 
                 channel_fft=True,
                 org_spe_dim = 200,
                 center_choice = 0):
        super().__init__()
        self.name = 'FEDFormerPyramid_v4'
        self.num_classes = num_classes
        self.num_stages = num_stages
        self.img_size = img_size
        self.org_spe_dim = org_spe_dim
        self.num_features = embed_dim[-1]  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        
        self.conv3d_dim = 8
        self.conv2d_indim = self.conv3d_dim * in_chans
        self.conv2d_outdim = 64
        
        sizes = []
        for i in range(num_stages):
            sizes.append(img_size - i*2) 

        self.fft = ChannelFilter(org_spe_dim, img_size=1)
        
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=self.conv3d_dim, padding=(1,1,1),kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(self.conv3d_dim),
            nn.GELU(),
        )

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=self.conv2d_indim, out_channels=self.conv2d_outdim, padding=(1,1), kernel_size=(3, 3)),
            nn.BatchNorm2d(self.conv2d_outdim),
            nn.GELU(),
        )

            
        if uniform_drop:
            print('using uniform droppath with expect rate', drop_path_rate)
            dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        else:
            print('using linear droppath with expect rate', drop_path_rate)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        print('dpr is:',dpr)
        # dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = StemConv(img_size=img_size, patch_size=1, 
                                       in_chans=self.conv2d_outdim, embed_dim=embed_dim[0])
            else:
                patch_embed = OverlapPatchEmbed(sizes[i-1], embed_dim[i-1], embed_dim[i])
                
            blk = nn.Sequential(*[
                BlockLayerScale(
                dim=embed_dim[i], mlp_ratio=mlp_ratio[i],
                drop=drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer, img_size=sizes[i], 
                init_values=init_values, sample_ration=sample_ration, channel_fft=channel_fft)
            for j in range(depth[i])
            ])
            
            spablk = nn.Sequential(*[
                SpatialBlock(
                dim=embed_dim[i], spe_dim=org_spe_dim,
                mlp_ratio=mlp_ratio[i],
                drop=drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer, num_heads=num_heads[i], init_values=init_values,center_choice=center_choice)
            for j in range(depth[i])
            ])
            
            norm = norm_layer(embed_dim[i])
            cur += depth[i]
            
            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", blk)
            setattr(self, f"spablock{i + 1}", spablk)
            setattr(self, f"norm{i + 1}", norm)

            
        self.spe_norm = nn.Sequential(
            nn.LayerNorm(org_spe_dim),
            # nn.GELU(),
        )
         
        if dropcls > 0:
            print('dropout %.2f before classifier' % dropcls)
            self.final_dropout = nn.Dropout(p=dropcls)
            self.to_logits_spe = nn.Sequential(
                # nn.LayerNorm(dataset_min_square),
                nn.Dropout(p=dropcls),
                nn.Linear(org_spe_dim, num_classes)
            )    
            
            self.to_logits_main = nn.Sequential(
                # nn.LayerNorm(self.num_features),
                nn.Dropout(p=dropcls),
                nn.Linear(self.num_features, num_classes)
            )

            self.to_logits_spa = nn.Sequential(
                # nn.LayerNorm(self.num_features),
                nn.Dropout(p=dropcls),
                nn.Linear(self.num_features, num_classes)
            ) 
        else:
            self.to_logits_spe = nn.Sequential(
                # nn.LayerNorm(dataset_min_square),
                nn.Linear(org_spe_dim, num_classes)
            )    
            
            self.to_logits_main = nn.Sequential(
                # nn.LayerNorm(self.num_features),
                nn.Linear(self.num_features, num_classes)
            )

            self.to_logits_spa = nn.Sequential(
                # nn.LayerNorm(self.num_features),
                nn.Linear(self.num_features, num_classes)
            )
        
        # trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x_spa, x_spe):
        # print('x_spe:',x_spe.shape) #64，1，1,200
        # print('x_spa:',x_spa.shape) #64，9，9，15
        # 我的hyper需要的标准输入为b,n,h,w,c，一般输入都是B H W C
        x_spe = x_spe.squeeze(1) #64，1,200
        x_spe_fft = self.fft(x_spe).squeeze(1)
        # print('x_spe_fft:',x_spe_fft.shape)
        output_spe = self.spe_norm(x_spe_fft)
        # print('output_spe:',output_spe.shape)
        
        x = x_spa.unsqueeze(1) #64，1, 9，9，15
        x = rearrange(x, 'b n h w c -> b n c h w')
        x = self.conv3d_features(x)
        x = rearrange(x, 'b c h w y -> b (c h) w y')
        x = self.conv2d_features(x)
        x1 = x
        x2 = x
        # print('x:', x.shape)
        for i in range(3):
            # print('i:',i)
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            spablock = getattr(self, f"spablock{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            
            x1 = patch_embed(x1)
            for blk in block:
                x1 = blk(x1)
            x1 = norm(x1)

            x2 = patch_embed(x2)
            for spablk in spablock:
                x2 = spablk((x_spe,x2))
            x2 = norm(x2)
            
        output_main = x1.mean(1)
        # output_main = torch.cat([fre[1].mean(1),fre[2].mean(1)],dim=1)
        output_spa = x2.mean(1)
        # print('output_main:',output_main.shape)
        # print('output_spa:',output_spa.shape)
        
        return output_main, output_spe, output_spa

    def forward(self, x_spa, x_spe):
        fea_main, fea_spe, fea_spa = self.forward_features(x_spa, x_spe)
        output_main = self.to_logits_main(fea_main)
        output_spe = self.to_logits_spe(fea_spe)
        output_spa = self.to_logits_spa(fea_spa)
        
        return output_main, output_spe, output_spa

#版本5
class FEDFormerPyramid_v5(nn.Module):
    def __init__(self, img_size=13, 
                 in_chans=30, 
                 num_classes=16, 
                 embed_dim=[32, 32, 32], 
                 depth=[2,2,2],
                 mlp_ratio=[1,1,1], 
                 num_heads=[2,4,8], 
                 drop_rate=0., 
                 drop_path_rate=0., 
                 dropcls=0.1, 
                 norm_layer=None, 
                 uniform_drop = False,
                 num_stages=3,
                 init_values=0.001, 
                 sample_ration=0.5, 
                 channel_fft=True,
                 org_spe_dim = 200,
                 center_choice = 0):
        super().__init__()
        self.name = 'FEDFormerPyramid_v5'
        self.num_classes = num_classes
        self.num_stages = num_stages
        self.img_size = img_size
        self.org_spe_dim = org_spe_dim
        self.num_features = embed_dim[-1]  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        
        self.conv3d_dim = 8
        self.conv2d_indim = self.conv3d_dim * in_chans
        self.conv2d_outdim = 64
        
        sizes = []
        for i in range(num_stages):
            sizes.append(img_size - i*2) 

        self.fft = ChannelFilter(org_spe_dim, img_size=1)
        
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=self.conv3d_dim, padding=(1,1,1),kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(self.conv3d_dim),
            nn.GELU(),
        )

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=self.conv2d_indim, out_channels=self.conv2d_outdim, padding=(1,1), kernel_size=(3, 3)),
            nn.BatchNorm2d(self.conv2d_outdim),
            nn.GELU(),
        )

            
        if uniform_drop:
            print('using uniform droppath with expect rate', drop_path_rate)
            dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        else:
            print('using linear droppath with expect rate', drop_path_rate)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        print('dpr is:',dpr)
        # dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = StemConv(img_size=img_size, patch_size=1, 
                                       in_chans=self.conv2d_outdim, embed_dim=embed_dim[0])
            else:
                patch_embed = OverlapPatchEmbed(sizes[i-1], embed_dim[i-1], embed_dim[i])
                
            blk = nn.Sequential(*[
                BlockLayerScale(
                dim=embed_dim[i], mlp_ratio=mlp_ratio[i],
                drop=drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer, img_size=sizes[i], 
                init_values=init_values, sample_ration=sample_ration, channel_fft=channel_fft)
            for j in range(depth[i])
            ])
            
            spablk = nn.Sequential(*[
                SpatialBlock(
                dim=embed_dim[i], spe_dim=org_spe_dim,
                mlp_ratio=mlp_ratio[i],
                drop=drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer, num_heads=num_heads[i], init_values=init_values,center_choice=center_choice)
            for j in range(depth[i])
            ])
            
            norm = norm_layer(embed_dim[i])
            cur += depth[i]
            
            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", blk)
            setattr(self, f"spablock{i + 1}", spablk)
            setattr(self, f"norm{i + 1}", norm)

            
        self.spe_norm = nn.Sequential(
            nn.LayerNorm(org_spe_dim),
            # nn.GELU(),
        )
         
        if dropcls > 0:
            print('dropout %.2f before classifier' % dropcls)
            self.final_dropout = nn.Dropout(p=dropcls)
            self.to_logits_spe = nn.Sequential(
                # nn.LayerNorm(dataset_min_square),
                nn.Dropout(p=dropcls),
                nn.Linear(org_spe_dim, num_classes)
            )    
            
            self.to_logits_main = nn.Sequential(
                # nn.LayerNorm(self.num_features),
                nn.Dropout(p=dropcls),
                nn.Linear(self.num_features, num_classes)
            )

            self.to_logits_spa = nn.Sequential(
                # nn.LayerNorm(self.num_features),
                nn.Dropout(p=dropcls),
                nn.Linear(self.num_features, num_classes)
            ) 
        else:
            self.to_logits_spe = nn.Sequential(
                # nn.LayerNorm(dataset_min_square),
                nn.Linear(org_spe_dim, num_classes)
            )    
            
            self.to_logits_main = nn.Sequential(
                # nn.LayerNorm(self.num_features),
                nn.Linear(self.num_features, num_classes)
            )

            self.to_logits_spa = nn.Sequential(
                # nn.LayerNorm(self.num_features),
                nn.Linear(self.num_features, num_classes)
            )
        
        # trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x_spa, x_spe):
        # print('x_spe:',x_spe.shape) #64，1，1,200
        # print('x_spa:',x_spa.shape) #64，9，9，15
        # 我的hyper需要的标准输入为b,n,h,w,c，一般输入都是B H W C
        fre = []
        # b, _, _, _ = x_spa.size()
        x_spe = x_spe.squeeze(1) #64，1,200
        # print('x_spe_paded:',x_spe_paded.shape)
        x_spe_fft = self.fft(x_spe).squeeze(1)
        # print('x_spe_fft:',x_spe_fft.shape)
        output_spe = self.spe_norm(x_spe_fft)
        # print('output_spe:',output_spe.shape)
        
        x = x_spa.unsqueeze(1) #64，1, 9，9，15
        x = rearrange(x, 'b n h w c -> b n c h w')
        x = self.conv3d_features(x)
        x = rearrange(x, 'b c h w y -> b (c h) w y')
        x = self.conv2d_features(x)
        x1 = x
        x2 = x
        # print('x:', x.shape)
        for i in range(3):
            # print('i:',i)
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            spablock = getattr(self, f"spablock{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            
            x1 = patch_embed(x1)
            for blk in block:
                x1 = blk(x1)
            x1 = norm(x1)
            fre.append(x1)

            x2 = patch_embed(x2)
            for spablk in spablock:
                x2 = spablk((x_spe,x2))
            x2 = x2 + fre[i]
            x2 = norm(x2)
            
        output_main = x2.mean(1)
        # output_main = torch.cat([fre[1].mean(1),fre[2].mean(1)],dim=1)
        output_fre = x1.mean(1) #fre对应spa loss
        # print('output_main:',output_main.shape)
        # print('output_fre:',output_fre.shape)
        
        return output_main, output_spe, output_fre

    def forward(self, x_spa, x_spe):
        fea_main, fea_spe, fea_spa = self.forward_features(x_spa, x_spe)
        output_main = self.to_logits_main(fea_main)
        output_spe = self.to_logits_spe(fea_spe)
        output_fre = self.to_logits_spa(fea_spa)
        
        return output_main, output_spe, output_fre

#版本6 最后concat三个输出用一个LOSS
class FEDFormerPyramid_v6(nn.Module):
    def __init__(self, img_size=13, 
                 in_chans=30, 
                 num_classes=16, 
                 embed_dim=[32, 32, 32], 
                 depth=[2,2,2],
                 mlp_ratio=[1,1,1], 
                 num_heads=[2,4,8], 
                 drop_rate=0., 
                 drop_path_rate=0., 
                 dropcls=0.1, 
                 norm_layer=None, 
                 uniform_drop = False,
                 num_stages=3,
                 init_values=0.001, 
                 sample_ration=0.5, 
                 channel_fft=True,
                 org_spe_dim = 200,
                 center_choice = 0):
        super().__init__()
        self.name = 'FEDFormerPyramid_v6'
        self.num_classes = num_classes
        self.num_stages = num_stages
        self.img_size = img_size
        self.org_spe_dim = org_spe_dim
        self.num_features = embed_dim[-1]  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        
        self.conv3d_dim = 8
        self.conv2d_indim = self.conv3d_dim * in_chans
        self.conv2d_outdim = 64
        
        sizes = []
        for i in range(num_stages):
            sizes.append(img_size - i*2) 

        self.fft = ChannelFilter(org_spe_dim, img_size=1)
        
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=self.conv3d_dim, padding=(1,1,1),kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(self.conv3d_dim),
            nn.GELU(),
        )

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=self.conv2d_indim, out_channels=self.conv2d_outdim, padding=(1,1), kernel_size=(3, 3)),
            nn.BatchNorm2d(self.conv2d_outdim),
            nn.GELU(),
        )

            
        if uniform_drop:
            print('using uniform droppath with expect rate', drop_path_rate)
            dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        else:
            print('using linear droppath with expect rate', drop_path_rate)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        print('dpr is:',dpr)
        # dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = StemConv(img_size=img_size, patch_size=1, 
                                       in_chans=self.conv2d_outdim, embed_dim=embed_dim[0])
            else:
                patch_embed = OverlapPatchEmbed(sizes[i-1], embed_dim[i-1], embed_dim[i])
                
            blk = nn.Sequential(*[
                BlockLayerScale(
                dim=embed_dim[i], mlp_ratio=mlp_ratio[i],
                drop=drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer, img_size=sizes[i], 
                init_values=init_values, sample_ration=sample_ration, channel_fft=channel_fft)
            for j in range(depth[i])
            ])
            
            spablk = nn.Sequential(*[
                SpatialBlock(
                dim=embed_dim[i], spe_dim=org_spe_dim,
                mlp_ratio=mlp_ratio[i],
                drop=drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer, num_heads=num_heads[i], init_values=init_values,center_choice=center_choice)
            for j in range(depth[i])
            ])
            
            norm = norm_layer(embed_dim[i])
            cur += depth[i]
            
            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", blk)
            setattr(self, f"spablock{i + 1}", spablk)
            setattr(self, f"norm{i + 1}", norm)

            
        self.spe_norm = nn.Sequential(
            nn.LayerNorm(org_spe_dim),
            # nn.GELU(),
        )
         
        if dropcls > 0:
            print('dropout %.2f before classifier' % dropcls)
            self.final_dropout = nn.Dropout(p=dropcls)
            # self.to_logits_spe = nn.Sequential(
            #     # nn.LayerNorm(dataset_min_square),
            #     nn.Dropout(p=dropcls),
            #     nn.Linear(org_spe_dim, num_classes)
            # )    
            
            self.to_logits_main = nn.Sequential(
                # nn.LayerNorm(self.num_features),
                nn.Dropout(p=dropcls),
                nn.Linear(self.num_features + self.num_features + org_spe_dim, num_classes)
            )

            # self.to_logits_spa = nn.Sequential(
            #     # nn.LayerNorm(self.num_features),
            #     nn.Dropout(p=dropcls),
            #     nn.Linear(self.num_features, num_classes)
            # ) 
        else:
            # self.to_logits_spe = nn.Sequential(
            #     # nn.LayerNorm(dataset_min_square),
            #     nn.Linear(org_spe_dim, num_classes)
            # )    
            
            self.to_logits_main = nn.Sequential(
                # nn.LayerNorm(self.num_features),
                nn.Linear(self.num_features + self.num_features + org_spe_dim, num_classes)
            )

            # self.to_logits_spa = nn.Sequential(
            #     # nn.LayerNorm(self.num_features),
            #     nn.Linear(self.num_features, num_classes)
            # )
        
        # trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x_spa, x_spe):
        # print('x_spe:',x_spe.shape) #64，1，1,200
        # print('x_spa:',x_spa.shape) #64，9，9，15
        # 我的hyper需要的标准输入为b,n,h,w,c，一般输入都是B H W C
        fre = []
        # b, _, _, _ = x_spa.size()
        x_spe = x_spe.squeeze(1) #64，1,200
        # print('x_spe_paded:',x_spe_paded.shape)
        x_spe_fft = self.fft(x_spe).squeeze(1)
        # print('x_spe_fft:',x_spe_fft.shape)
        output_spe = self.spe_norm(x_spe_fft)
        # print('output_spe:',output_spe.shape)
        
        x = x_spa.unsqueeze(1) #64，1, 9，9，15
        x = rearrange(x, 'b n h w c -> b n c h w')
        x = self.conv3d_features(x)
        x = rearrange(x, 'b c h w y -> b (c h) w y')
        x = self.conv2d_features(x)
        x1 = x
        x2 = x
        # print('x:', x.shape)
        for i in range(3):
            # print('i:',i)
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            spablock = getattr(self, f"spablock{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            
            x1 = patch_embed(x1)
            for blk in block:
                x1 = blk(x1)
            x1 = norm(x1)
            fre.append(x1)

            x2 = patch_embed(x2)
            for spablk in spablock:
                x2 = spablk((x_spe,x2))
            x2 = x2 + fre[i]
            x2 = norm(x2)
            
        output_main = x2.mean(1)
        # output_main = torch.cat([fre[1].mean(1),fre[2].mean(1)],dim=1)
        output_fre = x1.mean(1) #fre对应spa loss
        # print('output_main:',output_main.shape)
        # print('output_fre:',output_fre.shape)
        
        return output_main, output_spe, output_fre

    def forward(self, x_spa, x_spe):
        fea_main, fea_spe, fea_spa = self.forward_features(x_spa, x_spe)
        final_output = torch.cat([fea_main, fea_spe, fea_spa],dim=1)
        output_main = self.to_logits_main(final_output)
        # output_spe = self.to_logits_spe(fea_spe)
        # output_fre = self.to_logits_spa(fea_spa)
        
        return output_main, output_main, output_main

#版本7
class FEDFormerPyramid_v7(nn.Module):
    def __init__(self, img_size=13, 
                 in_chans=30, 
                 num_classes=16, 
                 embed_dim=[32, 32, 32], 
                 depth=[2,2,2],
                 mlp_ratio=[1,1,1], 
                 num_heads=[2,4,8], 
                 drop_rate=0., 
                 drop_path_rate=0., 
                 dropcls=0.1, 
                 norm_layer=None, 
                 uniform_drop = False,
                 num_stages=3,
                 init_values=0.001, 
                 sample_ration=0.5, 
                 channel_fft=True,
                 org_spe_dim = 200,
                 center_choice = 0,
                 ce_choice = 0):
        super().__init__()
        self.name = 'FEDFormerPyramid_v7'
        self.num_classes = num_classes
        self.num_stages = num_stages
        self.img_size = img_size
        self.org_spe_dim = org_spe_dim
        self.num_features = embed_dim[-1]  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        
        self.conv3d_dim = 8
        self.conv2d_indim = self.conv3d_dim * in_chans
        self.conv2d_outdim = 64
        
        sizes = []
        for i in range(num_stages):
            sizes.append(img_size - i*2) 

        self.fft = ChannelFilter(org_spe_dim, img_size=1)
        
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=self.conv3d_dim, padding=(1,1,1),kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(self.conv3d_dim),
            nn.GELU(),
        )

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=self.conv2d_indim, out_channels=self.conv2d_outdim, padding=(1,1), kernel_size=(3, 3)),
            nn.BatchNorm2d(self.conv2d_outdim),
            nn.GELU(),
        )

            
        if uniform_drop:
            print('using uniform droppath with expect rate', drop_path_rate)
            dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        else:
            print('using linear droppath with expect rate', drop_path_rate)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        print('dpr is:',dpr)
        # dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = StemConv(img_size=img_size, patch_size=1, 
                                       in_chans=self.conv2d_outdim, embed_dim=embed_dim[0])
            else:
                patch_embed = OverlapPatchEmbed(sizes[i-1], embed_dim[i-1], embed_dim[i])
                
            blk = nn.Sequential(*[
                BlockLayerScale(
                dim=embed_dim[i], mlp_ratio=mlp_ratio[i],
                drop=drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer, img_size=sizes[i], 
                init_values=init_values, sample_ration=sample_ration, channel_fft=channel_fft)
            for j in range(depth[i])
            ])
            
            spablk = nn.Sequential(*[
                SpatialBlock_v2(
                dim=embed_dim[i], spe_dim=org_spe_dim,
                mlp_ratio=mlp_ratio[i],
                drop=drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer, num_heads=num_heads[i],
                init_values=init_values,center_choice=center_choice, ce_choice=ce_choice)
            for j in range(depth[i])
            ])
            
            norm = norm_layer(embed_dim[i])
            cur += depth[i]
            
            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", blk)
            setattr(self, f"spablock{i + 1}", spablk)
            setattr(self, f"norm{i + 1}", norm)

            
        self.spe_norm = nn.Sequential(
            nn.LayerNorm(org_spe_dim),
            # nn.GELU(),
        )
         
        if dropcls > 0:
            print('dropout %.2f before classifier' % dropcls)
            self.final_dropout = nn.Dropout(p=dropcls)
            # self.to_logits_spe = nn.Sequential(
            #     # nn.LayerNorm(dataset_min_square),
            #     nn.Dropout(p=dropcls),
            #     nn.Linear(org_spe_dim, num_classes)
            # )    
            
            self.to_logits_main = nn.Sequential(
                # nn.LayerNorm(self.num_features),
                nn.Dropout(p=dropcls),
                nn.Linear(self.num_features + self.num_features + org_spe_dim, num_classes)
            )

            # self.to_logits_spa = nn.Sequential(
            #     # nn.LayerNorm(self.num_features),
            #     nn.Dropout(p=dropcls),
            #     nn.Linear(self.num_features, num_classes)
            # ) 
        else:
            # self.to_logits_spe = nn.Sequential(
            #     # nn.LayerNorm(dataset_min_square),
            #     nn.Linear(org_spe_dim, num_classes)
            # )    
            
            self.to_logits_main = nn.Sequential(
                # nn.LayerNorm(self.num_features),
                nn.Linear(self.num_features + self.num_features + org_spe_dim, num_classes)
            )

            # self.to_logits_spa = nn.Sequential(
            #     # nn.LayerNorm(self.num_features),
            #     nn.Linear(self.num_features, num_classes)
            # )
        
        # trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x_spa, x_spe):
        # print('x_spe:',x_spe.shape) #64，1，1,200
        # print('x_spa:',x_spa.shape) #64，9，9，15
        # 我的hyper需要的标准输入为b,n,h,w,c，一般输入都是B H W C
        fre = []
        # b, _, _, _ = x_spa.size()
        x_spe = x_spe.squeeze(1) #64，1,200
        # print('x_spe_paded:',x_spe_paded.shape)
        x_spe_fft = self.fft(x_spe).squeeze(1)
        # print('x_spe_fft:',x_spe_fft.shape)
        output_spe = self.spe_norm(x_spe_fft)
        # print('output_spe:',output_spe.shape)
        
        x = x_spa.unsqueeze(1) #64，1, 9，9，15
        x = rearrange(x, 'b n h w c -> b n c h w')
        x = self.conv3d_features(x)
        x = rearrange(x, 'b c h w y -> b (c h) w y')
        x = self.conv2d_features(x)
        x1 = x
        x2 = x
        # print('x:', x.shape)
        for i in range(3):
            # print('i:',i)
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            spablock = getattr(self, f"spablock{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            
            x1 = patch_embed(x1)
            for blk in block:
                x1 = blk(x1)
            x1 = norm(x1)
            fre.append(x1)

            x2 = patch_embed(x2)
            for spablk in spablock:
                x2 = spablk((x_spe,x2))
            x2 = x2 + fre[i]
            x2 = norm(x2)
            
        output_main = x2.mean(1)
        # output_main = torch.cat([fre[1].mean(1),fre[2].mean(1)],dim=1)
        output_fre = x1.mean(1) #fre对应spa loss
        # print('output_main:',output_main.shape)
        # print('output_fre:',output_fre.shape)
        
        return output_main, output_spe, output_fre

    def forward(self, x_spa, x_spe):
        fea_main, fea_spe, fea_spa = self.forward_features(x_spa, x_spe)
        final_output = torch.cat([fea_main, fea_spe, fea_spa],dim=1)
        output_main = self.to_logits_main(final_output)
        # output_spe = self.to_logits_spe(fea_spe)
        # output_fre = self.to_logits_spa(fea_spa)
        
        return output_main#, output_main, output_main

#版本8 加入了光谱分组
class FEDFormerPyramid_v8(nn.Module):
    def __init__(self, img_size=13, 
                 in_chans=30, 
                 num_classes=16, 
                 embed_dim=[32, 32, 32], 
                 depth=[2,2,2],
                 mlp_ratio=[1,1,1], 
                 num_heads=[2,4,8], 
                 drop_rate=0., 
                 drop_path_rate=0., 
                 dropcls=0.1, 
                 norm_layer=None, 
                 uniform_drop = False,
                 num_stages=3,
                 init_values=0.001, 
                 sample_ration=0.5, 
                 channel_fft=True,
                 org_spe_dim = 200,
                 center_choice = 0,
                 ce_choice = 0):
        super().__init__()
        self.name = 'FEDFormerPyramid_v8'
        self.num_classes = num_classes
        self.num_stages = num_stages
        self.img_size = img_size
        self.org_spe_dim = org_spe_dim
        self.num_features = embed_dim[-1]  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        
        self.conv3d_dim = 8
        self.conv2d_indim = self.conv3d_dim * in_chans
        self.conv2d_outdim = 64
        
        sizes = []
        for i in range(num_stages):
            sizes.append(img_size - i*2) 
        # """---------------------------IN--------------------------"""
        # self.fft_group_all = ChannelFilter(org_spe_dim, img_size=1)
        # self.fft_group1 = ChannelFilter(35, img_size=1)
        # self.fft_group2 = ChannelFilter(21, img_size=1)
        # self.fft_group3 = ChannelFilter(26, img_size=1)
        # self.fft_group4 = ChannelFilter(21, img_size=1)
        # self.fft_group5 = ChannelFilter(41, img_size=1)
        # self.fft_group6 = ChannelFilter(56, img_size=1)
        
        """---------------------------HU--------------------------"""
        self.fft_group_all = ChannelFilter(org_spe_dim, img_size=1)
        self.fft_group1 = ChannelFilter(85, img_size=1)
        self.fft_group2 = ChannelFilter(35, img_size=1)
        self.fft_group3 = ChannelFilter(24, img_size=1)
        
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=self.conv3d_dim, padding=(1,1,1),kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(self.conv3d_dim),
            nn.GELU(),
        )

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=self.conv2d_indim, out_channels=self.conv2d_outdim, padding=(1,1), kernel_size=(3, 3)),
            nn.BatchNorm2d(self.conv2d_outdim),
            nn.GELU(),
        )

            
        if uniform_drop:
            print('using uniform droppath with expect rate', drop_path_rate)
            dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        else:
            print('using linear droppath with expect rate', drop_path_rate)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        print('dpr is:',dpr)
        # dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = StemConv(img_size=img_size, patch_size=1, 
                                       in_chans=self.conv2d_outdim, embed_dim=embed_dim[0])
            else:
                patch_embed = OverlapPatchEmbed(sizes[i-1], embed_dim[i-1], embed_dim[i])
                
            blk = nn.Sequential(*[
                BlockLayerScale(
                dim=embed_dim[i], mlp_ratio=mlp_ratio[i],
                drop=drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer, img_size=sizes[i], 
                init_values=init_values, sample_ration=sample_ration, channel_fft=channel_fft)
            for j in range(depth[i])
            ])
            
            spablk = nn.Sequential(*[
                SpatialBlock_v2(
                dim=embed_dim[i], spe_dim=org_spe_dim,
                mlp_ratio=mlp_ratio[i],
                drop=drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer, num_heads=num_heads[i],
                init_values=init_values,center_choice=center_choice, ce_choice=ce_choice)
            for j in range(depth[i])
            ])
            
            norm = norm_layer(embed_dim[i])
            cur += depth[i]
            
            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", blk)
            setattr(self, f"spablock{i + 1}", spablk)
            setattr(self, f"norm{i + 1}", norm)

            
        self.spe_norm = nn.Sequential(
            nn.LayerNorm(org_spe_dim),
            # nn.GELU(),
        )
         
        if dropcls > 0:
            print('dropout %.2f before classifier' % dropcls)
            self.final_dropout = nn.Dropout(p=dropcls)

            self.to_logits_main = nn.Sequential(
                # nn.LayerNorm(self.num_features),
                nn.Dropout(p=dropcls),
                nn.Linear(self.num_features + self.num_features + org_spe_dim, num_classes)
            )

        else:

            self.to_logits_main = nn.Sequential(
                # nn.LayerNorm(self.num_features),
                nn.Linear(self.num_features + self.num_features + org_spe_dim, num_classes)
            )

        # trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x_spa, x_spe):
        # print('x_spe:',x_spe.shape) #64，1，1,200
        # print('x_spa:',x_spa.shape) #64，9，9，15
        # 我的hyper需要的标准输入为b,n,h,w,c，一般输入都是B H W C       
        fre = []
        x_spe = x_spe.squeeze(1) #64，1,200
        # """---------------------------IN--------------------------"""
        # """---------------------------Group01--------------------------"""
        # group01_output = self.fft_group1(x_spe[:,:,0:35]).squeeze(1)
        # """---------------------------Group02--------------------------"""
        # group02_output = self.fft_group2(x_spe[:,:,35:56]).squeeze(1)
        # """---------------------------Group03--------------------------"""
        # group03_output = self.fft_group3(x_spe[:,:,56:82]).squeeze(1)
        # """---------------------------Group04--------------------------"""
        # group04_output = self.fft_group4(x_spe[:,:,82:103]).squeeze(1)
        # """---------------------------Group05--------------------------"""
        # group05_output = self.fft_group5(x_spe[:,:,103:144]).squeeze(1)
        # """---------------------------Group06--------------------------"""
        # group06_output = self.fft_group6(x_spe[:,:,144:200]).squeeze(1)
            
        # x_spe_fft = torch.cat(
        #     [group01_output, group02_output, group03_output, group04_output, group05_output,
        #      group06_output], dim=1)
        
        """---------------------------HU--------------------------"""
        """---------------------------Group01--------------------------"""
        group01_output = self.fft_group1(x_spe[:,:,0:85]).squeeze(1)
        """---------------------------Group02--------------------------"""
        group02_output = self.fft_group2(x_spe[:,:,85:120]).squeeze(1)
        """---------------------------Group03--------------------------"""
        group03_output = self.fft_group3(x_spe[:,:,120:144]).squeeze(1)
        
        x_spe_fft = torch.cat(
            [group01_output, group02_output, group03_output], dim=1)
        
        output_spe = self.spe_norm(x_spe_fft)

        # print('x_spe_paded:',x_spe_paded.shape)
        # x_spe_fft = self.fft_group_all(x_spe).squeeze(1)
        # print('x_spe_fft:',x_spe_fft.shape)
        output_spe = self.spe_norm(x_spe_fft)
        # print('output_spe:',output_spe.shape)
        
        x = x_spa.unsqueeze(1) #64，1, 9，9，15
        x = rearrange(x, 'b n h w c -> b n c h w')
        x = self.conv3d_features(x)
        x = rearrange(x, 'b c h w y -> b (c h) w y')
        x = self.conv2d_features(x)
        x1 = x
        x2 = x
        # print('x:', x.shape)
        for i in range(3):
            # print('i:',i)
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            spablock = getattr(self, f"spablock{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            
            x1 = patch_embed(x1)
            for blk in block:
                x1 = blk(x1)
            x1 = norm(x1)
            fre.append(x1)

            x2 = patch_embed(x2)
            for spablk in spablock:
                x2 = spablk((x_spe,x2))
            x2 = x2 + fre[i]
            x2 = norm(x2)
            
        output_main = x2.mean(1)
        # output_main = torch.cat([fre[1].mean(1),fre[2].mean(1)],dim=1)
        output_fre = x1.mean(1) #fre对应spa loss
        # print('output_main:',output_main.shape)
        # print('output_fre:',output_fre.shape)
        
        return output_main, output_spe, output_fre

    def forward(self, x_spa, x_spe):
        fea_main, fea_spe, fea_spa = self.forward_features(x_spa, x_spe)
        final_output = torch.cat([fea_main, fea_spe, fea_spa],dim=1)
        output_main = self.to_logits_main(final_output)
        
        return output_main#, output_main, output_main

#版本9,去除fre分支
class FEDFormerPyramid_wofre(nn.Module):
    def __init__(self, img_size=13, 
                 in_chans=30, 
                 num_classes=16, 
                 embed_dim=[32, 32, 32], 
                 depth=[2,2,2],
                 mlp_ratio=[1,1,1], 
                 num_heads=[2,4,8], 
                 drop_rate=0., 
                 drop_path_rate=0., 
                 dropcls=0.1, 
                 norm_layer=None, 
                 uniform_drop = False,
                 num_stages=3,
                 init_values=0.001, 
                 sample_ration=0.5, 
                 channel_fft=True,
                 org_spe_dim = 200,
                 center_choice = 0,
                 ce_choice = 0):
        super().__init__()
        self.name = 'FEDFormerPyramid_wofre'
        self.num_classes = num_classes
        self.num_stages = num_stages
        self.img_size = img_size
        self.org_spe_dim = org_spe_dim
        self.num_features = embed_dim[-1]  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        
        self.conv3d_dim = 8
        self.conv2d_indim = self.conv3d_dim * in_chans
        self.conv2d_outdim = 64
        
        sizes = []
        for i in range(num_stages):
            sizes.append(img_size - i*2) 

        self.fft = ChannelFilter(org_spe_dim, img_size=1)
        
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=self.conv3d_dim, padding=(1,1,1),kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(self.conv3d_dim),
            nn.GELU(),
        )

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=self.conv2d_indim, out_channels=self.conv2d_outdim, padding=(1,1), kernel_size=(3, 3)),
            nn.BatchNorm2d(self.conv2d_outdim),
            nn.GELU(),
        )

            
        if uniform_drop:
            print('using uniform droppath with expect rate', drop_path_rate)
            dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        else:
            print('using linear droppath with expect rate', drop_path_rate)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        print('dpr is:',dpr)
        # dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = StemConv(img_size=img_size, patch_size=1, 
                                       in_chans=self.conv2d_outdim, embed_dim=embed_dim[0])
            else:
                patch_embed = OverlapPatchEmbed(sizes[i-1], embed_dim[i-1], embed_dim[i])
                
            # blk = nn.Sequential(*[
            #     BlockLayerScale(
            #     dim=embed_dim[i], mlp_ratio=mlp_ratio[i],
            #     drop=drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer, img_size=sizes[i], 
            #     init_values=init_values, sample_ration=sample_ration, channel_fft=channel_fft)
            # for j in range(depth[i])
            # ])
            
            spablk = nn.Sequential(*[
                SpatialBlock_v2(
                dim=embed_dim[i], spe_dim=org_spe_dim,
                mlp_ratio=mlp_ratio[i],
                drop=drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer, num_heads=num_heads[i],
                init_values=init_values,center_choice=center_choice, ce_choice=ce_choice)
            for j in range(depth[i])
            ])
            
            norm = norm_layer(embed_dim[i])
            cur += depth[i]
            
            setattr(self, f"patch_embed{i + 1}", patch_embed)
            # setattr(self, f"block{i + 1}", blk)
            setattr(self, f"spablock{i + 1}", spablk)
            setattr(self, f"norm{i + 1}", norm)

            
        self.spe_norm = nn.Sequential(
            nn.LayerNorm(org_spe_dim),
            # nn.GELU(),
        )
         
        if dropcls > 0:
            print('dropout %.2f before classifier' % dropcls)
            self.final_dropout = nn.Dropout(p=dropcls) 
            
            self.to_logits_main = nn.Sequential(
                # nn.LayerNorm(self.num_features),
                nn.Dropout(p=dropcls),
                nn.Linear(self.num_features  + org_spe_dim, num_classes)
            )
        else: 
            self.to_logits_main = nn.Sequential(
                # nn.LayerNorm(self.num_features),
                nn.Linear(self.num_features  + org_spe_dim, num_classes)
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x_spa, x_spe):
        # print('x_spe:',x_spe.shape) #64，1，1,200
        # print('x_spa:',x_spa.shape) #64，9，9，15
        # 我的hyper需要的标准输入为b,n,h,w,c，一般输入都是B H W C
        # fre = []
        # b, _, _, _ = x_spa.size()
        x_spe = x_spe.squeeze(1) #64，1,200
        # print('x_spe_paded:',x_spe_paded.shape)
        x_spe_fft = self.fft(x_spe).squeeze(1)
        # print('x_spe_fft:',x_spe_fft.shape)
        output_spe = self.spe_norm(x_spe_fft)
        # print('output_spe:',output_spe.shape)
        
        x = x_spa.unsqueeze(1) #64，1, 9，9，15
        x = rearrange(x, 'b n h w c -> b n c h w')
        x = self.conv3d_features(x)
        x = rearrange(x, 'b c h w y -> b (c h) w y')
        x = self.conv2d_features(x)
        # x1 = x
        x2 = x
        # print('x:', x.shape)
        for i in range(3):
            # print('i:',i)
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            # block = getattr(self, f"block{i + 1}")
            spablock = getattr(self, f"spablock{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            
            # x1 = patch_embed(x1)
            # for blk in block:
            #     x1 = blk(x1)
            # x1 = norm(x1)
            # fre.append(x1)

            x2 = patch_embed(x2)
            for spablk in spablock:
                x2 = spablk((x_spe,x2))
            x2 = x2 #+ fre[i]
            x2 = norm(x2)
            
        output_main = x2.mean(1)
        # output_main = torch.cat([fre[1].mean(1),fre[2].mean(1)],dim=1)
        # output_fre = x1.mean(1) #fre对应spa loss
        # print('output_main:',output_main.shape)
        # print('output_fre:',output_fre.shape)
        
        return output_main, output_spe, #output_fre

    def forward(self, x_spa, x_spe):
        fea_main, fea_spe = self.forward_features(x_spa, x_spe)
        final_output = torch.cat([fea_main, fea_spe],dim=1)
        output_main = self.to_logits_main(final_output)
        
        return output_main

#版本10,去除spa分支
class FEDFormerPyramid_wospa(nn.Module):
    def __init__(self, img_size=13, 
                 in_chans=30, 
                 num_classes=16, 
                 embed_dim=[32, 32, 32], 
                 depth=[2,2,2],
                 mlp_ratio=[1,1,1], 
                 num_heads=[2,4,8], 
                 drop_rate=0., 
                 drop_path_rate=0., 
                 dropcls=0.1, 
                 norm_layer=None, 
                 uniform_drop = False,
                 num_stages=3,
                 init_values=0.001, 
                 sample_ration=0.5, 
                 channel_fft=True,
                 org_spe_dim = 200,
                 center_choice = 0,
                 ce_choice = 0):
        super().__init__()
        self.name = 'FEDFormerPyramid_wospa'
        self.num_classes = num_classes
        self.num_stages = num_stages
        self.img_size = img_size
        self.org_spe_dim = org_spe_dim
        self.num_features = embed_dim[-1]  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        
        self.conv3d_dim = 8
        self.conv2d_indim = self.conv3d_dim * in_chans
        self.conv2d_outdim = 64
        
        sizes = []
        for i in range(num_stages):
            sizes.append(img_size - i*2) 

        self.fft = ChannelFilter(org_spe_dim, img_size=1)
        
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=self.conv3d_dim, padding=(1,1,1),kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(self.conv3d_dim),
            nn.GELU(),
        )

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=self.conv2d_indim, out_channels=self.conv2d_outdim, padding=(1,1), kernel_size=(3, 3)),
            nn.BatchNorm2d(self.conv2d_outdim),
            nn.GELU(),
        )

            
        if uniform_drop:
            print('using uniform droppath with expect rate', drop_path_rate)
            dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        else:
            print('using linear droppath with expect rate', drop_path_rate)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        print('dpr is:',dpr)
        # dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = StemConv(img_size=img_size, patch_size=1, 
                                       in_chans=self.conv2d_outdim, embed_dim=embed_dim[0])
            else:
                patch_embed = OverlapPatchEmbed(sizes[i-1], embed_dim[i-1], embed_dim[i])
                
            blk = nn.Sequential(*[
                BlockLayerScale(
                dim=embed_dim[i], mlp_ratio=mlp_ratio[i],
                drop=drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer, img_size=sizes[i], 
                init_values=init_values, sample_ration=sample_ration, channel_fft=channel_fft)
            for j in range(depth[i])
            ])
            
            # spablk = nn.Sequential(*[
            #     SpatialBlock_v2(
            #     dim=embed_dim[i], spe_dim=org_spe_dim,
            #     mlp_ratio=mlp_ratio[i],
            #     drop=drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer, num_heads=num_heads[i],
            #     init_values=init_values,center_choice=center_choice, ce_choice=ce_choice)
            # for j in range(depth[i])
            # ])
            
            norm = norm_layer(embed_dim[i])
            cur += depth[i]
            
            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", blk)
            # setattr(self, f"spablock{i + 1}", spablk)
            setattr(self, f"norm{i + 1}", norm)

            
        self.spe_norm = nn.Sequential(
            nn.LayerNorm(org_spe_dim),
            # nn.GELU(),
        )
         
        if dropcls > 0:
            print('dropout %.2f before classifier' % dropcls)
            self.final_dropout = nn.Dropout(p=dropcls) 
            
            self.to_logits_main = nn.Sequential(
                # nn.LayerNorm(self.num_features),
                nn.Dropout(p=dropcls),
                nn.Linear(self.num_features  + org_spe_dim, num_classes)
            )
        else: 
            self.to_logits_main = nn.Sequential(
                # nn.LayerNorm(self.num_features),
                nn.Linear(self.num_features  + org_spe_dim, num_classes)
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x_spa, x_spe):
        # print('x_spe:',x_spe.shape) #64，1，1,200
        # print('x_spa:',x_spa.shape) #64，9，9，15
        # 我的hyper需要的标准输入为b,n,h,w,c，一般输入都是B H W C
        fre = []
        # b, _, _, _ = x_spa.size()
        x_spe = x_spe.squeeze(1) #64，1,200
        # print('x_spe_paded:',x_spe_paded.shape)
        x_spe_fft = self.fft(x_spe).squeeze(1)
        # print('x_spe_fft:',x_spe_fft.shape)
        output_spe = self.spe_norm(x_spe_fft)
        # print('output_spe:',output_spe.shape)
        
        x = x_spa.unsqueeze(1) #64，1, 9，9，15
        x = rearrange(x, 'b n h w c -> b n c h w')
        x = self.conv3d_features(x)
        x = rearrange(x, 'b c h w y -> b (c h) w y')
        x = self.conv2d_features(x)
        x1 = x
        x2 = x
        # print('x:', x.shape)
        for i in range(3):
            # print('i:',i)
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            # spablock = getattr(self, f"spablock{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            
            x1 = patch_embed(x1)
            for blk in block:
                x1 = blk(x1)
            x1 = norm(x1)
            fre.append(x1)

        #     x2 = patch_embed(x2)
        #     for spablk in spablock:
        #         x2 = spablk((x_spe,x2))
        #     x2 = x2 + fre[i]
        #     x2 = norm(x2)
            
        # output_main = x2.mean(1)
        # output_main = torch.cat([fre[1].mean(1),fre[2].mean(1)],dim=1)
        output_fre = x1.mean(1) #fre对应spa loss
        # print('output_main:',output_main.shape)
        # print('output_fre:',output_fre.shape)
        
        return output_spe, output_fre

    def forward(self, x_spa, x_spe):
        fea_spe, fea_spa = self.forward_features(x_spa, x_spe)
        final_output = torch.cat([fea_spe, fea_spa],dim=1)
        output_main = self.to_logits_main(final_output)
        
        return output_main

#版本11,去除spe分支
class FEDFormerPyramid_wospe(nn.Module):
    def __init__(self, img_size=13, 
                 in_chans=30, 
                 num_classes=16, 
                 embed_dim=[32, 32, 32], 
                 depth=[2,2,2],
                 mlp_ratio=[1,1,1], 
                 num_heads=[2,4,8], 
                 drop_rate=0., 
                 drop_path_rate=0., 
                 dropcls=0.1, 
                 norm_layer=None, 
                 uniform_drop = False,
                 num_stages=3,
                 init_values=0.001, 
                 sample_ration=0.5, 
                 channel_fft=True,
                 org_spe_dim = 200,
                 center_choice = 0,
                 ce_choice = 0):
        super().__init__()
        self.name = 'FEDFormerPyramid_wospe'
        self.num_classes = num_classes
        self.num_stages = num_stages
        self.img_size = img_size
        self.org_spe_dim = org_spe_dim
        self.num_features = embed_dim[-1]  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        
        self.conv3d_dim = 8
        self.conv2d_indim = self.conv3d_dim * in_chans
        self.conv2d_outdim = 64
        
        sizes = []
        for i in range(num_stages):
            sizes.append(img_size - i*2) 

        self.fft = ChannelFilter(org_spe_dim, img_size=1)
        
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=self.conv3d_dim, padding=(1,1,1),kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(self.conv3d_dim),
            nn.GELU(),
        )

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=self.conv2d_indim, out_channels=self.conv2d_outdim, padding=(1,1), kernel_size=(3, 3)),
            nn.BatchNorm2d(self.conv2d_outdim),
            nn.GELU(),
        )

            
        if uniform_drop:
            print('using uniform droppath with expect rate', drop_path_rate)
            dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        else:
            print('using linear droppath with expect rate', drop_path_rate)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        print('dpr is:',dpr)
        # dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = StemConv(img_size=img_size, patch_size=1, 
                                       in_chans=self.conv2d_outdim, embed_dim=embed_dim[0])
            else:
                patch_embed = OverlapPatchEmbed(sizes[i-1], embed_dim[i-1], embed_dim[i])
                
            blk = nn.Sequential(*[
                BlockLayerScale(
                dim=embed_dim[i], mlp_ratio=mlp_ratio[i],
                drop=drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer, img_size=sizes[i], 
                init_values=init_values, sample_ration=sample_ration, channel_fft=channel_fft)
            for j in range(depth[i])
            ])
            
            spablk = nn.Sequential(*[
                SpatialBlock_v2(
                dim=embed_dim[i], spe_dim=org_spe_dim,
                mlp_ratio=mlp_ratio[i],
                drop=drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer, num_heads=num_heads[i],
                init_values=init_values,center_choice=center_choice, ce_choice=ce_choice)
            for j in range(depth[i])
            ])
            
            norm = norm_layer(embed_dim[i])
            cur += depth[i]
            
            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", blk)
            setattr(self, f"spablock{i + 1}", spablk)
            setattr(self, f"norm{i + 1}", norm)

            
        self.spe_norm = nn.Sequential(
            nn.LayerNorm(org_spe_dim),
            # nn.GELU(),
        )
         
        if dropcls > 0:
            print('dropout %.2f before classifier' % dropcls)
            self.final_dropout = nn.Dropout(p=dropcls) 
            
            self.to_logits_main = nn.Sequential(
                # nn.LayerNorm(self.num_features),
                nn.Dropout(p=dropcls),
                nn.Linear(self.num_features + self.num_features, num_classes)
            )
        else:
            self.to_logits_main = nn.Sequential(
                # nn.LayerNorm(self.num_features),
                nn.Linear(self.num_features  + self.num_features, num_classes)
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x_spa, x_spe):
        # print('x_spe:',x_spe.shape) #64，1，1,200
        # print('x_spa:',x_spa.shape) #64，9，9，15
        # 我的hyper需要的标准输入为b,n,h,w,c，一般输入都是B H W C
        fre = []
        # b, _, _, _ = x_spa.size()
        x_spe = x_spe.squeeze(1) #64，1,200
        # print('x_spe_paded:',x_spe_paded.shape)
        # x_spe_fft = self.fft(x_spe).squeeze(1)
        # print('x_spe_fft:',x_spe_fft.shape)
        # output_spe = self.spe_norm(x_spe_fft)
        # print('output_spe:',output_spe.shape)
        
        x = x_spa.unsqueeze(1) #64，1, 9，9，15
        x = rearrange(x, 'b n h w c -> b n c h w')
        x = self.conv3d_features(x)
        x = rearrange(x, 'b c h w y -> b (c h) w y')
        x = self.conv2d_features(x)
        x1 = x
        x2 = x
        # print('x:', x.shape)
        for i in range(3):
            # print('i:',i)
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            spablock = getattr(self, f"spablock{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            
            x1 = patch_embed(x1)
            for blk in block:
                x1 = blk(x1)
            x1 = norm(x1)
            fre.append(x1)

            x2 = patch_embed(x2)
            for spablk in spablock:
                x2 = spablk((x_spe,x2))
            x2 = x2 + fre[i]
            x2 = norm(x2)
            
        output_main = x2.mean(1)
        # output_main = torch.cat([fre[1].mean(1),fre[2].mean(1)],dim=1)
        output_fre = x1.mean(1) #fre对应spa loss
        # print('output_main:',output_main.shape)
        # print('output_fre:',output_fre.shape)
        
        return output_main, output_fre

    def forward(self, x_spa, x_spe):
        fea_main, fea_spa = self.forward_features(x_spa, x_spe)
        final_output = torch.cat([fea_main, fea_spa],dim=1)
        output_main = self.to_logits_main(final_output)
        # output_spe = self.to_logits_spe(fea_spe)
        # output_fre = self.to_logits_spa(fea_spa)
        
        return output_main#, output_main, output_main

#版本12,最终多域特征加权concat输出
class FEDFormerPyramid_v12(nn.Module):
    def __init__(self, img_size=13, 
                 in_chans=30, 
                 num_classes=16, 
                 embed_dim=[32, 32, 32], 
                 depth=[2,2,2],
                 mlp_ratio=[1,1,1], 
                 num_heads=[2,4,8], 
                 drop_rate=0., 
                 drop_path_rate=0., 
                 dropcls=0.1, 
                 norm_layer=None, 
                 uniform_drop = False,
                 num_stages=3,
                 init_values=0.001, 
                 sample_ration=0.5, 
                 channel_fft=True,
                 org_spe_dim = 200,
                 center_choice = 0,
                 ce_choice = 0):
        super().__init__()
        self.name = 'FEDFormerPyramid_v12'
        self.num_classes = num_classes
        self.num_stages = num_stages
        self.img_size = img_size
        self.org_spe_dim = org_spe_dim
        self.num_features = embed_dim[-1]  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        
        self.conv3d_dim = 8
        self.conv2d_indim = self.conv3d_dim * in_chans
        self.conv2d_outdim = 64
        
        sizes = []
        for i in range(num_stages):
            sizes.append(img_size - i*2) 

        self.fft = ChannelFilter(org_spe_dim, img_size=1)
        
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=self.conv3d_dim, padding=(1,1,1),kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(self.conv3d_dim),
            nn.GELU(),
        )

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=self.conv2d_indim, out_channels=self.conv2d_outdim, padding=(1,1), kernel_size=(3, 3)),
            nn.BatchNorm2d(self.conv2d_outdim),
            nn.GELU(),
        )

            
        if uniform_drop:
            print('using uniform droppath with expect rate', drop_path_rate)
            dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        else:
            print('using linear droppath with expect rate', drop_path_rate)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        print('dpr is:',dpr)
        # dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = StemConv(img_size=img_size, patch_size=1, 
                                       in_chans=self.conv2d_outdim, embed_dim=embed_dim[0])
            else:
                patch_embed = OverlapPatchEmbed(sizes[i-1], embed_dim[i-1], embed_dim[i])
                
            blk = nn.Sequential(*[
                BlockLayerScale(
                dim=embed_dim[i], mlp_ratio=mlp_ratio[i],
                drop=drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer, img_size=sizes[i], 
                init_values=init_values, sample_ration=sample_ration, channel_fft=channel_fft)
            for j in range(depth[i])
            ])
            
            spablk = nn.Sequential(*[
                SpatialBlock_v2(
                dim=embed_dim[i], spe_dim=org_spe_dim,
                mlp_ratio=mlp_ratio[i],
                drop=drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer, num_heads=num_heads[i],
                init_values=init_values,center_choice=center_choice, ce_choice=ce_choice)
            for j in range(depth[i])
            ])
            
            norm = norm_layer(embed_dim[i])
            cur += depth[i]
            
            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", blk)
            setattr(self, f"spablock{i + 1}", spablk)
            setattr(self, f"norm{i + 1}", norm)

            
        self.spe_norm = nn.Sequential(
            nn.LayerNorm(org_spe_dim),
            # nn.GELU(),
        )
         
        if dropcls > 0:
            print('dropout %.2f before classifier' % dropcls)
            self.final_dropout = nn.Dropout(p=dropcls)
            # self.to_logits_spe = nn.Sequential(
            #     # nn.LayerNorm(dataset_min_square),
            #     nn.Dropout(p=dropcls),
            #     nn.Linear(org_spe_dim, num_classes)
            # )    
            
            self.to_logits_main = nn.Sequential(
                # nn.LayerNorm(self.num_features),
                nn.Dropout(p=dropcls),
                nn.Linear(self.num_features + self.num_features + org_spe_dim, num_classes)
            )

            # self.to_logits_spa = nn.Sequential(
            #     # nn.LayerNorm(self.num_features),
            #     nn.Dropout(p=dropcls),
            #     nn.Linear(self.num_features, num_classes)
            # ) 
        else:
            # self.to_logits_spe = nn.Sequential(
            #     # nn.LayerNorm(dataset_min_square),
            #     nn.Linear(org_spe_dim, num_classes)
            # )    
            
            self.to_logits_main = nn.Sequential(
                # nn.LayerNorm(self.num_features),
                nn.Linear(self.num_features + self.num_features + org_spe_dim, num_classes)
            )

            # self.to_logits_spa = nn.Sequential(
            #     # nn.LayerNorm(self.num_features),
            #     nn.Linear(self.num_features, num_classes)
            # )
        
        self.scale_main = nn.Parameter(1e-5 * torch.ones((self.num_features)),requires_grad=True)
        self.scale_spe = nn.Parameter(1e-5 * torch.ones((org_spe_dim)),requires_grad=True)
        self.scale_fre = nn.Parameter(1e-5 * torch.ones((self.num_features)),requires_grad=True)
        
        # trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x_spa, x_spe):
        # print('x_spe:',x_spe.shape) #64，1，1,200
        # print('x_spa:',x_spa.shape) #64，9，9，15
        # 我的hyper需要的标准输入为b,n,h,w,c，一般输入都是B H W C
        fre = []
        # b, _, _, _ = x_spa.size()
        x_spe = x_spe.squeeze(1) #64，1,200
        # print('x_spe_paded:',x_spe_paded.shape)
        x_spe_fft = self.fft(x_spe).squeeze(1)
        # print('x_spe_fft:',x_spe_fft.shape)
        output_spe = self.spe_norm(x_spe_fft)
        # print('output_spe:',output_spe.shape)
        
        x = x_spa.unsqueeze(1) #64，1, 9，9，15
        x = rearrange(x, 'b n h w c -> b n c h w')
        x = self.conv3d_features(x)
        x = rearrange(x, 'b c h w y -> b (c h) w y')
        x = self.conv2d_features(x)
        x1 = x
        x2 = x
        # print('x:', x.shape)
        for i in range(3):
            # print('i:',i)
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            spablock = getattr(self, f"spablock{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            
            x1 = patch_embed(x1)
            for blk in block:
                x1 = blk(x1)
            x1 = norm(x1)
            fre.append(x1)

            x2 = patch_embed(x2)
            for spablk in spablock:
                x2 = spablk((x_spe,x2))
            x2 = x2 + fre[i]
            x2 = norm(x2)
            
        output_main = x2.mean(1)
        # output_main = torch.cat([fre[1].mean(1),fre[2].mean(1)],dim=1)
        output_fre = x1.mean(1) #fre对应spa loss
        # print('output_main:',output_main.shape)
        # print('output_fre:',output_fre.shape)
        
        return output_main, output_spe, output_fre

    def forward(self, x_spa, x_spe):
        fea_main, fea_spe, fea_fre = self.forward_features(x_spa, x_spe)
        fea_main = self.scale_main * fea_main
        fea_spe = self.scale_spe * fea_spe
        fea_fre = self.scale_fre * fea_fre
        final_output = torch.cat([fea_main, fea_spe, fea_fre],dim=1)
        output_main = self.to_logits_main(final_output)
        # output_spe = self.to_logits_spe(fea_spe)
        # output_fre = self.to_logits_spa(fea_spa)
        
        return output_main#, output_main, output_main



if __name__ == '__main__':
    net = FEDFormerPyramid_v7(img_size=13, 
                                in_chans=15, 
                                num_classes=16, 
                                embed_dim=[64,64,64],  
                                depth=[1,1,1],
                                mlp_ratio=[1,1,1], 
                                num_heads=[8,8,8], #只影响SPA的多头
                                drop_rate=0.1, 
                                drop_path_rate=0.1, 
                                dropcls=0.1, 
                                norm_layer=None, 
                                uniform_drop = False,
                                num_stages=3,
                                init_values=0.001, 
                                sample_ration=0.5, 
                                channel_fft=True,
                                org_spe_dim= 274,
                                center_choice = 0,
                                ce_choice = 0).cuda()
    a = torch.randn(1, 13, 13, 15).cuda()
    b = torch.randn(1, 1, 1, 274).cuda()
    # summary(net, (9, 9, 15), batch_size=1)
    # print('ok!!')
    flops, params = profile(net, inputs=(a,b))
    flops, params = clever_format([flops, params], '%.3f')

    print('模型参数：',params)
    print('每一个样本浮点运算量：',flops)
