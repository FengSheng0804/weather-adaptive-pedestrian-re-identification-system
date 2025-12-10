import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import BasicConv, EBlock, DBlock, FAM, SCM

# 修改后的ConvIR专家网络（去雪），采用的是经过重参数化后的网络结构
class ConvIRExpert(nn.Module):
    def __init__(self, num_res=8, base_channel=32, return_features=True):
        super(ConvIRExpert, self).__init__()
        self.return_features = return_features
                
        # 编码器
        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])
        
        # 特征提取
        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])
        
        # 解码器
        self.Decoder = nn.ModuleList([
            DBlock(base_channel*4, num_res),
            DBlock(base_channel*2, num_res),
            DBlock(base_channel, num_res)
        ])
        
        # 连接层
        self.Convs = nn.ModuleList([
            BasicConv(base_channel*4, base_channel*2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel*2, base_channel, kernel_size=1, relu=True, stride=1),
        ])
        
        # 输出层
        self.ConvsOut = nn.ModuleList([
            BasicConv(base_channel*4, 3, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel*2, 3, kernel_size=3, relu=False, stride=1),
        ])
        
        # 特征对齐模块
        self.FAM1 = FAM(base_channel*4)
        self.SCM1 = SCM(base_channel*4)
        self.FAM2 = FAM(base_channel*2)
        self.SCM2 = SCM(base_channel*2)
        
        # MoE适配层
        self.moe_adapter = nn.Conv2d(base_channel*4, base_channel*4, kernel_size=1)

    def forward(self, x):
        features = []
        
        # 多尺度输入
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)
        
        outputs = []
        
        # 编码路径
        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)
        
        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)
        
        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z_encoded = self.Encoder[2](z)
        
        # 中间层特征提取
        features.append(self.moe_adapter(z_encoded))
        
        # 解码路径
        z = self.Decoder[0](z_encoded)
        z_ = self.ConvsOut[0](z)
        outputs.append(z_ + x_4)
        
        z = self.feat_extract[3](z)
        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        outputs.append(z_ + x_2)
        
        z = self.feat_extract[4](z)
        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z + x)
        
        if self.return_features:
            return outputs[-1], features  # 返回最终输出和特征
        return outputs[-1]