import torch
import torch.nn as nn
import torch.nn.functional as F

# 修改后的PReNet专家网络（去雨）
class PReNetExpert(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True, return_features=True):
        super(PReNetExpert, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU
        self.return_features = return_features
        
        # 原有结构
        self.conv0 = nn.Sequential(nn.Conv2d(6, 32, 3, 1, 1), nn.ReLU())
        
        self.res_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(),
                nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU()
            ) for _ in range(5)
        ])
        
        # LSTM门控机制
        self.conv_i = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.Sigmoid())
        self.conv_f = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.Sigmoid())
        self.conv_g = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.Tanh())
        self.conv_o = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.Sigmoid())
        
        self.conv_out = nn.Conv2d(32, 3, 3, 1, 1)
        
        # MoE适配层
        self.feature_extractor = nn.Conv2d(32, 64, 3, 1, 1)

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        
        x = input
        h = torch.zeros(batch_size, 32, row, col).to(input.device)
        c = torch.zeros(batch_size, 32, row, col).to(input.device)
        
        features = []
        x_list = []
        
        for i in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.conv0(x)
            
            # LSTM更新
            x_cat = torch.cat((x, h), 1)
            i_gate = self.conv_i(x_cat)
            f_gate = self.conv_f(x_cat)
            g_gate = self.conv_g(x_cat)
            o_gate = self.conv_o(x_cat)
            
            c = f_gate * c + i_gate * g_gate
            h = o_gate * torch.tanh(c)
            
            x = h
            # 残差连接
            for res_conv in self.res_convs:
                resx = x
                x = F.relu(res_conv(x) + resx)
            
            # 中间层特征提取
            if i == self.iteration // 2:  # 在中间迭代提取特征
                feature = self.feature_extractor(x)
                features.append(feature)
            
            x = self.conv_out(x)
            x = x + input
            x_list.append(x)
        
        if self.return_features:
            return x, features
        return x