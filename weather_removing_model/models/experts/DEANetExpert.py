import torch.nn as nn

from modules import DEABlockTrain, DEBlockTrain, CGAFusion

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

# 修改后的DEANet专家网络（去雾）
class DEANetExpert(nn.Module):
    def __init__(self, base_dim=32, return_features=True):
        super(DEANetExpert, self).__init__()
        self.return_features = return_features
        
        # 保持原有结构，但添加特征提取点
        self.down1 = nn.Sequential(nn.Conv2d(3, base_dim, kernel_size=3, stride=1, padding=1))
        self.down2 = nn.Sequential(nn.Conv2d(base_dim, base_dim*2, kernel_size=3, stride=2, padding=1), nn.ReLU(True))
        self.down3 = nn.Sequential(nn.Conv2d(base_dim*2, base_dim*4, kernel_size=3, stride=2, padding=1), nn.ReLU(True))
        
        # Level 1 blocks
        self.down_level1_blocks = nn.Sequential(*[DEBlockTrain(default_conv, base_dim, 3) for _ in range(4)])
        self.up_level1_blocks = nn.Sequential(*[DEBlockTrain(default_conv, base_dim, 3) for _ in range(4)])
        
        # Level 2 blocks
        self.fe_level_2 = nn.Conv2d(base_dim*2, base_dim*2, kernel_size=3, stride=1, padding=1)
        self.down_level2_blocks = nn.Sequential(*[DEBlockTrain(default_conv, base_dim*2, 3) for _ in range(4)])
        self.up_level2_blocks = nn.Sequential(*[DEBlockTrain(default_conv, base_dim*2, 3) for _ in range(4)])
        
        # Level 3 blocks
        self.fe_level_3 = nn.Conv2d(base_dim*4, base_dim*4, kernel_size=3, stride=1, padding=1)
        self.level3_blocks = nn.Sequential(*[DEABlockTrain(default_conv, base_dim*4, 3) for _ in range(8)])
        
        # Up-sample
        self.up1 = nn.Sequential(nn.ConvTranspose2d(base_dim*4, base_dim*2, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(base_dim*2, base_dim, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(True))
        self.up3 = nn.Sequential(nn.Conv2d(base_dim, 3, kernel_size=3, stride=1, padding=1))
        
        # Feature fusion
        self.mix1 = CGAFusion(base_dim*4, reduction=8)
        self.mix2 = CGAFusion(base_dim*2, reduction=4)
        
        # MoE适配层
        self.moe_adapter = nn.Conv2d(base_dim*4, base_dim*4, kernel_size=1)

    def forward(self, x):
        features = []
        
        # Level 1
        x_down1 = self.down1(x)
        x_down1 = self.down_level1_blocks(x_down1)
        
        # Level 2
        x_down2 = self.down2(x_down1)
        x_down2_init = self.fe_level_2(x_down2)
        x_down2_init = self.down_level2_blocks(x_down2_init)
        
        # Level 3 (中间层特征提取点)
        x_down3 = self.down3(x_down2_init)
        x_down3_init = self.fe_level_3(x_down3)
        x_level3 = self.level3_blocks(x_down3_init)
        features.append(x_level3)  # 中间层特征
        
        x_level3_mix = self.mix1(x_down3, x_level3)
        
        # Up-sample路径
        x_up1 = self.up1(x_level3_mix)
        x_up1 = self.up_level2_blocks(x_up1)
        
        x_level2_mix = self.mix2(x_down2, x_up1)
        x_up2 = self.up2(x_level2_mix)
        x_up2 = self.up_level1_blocks(x_up2)
        out = self.up3(x_up2)
        
        if self.return_features:
            return out, features
        return out