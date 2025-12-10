import torch
import torch.nn as nn
import torch.nn.functional as F

from experts.PReNetExpert import PReNetExpert
from experts.DEANetExpert import DEANetExpert
from experts.ConvIRExpert import ConvIRExpert


# MoE门控网络
class MoEGate(nn.Module):
    def __init__(self, input_channels=3, num_experts=3, hidden_dim=64):
        super(MoEGate, self).__init__()
        self.num_experts = num_experts
        
        # 门控网络
        self.gate_net = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, 3, padding=1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=1)
        )
        
        # 特征融合门控
        self.feature_gate = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 3, 1),
            nn.Sigmoid()
        )

    def forward(self, x, expert_features):
        # 计算专家权重
        expert_weights = self.gate_net(x)  # [B, num_experts]
        
        # 特征融合权重
        if expert_features:
            # 将所有专家特征拼接
            concat_features = torch.cat(expert_features, dim=1)
            feature_weights = self.feature_gate(concat_features)
            feature_weights = F.softmax(feature_weights, dim=1)
        else:
            feature_weights = None
            
        return expert_weights, feature_weights
    

# 特征融合模块
class FeatureFusionModule(nn.Module):
    def __init__(self, channels_list=[128, 64, 128], output_channels=64):
        super(FeatureFusionModule, self).__init__()
        
        # 多尺度特征融合
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, output_channels, 1),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(True)
            ) for channels in channels_list
        ])
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(output_channels * len(channels_list), output_channels, 1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.ReLU(True)
        )
        
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(output_channels, output_channels // 4, 1),
            nn.ReLU(True),
            nn.Conv2d(output_channels // 4, output_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, features, weights=None):
        adapted_features = []
        
        for i, (adapter, feature) in enumerate(zip(self.adapters, features)):
            if feature is not None:
                adapted = adapter(feature)
                if weights is not None and i < weights.size(1):
                    # 应用权重
                    weight = weights[:, i:i+1].unsqueeze(-1).unsqueeze(-1)
                    adapted = adapted * weight
                adapted_features.append(adapted)
        
        if not adapted_features:
            return None
            
        # 拼接和融合
        fused = torch.cat(adapted_features, dim=1)
        fused = self.fusion_conv(fused)
        
        # 注意力机制
        attention_map = self.attention(fused)
        fused = fused * attention_map
        
        return fused
    

# MoE架构整体
class MoE(nn.Module):
    def __init__(self):
        super(MoE, self).__init__()
        
        # 三个专家网络
        self.derain_expert = PReNetExpert(recurrent_iter=6, return_features=True)
        self.dehaze_expert = DEANetExpert(base_dim=32, return_features=True)
        self.desnow_expert = ConvIRExpert(return_features=True)
        
        # MoE门控网络
        self.moe_gate = MoEGate(input_channels=3, num_experts=3)
        
        # 特征融合模块
        self.feature_fusion = FeatureFusionModule(
            channels_list=[64, 128, 128],  # 对应三个专家的特征通道数
            output_channels=64
        )
        
        # 最终重建网络
        self.reconstruction_net = nn.Sequential(
            nn.Conv2d(64 + 3, 64, 3, padding=1),  # 融合特征 + 原始输入
            nn.ReLU(True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()
        )
        
        # 残差连接
        self.residual_conv = nn.Conv2d(3, 3, 1)

    def forward(self, x):
        # 保存原始输入
        original_input = x
        
        # 并行运行三个专家
        derain_output, derain_features = self.derain_expert(x)
        dehaze_output, dehaze_features = self.dehaze_expert(x)
        desnow_output, desnow_features = self.desnow_expert(x)
        
        # 提取中间层特征
        expert_features = []
        if derain_features:
            expert_features.append(derain_features[0])  # PReNet特征
        if dehaze_features:
            expert_features.append(dehaze_features[0])  # DEANet特征
        if desnow_features:
            expert_features.append(desnow_features[0])  # ConvIR特征
        
        # 门控网络计算权重
        expert_weights, feature_weights = self.moe_gate(x, expert_features)
        
        # 特征融合
        fused_features = self.feature_fusion(expert_features, feature_weights)
        
        # 专家输出加权融合
        expert_outputs = torch.stack([derain_output, dehaze_output, desnow_output], dim=1)  # [B, 3, C, H, W]
        batch_size = expert_outputs.size(0)
        expert_weights = expert_weights.view(batch_size, 3, 1, 1, 1)  # 扩展维度以进行广播
        
        weighted_outputs = expert_outputs * expert_weights
        moe_output = torch.sum(weighted_outputs, dim=1)
        
        # 最终重建（融合特征 + MoE输出）
        if fused_features is not None:
            # 调整特征尺寸以匹配输出
            if fused_features.size(2) != moe_output.size(2):
                fused_features = F.interpolate(fused_features, size=moe_output.shape[2:], mode='bilinear', align_corners=False)
            
            combined = torch.cat([moe_output, fused_features], dim=1)
        else:
            combined = moe_output
        
        # 最终重建
        final_output = self.reconstruction_net(combined)
        
        # 残差连接
        final_output = final_output + self.residual_conv(original_input)
        
        # 返回结果和中间信息（用于训练监控）
        return {
            'final_output': final_output,
            'moe_output': moe_output,
            'expert_weights': expert_weights.squeeze(),
            'derain_output': derain_output,
            'dehaze_output': dehaze_output,
            'desnow_output': desnow_output,
            'fused_features': fused_features
        }