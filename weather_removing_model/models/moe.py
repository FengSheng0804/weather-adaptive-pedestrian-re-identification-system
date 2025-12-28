import torch
import torch.nn as nn
import torch.nn.functional as F

from .experts.PReNetExpert import PReNetExpert
from .experts.DEANetExpert import DEANetExpert
from .experts.ConvIRExpert import ConvIRExpert


# MoE门控网络
class MoEGate(nn.Module):
    def __init__(self, input_channels=3, num_experts=3, hidden_dim=256, score_dim=1):
        super(MoEGate, self).__init__()
        self.num_experts = num_experts
        self.score_dim = score_dim
        # 门控网络：图像特征+score拼接
        self.gate_conv = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.expert_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, num_experts),
            nn.Sigmoid()
        )
        self.feature_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, num_experts),
            nn.Sigmoid()
        )

    def forward(self, x, score=None):
        # x: [B, C, H, W], score: [B, score_dim] or None
        feat = self.gate_conv(x)  # [B, hidden_dim]
        
        expert_weights = self.expert_fc(feat)   # [B, num_experts]
        feature_weights = self.feature_fc(feat) # [B, num_experts]

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
        # 统一空间尺寸
        ref_shape = adapted_features[0].shape[2:]
        resized_features = [
            F.interpolate(f, size=ref_shape, mode='bilinear', align_corners=False) if f.shape[2:] != ref_shape else f
            for f in adapted_features
        ]
        # 拼接和融合
        fused = torch.cat(resized_features, dim=1)
        fused = self.fusion_conv(fused)
        
        # 注意力机制
        attention_map = self.attention(fused)
        fused = fused * attention_map
        return fused
    

# MoE架构整体
class MoE(nn.Module):
    def __init__(self, score_dim=1):
        super(MoE, self).__init__()
        # 三个专家网络
        self.derain_expert = PReNetExpert(recurrent_iter=6, return_features=True)
        self.defog_expert = DEANetExpert(base_dim=32, return_features=True)
        self.desnow_expert = ConvIRExpert(num_res=8, base_channel=32, return_features=True)
        # MoE门控网络，支持score输入
        self.moe_gate = MoEGate(input_channels=3, num_experts=3, score_dim=score_dim)
        # 特征融合模块
        self.feature_fusion = FeatureFusionModule(
            channels_list=[128, 64, 128],  # 对应三个专家的特征通道数
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

    def forward(self, x, score=None):
        # 保存原始输入
        original_input = x
        # 并行运行三个专家
        defog_output, defog_features = self.defog_expert(x)
        derain_output, derain_features = self.derain_expert(x)
        desnow_output, desnow_features = self.desnow_expert(x)
        # 提取中间层特征，按照fog、rain、snow顺序
        expert_features = []
        if defog_features:
            expert_features.append(defog_features[0])  # DEANet特征，用于去雾
        if derain_features:
            expert_features.append(derain_features[0])  # PReNet特征，用于去雨
        if desnow_features:
            expert_features.append(desnow_features[0])  # ConvIR特征，用于去雪
        # 门控网络计算权重，传入score
        expert_weights, feature_weights = self.moe_gate(x, score)
        # 特征融合
        fused_features = self.feature_fusion(expert_features, feature_weights)
        # 专家输出加权融合
        expert_outputs = torch.stack([defog_output, derain_output, desnow_output], dim=1)  # [B, 3, C, H, W]
        batch_size = expert_outputs.size(0)
        expert_weights = expert_weights.view(batch_size, 3, 1, 1, 1)  # 扩展维度以进行广播
        
        # 采用残差组合方式：moe_output = x - sum(w_i * (x - O_i))
        # 假设专家输出 O_i 为去噪后的图像，则 (x - O_i) 为专家估计的噪声/天气成分
        # 这种方式允许叠加多种天气成分的去除
        x_expanded = x.unsqueeze(1) # [B, 1, C, H, W]
        residuals = x_expanded - expert_outputs
        weighted_residuals = residuals * expert_weights
        moe_output = x - torch.sum(weighted_residuals, dim=1)

        # 最终重建（融合特征 + MoE输出）
        if fused_features is not None:
            # 调整特征尺寸以匹配输出
            if fused_features.size(2) != moe_output.size(2):
                fused_features = F.interpolate(fused_features, size=moe_output.shape[2:], mode='bilinear', align_corners=False)
            combined = torch.cat([moe_output, fused_features], dim=1)
        else:
            combined = moe_output
        # 最终重建
        # 使用重建网络对MoE的粗略输出进行残差精修，而不是重新预测
        refinement = self.reconstruction_net(combined)
        final_output = moe_output + refinement
        
        # 返回结果和中间信息（用于训练监控）
        return {
            'final_output': final_output,
            'moe_output': moe_output,
            'expert_weights': expert_weights.squeeze(),
            'feature_weights': feature_weights.squeeze(),
            'defog_output': defog_output,
            'derain_output': derain_output,
            'desnow_output': desnow_output,
            'fused_features': fused_features
        }