"""Dynamic routing module snapshot; see paper_docs/release_status.md."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DRDBN(nn.Module):

    def __init__(self, in_channels=512, reduction=8, mode='dual'):
        super().__init__()
        self.in_channels = in_channels
        self.mode = mode
        self.f = -1
        self.spatial_branch = nn.Sequential(nn.Conv2d(in_channels, in_channels // 2, 3, padding=1, groups=8), nn.BatchNorm2d(in_channels // 2), nn.ReLU(inplace=True), nn.Conv2d(in_channels // 2, in_channels // 2, 3, padding=1, groups=8), nn.BatchNorm2d(in_channels // 2))
        self.context_branch = nn.Sequential(nn.Conv2d(in_channels, in_channels // 2, 3, padding=2, dilation=2, groups=8), nn.BatchNorm2d(in_channels // 2), nn.ReLU(inplace=True), nn.Conv2d(in_channels // 2, in_channels // 2, 3, padding=2, dilation=2, groups=8), nn.BatchNorm2d(in_channels // 2))
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, in_channels // reduction, 1), nn.ReLU(inplace=True), nn.Conv2d(in_channels // reduction, 2, 1), nn.Softmax(dim=1))
        self.enhance = nn.Sequential(nn.Conv2d(in_channels // 2, in_channels, 1), nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True))
        self.register_buffer('total_spatial', torch.tensor(0.0))
        self.register_buffer('total_context', torch.tensor(0.0))
        self.register_buffer('count', torch.tensor(0.0))
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.mode == 'dual':
            spatial_feat = self.spatial_branch(x)
            context_feat = self.context_branch(x)
            gate_weights = self.gate(x)
            w_spatial = gate_weights[:, 0:1, :, :]
            w_context = gate_weights[:, 1:2, :, :]
            fused = w_spatial * spatial_feat + w_context * context_feat
            if self.training:
                self.total_spatial += w_spatial.mean().item() * x.size(0)
                self.total_context += w_context.mean().item() * x.size(0)
                self.count += x.size(0)
        elif self.mode == 'spatial':
            fused = self.spatial_branch(x)
        elif self.mode == 'context':
            fused = self.context_branch(x)
        else:
            raise ValueError(f'未知的 DRDBN 模式: {self.mode}')
        out = self.enhance(fused) + x
        return out

    def get_routing_stats(self):
        if self.mode != 'dual':
            return {'spatial_avg': 1.0 if self.mode == 'spatial' else 0.0, 'context_avg': 1.0 if self.mode == 'context' else 0.0}
        if self.count > 0:
            return {'spatial_avg': (self.total_spatial / self.count).item(), 'context_avg': (self.total_context / self.count).item()}
        return {'spatial_avg': 0.5, 'context_avg': 0.5}

    def reset_stats(self):
        self.total_spatial.zero_()
        self.total_context.zero_()
        self.count.zero_()
