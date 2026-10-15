"""Progressive fusion module snapshot; see paper_docs/release_status.md."""

import logging

LOGGER = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossScaleAttention(nn.Module):

    def __init__(self, low_channels, high_channels):
        super().__init__()
        self.align_conv = nn.Conv2d(high_channels, low_channels, 1)
        self.local_aggregator = nn.Sequential(nn.Conv2d(low_channels * 2, low_channels, kernel_size=1, bias=False), nn.BatchNorm2d(low_channels), nn.ReLU(inplace=True), nn.Conv2d(low_channels, low_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(low_channels), nn.ReLU(inplace=True))
        self.concat_reduce = nn.Sequential(nn.Conv2d(low_channels * 2, low_channels, kernel_size=1, bias=False), nn.BatchNorm2d(low_channels), nn.ReLU(inplace=True))

    def forward(self, low_feat, high_feat):
        high_feat_up = F.interpolate(self.align_conv(high_feat), size=low_feat.shape[2:], mode='bilinear', align_corners=False)
        fused_feat = torch.cat([low_feat, high_feat_up], dim=1)
        fused = self.local_aggregator(fused_feat)
        return fused

class PMAF(nn.Module):

    def __init__(self, in_channels_list=[256, 512, 512], out_channels=256):
        super().__init__()
        self.f = -1
        self.edge_loss_weight = 0.1
        LOGGER.info(f'PMAF初始化: 输入通道 {in_channels_list} -> 输出通道 {out_channels}')
        self.lateral_convs = nn.ModuleList([nn.Conv2d(ch, out_channels, 1) for ch in in_channels_list])
        self.fusion_stages = nn.ModuleList([CrossScaleAttention(out_channels, out_channels), CrossScaleAttention(out_channels, out_channels)])
        self.edge_heads = nn.ModuleList([nn.Sequential(nn.Conv2d(out_channels, out_channels // 4, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(out_channels // 4, 1, 1)) for _ in range(len(in_channels_list))])

    def forward(self, features, gt_edges=None):
        feats = [conv(f) for conv, f in zip(self.lateral_convs, features)]
        p3, p4, p5 = feats
        edge_losses = []
        p4_enhanced = self.fusion_stages[1](p4, p5)
        if self.training and gt_edges is not None and (len(gt_edges) > 2):
            if gt_edges[2] is not None:
                target_size = p5.shape[2:]
                gt_edge_resized = F.interpolate(gt_edges[2], size=target_size, mode='bilinear', align_corners=False)
                edge_pred_p5 = self.edge_heads[2](p5)
                edge_loss_p5 = F.binary_cross_entropy_with_logits(edge_pred_p5, gt_edge_resized)
                edge_losses.append(edge_loss_p5 * self.edge_loss_weight)
            else:
                edge_losses.append((self.edge_heads[2](p5) * 0.0).sum())
        p3_enhanced = self.fusion_stages[0](p3, p4_enhanced)
        if self.training and gt_edges is not None:
            if len(gt_edges) > 1:
                if gt_edges[1] is not None:
                    target_size = p4_enhanced.shape[2:]
                    gt_edge_resized = F.interpolate(gt_edges[1], size=target_size, mode='bilinear', align_corners=False)
                    edge_pred_p4 = self.edge_heads[1](p4_enhanced)
                    edge_loss_p4 = F.binary_cross_entropy_with_logits(edge_pred_p4, gt_edge_resized)
                    edge_losses.append(edge_loss_p4 * self.edge_loss_weight)
                else:
                    edge_losses.append((self.edge_heads[1](p4_enhanced) * 0.0).sum())
            if len(gt_edges) > 0:
                if gt_edges[0] is not None:
                    target_size = p3_enhanced.shape[2:]
                    gt_edge_resized = F.interpolate(gt_edges[0], size=target_size, mode='bilinear', align_corners=False)
                    edge_pred_p3 = self.edge_heads[0](p3_enhanced)
                    edge_loss_p3 = F.binary_cross_entropy_with_logits(edge_pred_p3, gt_edge_resized)
                    edge_losses.append(edge_loss_p3 * self.edge_loss_weight)
                else:
                    edge_losses.append((self.edge_heads[0](p3_enhanced) * 0.0).sum())
        output_features = [p3_enhanced, p4_enhanced, p5]
        if self.training:
            return (output_features, edge_losses)
        return output_features

def test_pmaf_output():
    LOGGER.info('\n测试PMAF输出格式...')
    pmaf = PMAF()
    pmaf.eval()
    import torch
    p3 = torch.randn(2, 256, 80, 80)
    p4 = torch.randn(2, 512, 40, 40)
    p5 = torch.randn(2, 512, 20, 20)
    features = [p3, p4, p5]
    output = pmaf(features)
    assert isinstance(output, list), f'推理模式应返回list，实际{type(output)}'
    assert len(output) == 3, f'应返回3个特征，实际{len(output)}'
    LOGGER.info(f'✅ 推理模式: 返回{len(output)}个特征')
    pmaf.train()
    gt_edges = [torch.rand(2, 1, 80, 80), torch.rand(2, 1, 40, 40), torch.rand(2, 1, 20, 20)]
    output_train = pmaf(features, gt_edges)
    assert isinstance(output_train, tuple), f'训练模式应返回tuple，实际{type(output_train)}'
    LOGGER.info(f'✅ 训练模式: 返回特征+{len(output_train[1])}个边缘损失')
    LOGGER.info('=' * 60)
if __name__ == '__main__':
    test_pmaf_output()
