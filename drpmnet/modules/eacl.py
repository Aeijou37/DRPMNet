"""Boundary contrastive loss snapshot; see paper_docs/release_status.md."""

import logging

LOGGER = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F

class EACL(nn.Module):

    def __init__(self, in_channels=256, embed_dim=128, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.max_edge_pts = 200
        self._initialized = False
        self.projector = nn.Sequential(nn.Conv2d(in_channels, embed_dim, 1), nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True))
        self.predictor = nn.Sequential(nn.Conv2d(embed_dim, embed_dim, 1), nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True), nn.Conv2d(embed_dim, embed_dim, 1))
        self.edge_extractor = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, 1, 1), nn.Sigmoid())
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

    def forward(self, strong_feat, weak_feat, edge_mask=None):
        if isinstance(strong_feat, (tuple, list)):
            strong_feat = strong_feat[0]
        if isinstance(weak_feat, (tuple, list)):
            weak_feat = weak_feat[0]
        if strong_feat.dim() == 3:
            strong_feat = strong_feat.unsqueeze(0)
        if weak_feat.dim() == 3:
            weak_feat = weak_feat.unsqueeze(0)
        if edge_mask is None:
            edge_mask = self.edge_extractor(strong_feat)
        edge_mask_binary = (edge_mask > 0.5).float()
        z_strong = self.projector(strong_feat)
        z_weak = self.projector(weak_feat)
        p_strong = self.predictor(z_strong)
        batch_size = strong_feat.shape[0]
        total_loss = torch.tensor(0.0, device=strong_feat.device, dtype=torch.float32)
        valid_samples = 0
        for b in range(batch_size):
            edge_positions = edge_mask_binary[b].squeeze().nonzero(as_tuple=True)
            n_pts = len(edge_positions[0])
            if n_pts < 10:
                continue
            if n_pts > self.max_edge_pts:
                idx = torch.randperm(n_pts, device=strong_feat.device)[:self.max_edge_pts]
                edge_positions = (edge_positions[0][idx], edge_positions[1][idx])
            p_s = p_strong[b, :, edge_positions[0], edge_positions[1]]
            z_w = z_weak[b, :, edge_positions[0], edge_positions[1]].detach()
            p_s = p_s.float()
            z_w = z_w.float()
            p_s_norm = F.normalize(p_s, dim=0, eps=1e-08)
            z_w_norm = F.normalize(z_w, dim=0, eps=1e-08)
            sim_matrix = torch.matmul(p_s_norm.T, z_w_norm) / self.temperature
            labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
            loss = F.cross_entropy(sim_matrix, labels, reduction='mean')
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            total_loss = total_loss + loss
            valid_samples += 1
        if valid_samples == 0:
            dummy_loss = (p_strong * 0.0).sum() + (edge_mask * 0.0).sum()
            return dummy_loss
        return total_loss / valid_samples + (edge_mask * 0.0).sum()

    def initialize(self, num_batches=10):
        LOGGER.info('初始化 EACL 模块...')
        device = next(self.parameters()).device
        with torch.no_grad():
            for i in range(num_batches):
                dummy = torch.randn(2, 256, 80, 80, device=device)
                _ = self.edge_extractor(dummy)
                _ = self.predictor(self.projector(dummy))
                if i % 5 == 0:
                    LOGGER.info(f'  预热批次 {i + 1}/{num_batches}')
        self._initialized = True
        LOGGER.info('✅ EACL 初始化完成')
