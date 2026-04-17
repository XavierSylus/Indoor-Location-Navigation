"""
IMU 传感器相对位移预测模型 (imu_delta_model.py)
================================================
架构：Conv1d + GRU + MLP  (参考 Kaggle Indoor第1名方案)

核心思路：
  - CNN 分支：从 12 通道 IMU 原始时序中提取局部运动模式（步态脉冲、转向特征）
  - GRU 分支：在 CNN 特征之上建模长距离依赖（航向累积）
  - MLP 分支：融入辅助标量特征（时间差、步数估计、楼层信息等）
  - 融合头：拼接后回归 Delta X/Y

输入：
  - imu_seq: (batch, 12, 100)  —— 12 通道 × 100 时间步
  - aux_feat: (batch, aux_dim) —— 辅助标量特征

输出：
  - delta_xy: (batch, 2) —— 预测的 (Δx, Δy) 单位米
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1dBlock(nn.Module):
    """Conv1d + BatchNorm + GELU + Dropout"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.bn(self.conv(x))))


class IMUDeltaNet(nn.Module):
    """
    Conv1d + GRU + MLP 融合网络，预测连续 waypoint 间的相对位移

    参数均从配置字典读取，禁止硬编码：
      model_cfg = config['model']
    """

    def __init__(
        self,
        imu_channels: int = 12,
        n_time_steps: int = 100,
        aux_dim: int = 10,
        cnn_channels: list = None,
        cnn_kernel_sizes: list = None,
        cnn_dropout: float = 0.2,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        gru_dropout: float = 0.3,
        gru_bidirectional: bool = True,
        mlp_aux_hidden: list = None,
        mlp_aux_dropout: float = 0.2,
        head_hidden: list = None,
        head_dropout: float = 0.3,
        output_dim: int = 2,
    ):
        super().__init__()

        if cnn_channels is None:
            cnn_channels = [imu_channels, 64, 128, 256]
        if cnn_kernel_sizes is None:
            cnn_kernel_sizes = [7, 5, 3]
        if mlp_aux_hidden is None:
            mlp_aux_hidden = [32, 64]
        if head_hidden is None:
            head_hidden = [256, 128]

        # ── CNN 分支 ──
        cnn_layers = []
        for i in range(len(cnn_kernel_sizes)):
            cnn_layers.append(
                Conv1dBlock(cnn_channels[i], cnn_channels[i + 1], cnn_kernel_sizes[i], cnn_dropout)
            )
        self.cnn = nn.Sequential(*cnn_layers)

        # ── GRU 分支 ──
        self.gru = nn.GRU(
            input_size=cnn_channels[-1],
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=gru_dropout if gru_layers > 1 else 0.0,
            bidirectional=gru_bidirectional,
        )
        gru_out_dim = gru_hidden * (2 if gru_bidirectional else 1)

        # ── MLP 辅助分支 ──
        mlp_layers = []
        prev_dim = aux_dim
        for h in mlp_aux_hidden:
            mlp_layers.extend([
                nn.Linear(prev_dim, h),
                nn.GELU(),
                nn.Dropout(mlp_aux_dropout),
            ])
            prev_dim = h
        self.mlp_aux = nn.Sequential(*mlp_layers)
        mlp_out_dim = mlp_aux_hidden[-1] if mlp_aux_hidden else aux_dim

        # ── 融合头 ──
        head_layers = []
        prev_dim = gru_out_dim + mlp_out_dim
        for h in head_hidden:
            head_layers.extend([
                nn.Linear(prev_dim, h),
                nn.GELU(),
                nn.Dropout(head_dropout),
            ])
            prev_dim = h
        head_layers.append(nn.Linear(prev_dim, output_dim))
        self.head = nn.Sequential(*head_layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        imu_seq: torch.Tensor,
        aux_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        imu_seq  : (B, C=12, T=100) 分箱后的 IMU 时序
        aux_feat : (B, aux_dim) 辅助标量特征

        Returns
        -------
        delta_xy : (B, 2) 预测 Δx, Δy
        """
        # CNN: (B, C, T) → (B, C', T)
        cnn_out = self.cnn(imu_seq)

        # GRU: (B, C', T) → (B, T, C') 需要 transpose
        gru_in = cnn_out.permute(0, 2, 1)  # (B, T, C')
        gru_out, _ = self.gru(gru_in)      # (B, T, gru_out_dim)

        # 取最后时间步的输出 + 全局平均池化，取均值融合
        gru_last = gru_out[:, -1, :]                     # (B, gru_out_dim)
        gru_mean = gru_out.mean(dim=1)                   # (B, gru_out_dim)
        gru_feat = (gru_last + gru_mean) / 2.0           # (B, gru_out_dim)

        # MLP 辅助
        aux_out = self.mlp_aux(aux_feat)                 # (B, mlp_out_dim)

        # 拼接 + 融合头
        fused = torch.cat([gru_feat, aux_out], dim=-1)   # (B, gru_out_dim + mlp_out_dim)
        delta_xy = self.head(fused)                      # (B, 2)

        return delta_xy

    @classmethod
    def from_config(cls, config: dict) -> "IMUDeltaNet":
        """从 YAML 配置字典构建模型"""
        data_cfg = config['data']
        model_cfg = config['model']

        return cls(
            imu_channels=data_cfg['imu_channels'],
            n_time_steps=data_cfg['n_time_steps'],
            aux_dim=data_cfg['aux_feature_dim'],
            cnn_channels=model_cfg['cnn']['channels'],
            cnn_kernel_sizes=model_cfg['cnn']['kernel_sizes'],
            cnn_dropout=model_cfg['cnn']['dropout'],
            gru_hidden=model_cfg['gru']['hidden_size'],
            gru_layers=model_cfg['gru']['num_layers'],
            gru_dropout=model_cfg['gru']['dropout'],
            gru_bidirectional=model_cfg['gru']['bidirectional'],
            mlp_aux_hidden=model_cfg['mlp_aux']['hidden_sizes'],
            mlp_aux_dropout=model_cfg['mlp_aux']['dropout'],
            head_hidden=model_cfg['head']['hidden_sizes'],
            head_dropout=model_cfg['head']['dropout'],
            output_dim=model_cfg['head']['output_dim'],
        )


class DeltaDistanceLoss(nn.Module):
    """
    自定义复合损失函数（参考 Kaggle Indoor 第5名方案）。

    同时约束三个维度：
      1. Delta MAE   — 位移向量 (dx, dy) 的绝对误差
      2. Distance MAE — 由 Delta 推导的行走距离误差
      3. Direction    — 行走方向的余弦相似度偏差

    所有计算在 AMP autocast 下安全运行，无额外计算图残留。
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.3,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pred   : (B, 2)  预测的 [Δx, Δy]
        target : (B, 2)  真实的 [Δx, Δy]
        """
        # (1) Delta MAE
        delta_loss = F.l1_loss(pred, target)

        # (2) Distance MAE
        dist_pred = torch.sqrt(pred[:, 0] ** 2 + pred[:, 1] ** 2 + self.eps)
        dist_true = torch.sqrt(target[:, 0] ** 2 + target[:, 1] ** 2 + self.eps)
        dist_loss = F.l1_loss(dist_pred, dist_true)

        # (3) Direction cosine loss（仅对有意义位移 > 0.1m）
        cos_sim = F.cosine_similarity(pred, target, dim=1, eps=self.eps)
        mask = dist_true > 0.1
        if mask.any():
            dir_loss = (1.0 - cos_sim[mask]).mean()
        else:
            dir_loss = pred.new_tensor(0.0)

        return self.alpha * delta_loss + self.beta * dist_loss + self.gamma * dir_loss
