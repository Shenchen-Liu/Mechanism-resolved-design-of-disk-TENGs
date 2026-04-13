#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model definition for the physics-consistent multitask surrogate.

`PhysicsMultiTaskTransformer` predicts Qsc, invC_sum, and FOMS_direct from the
four dimensionless design parameters used throughout the repository.
"""

import torch
import torch.nn as nn


class PhysicsMultiTaskTransformer(nn.Module):
    """
    多任务 Physics-Aware Transformer 回归模型

    核心特性:
    - 将 4 个标量特征 (n, E, dd, hh) 映射为 4 个向量 (Embedding)
    - 使用 Multi-Head Self-Attention 自动发现物理耦合（No ID 架构）
    - 共享 backbone 输出 3 个独立预测头
    - 支持提取 Attention Weights 用于可解释性分析
    """

    def __init__(
        self,
        input_dim=4,
        embed_dim=128,
        nhead=4,
        num_layers=2,
        dropout=0.05,
        head_hidden=64,
    ):
        """
        Args:
            input_dim:   输入特征数（默认 4: n, E, dd, hh）
            embed_dim:   每个特征的嵌入维度
            nhead:       多头注意力头数
            num_layers:  Transformer 编码器层数
            dropout:     Dropout 比例
            head_hidden: 各输出头隐藏层维度
        """
        super(PhysicsMultiTaskTransformer, self).__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # Feature Embedding: 每个标量特征映射为一个向量
        self.feature_embedding = nn.Linear(1, embed_dim)

        # No ID 架构: 不使用静态位置编码，让 Attention 基于物理数值学习耦合
        # （与原版 PhysicsTransformer 保持一致的设计理念）

        # Transformer Encoder (共享 backbone)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # 共享表征维度
        shared_dim = embed_dim * input_dim

        # 输出头 1: Qsc_MACRS
        self.head_qsc = nn.Sequential(
            nn.Linear(shared_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

        # 输出头 2: invC_sum
        self.head_invc = nn.Sequential(
            nn.Linear(shared_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

        # 输出头 3: FOMS_direct
        self.head_foms = nn.Sequential(
            nn.Linear(shared_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier 均匀初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, return_weights=False):
        """
        前向传播

        Args:
            x: (batch_size, 4) 输入特征 [n, E, dd, hh]（归一化后）
            return_weights: 是否返回 attention weights

        Returns:
            dict: {
                'pred_qsc':         (batch_size, 1),
                'pred_invc_sum':    (batch_size, 1),
                'pred_foms_direct': (batch_size, 1),
            }
            attention_weights (可选): (batch_size, nhead, 4, 4)
        """
        batch_size = x.size(0)

        # Step 1: Feature Embedding
        # x: (batch, 4) -> (batch, 4, 1) -> (batch, 4, embed_dim)
        x_expanded = x.unsqueeze(-1)  # (batch, 4, 1)
        embeddings = self.feature_embedding(x_expanded)  # (batch, 4, embed_dim)

        # Step 2: Transformer Encoder
        if return_weights:
            encoded = embeddings
            attention_weights_list = []

            for layer in self.transformer_encoder.layers:
                attn_output, attn_weights = layer.self_attn(
                    encoded, encoded, encoded,
                    need_weights=True,
                    average_attn_weights=False,
                )
                attention_weights_list.append(attn_weights)

                encoded = layer.norm1(encoded + layer.dropout1(attn_output))
                ff_output = layer.linear2(
                    layer.dropout(layer.activation(layer.linear1(encoded)))
                )
                encoded = layer.norm2(encoded + layer.dropout2(ff_output))

            attention_weights = attention_weights_list[0]
        else:
            encoded = self.transformer_encoder(embeddings)
            attention_weights = None

        # Step 3: Flatten 共享表征
        shared_repr = encoded.reshape(batch_size, -1)  # (batch, 4*embed_dim)

        # Step 4: 三个独立输出头
        pred_qsc = self.head_qsc(shared_repr)           # (batch, 1)
        pred_invc_sum = self.head_invc(shared_repr)      # (batch, 1)
        pred_foms_direct = self.head_foms(shared_repr)   # (batch, 1)

        outputs = {
            "pred_qsc": pred_qsc,
            "pred_invc_sum": pred_invc_sum,
            "pred_foms_direct": pred_foms_direct,
        }

        if return_weights:
            return outputs, attention_weights
        else:
            return outputs
