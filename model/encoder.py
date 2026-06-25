# model/encoder.py
# 编码器
import torch
import torch.nn as nn
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MSTCNBlock(nn.Module):
    """
    多尺度时序卷积网络 (Multi-Scale Temporal Convolutional Network)
    核心任务：将原始 47 维特征中的“相邻月份斜率/突变”提取为高阶特征向量
    """

    def __init__(self, in_channels=47, out_channels=128, kernel_sizes=(3, 5)):
        super().__init__()

        # 动态推导 Padding，消灭魔法数字，彻底解耦 Kernel Size 变化风险
        self.branch_short = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_sizes[0],
            padding=kernel_sizes[0] // 2,
            padding_mode='replicate'
        )

        self.branch_long = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_sizes[1],
            padding=kernel_sizes[1] // 2,
            padding_mode='replicate'
        )

        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        # x: (Batch, 7, 47) -> x_t: (Batch, 47, 7)
        x_t = x.transpose(1, 2)

        out_short = self.branch_short(x_t)
        out_long = self.branch_long(x_t)

        # 多尺度融合并转回 (Batch, 7, out_channels)
        out = (out_short + out_long).transpose(1, 2)

        out = self.layer_norm(out)
        out = self.activation(out)
        return out


class SerialLocalGlobalEncoder(nn.Module):
    """
    Twin-Expert MAE 的前端编码器 (串行架构)
    流程: 原始输入 -> MS-TCN -> 注入位置编码 -> Pos-Dropout -> Transformer -> 隐空间特征
    """

    def __init__(self, input_dim=47, time_steps=7, d_model=128, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # 1. 局部特征提取器
        self.ms_tcn = MSTCNBlock(in_channels=input_dim, out_channels=d_model)

        # 2. 可学习的时序位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, time_steps, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ❗ 架构师 Fix: 必须在注入位置编码后进行 Dropout，防止短序列灾难性过拟合！
        self.pos_drop = nn.Dropout(p=dropout)

        # 3. 全局交互 Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f"❌ 编码器期望输入 3D 张量 (Batch, {x.shape[1]}, {x.shape[2]})，但得到 {x.dim()}D 张量。")

        # MS-TCN
        x = self.ms_tcn(x)

        # 注入位置信息 + Dropout
        x = x + self.pos_embed.to(x.device)
        x = self.pos_drop(x)

        # Transformer + Norm
        x = self.transformer(x)
        x = self.norm(x)

        return x


# ==========================================
# 内存级防弹测试模块
# ==========================================
if __name__ == "__main__":
    logging.info("🚀 开始内存级 Encoder 架构连通性测试...")

    dummy_batch = torch.randn(16, 7, 47)
    encoder = SerialLocalGlobalEncoder(input_dim=47, time_steps=7, d_model=128, nhead=4, num_layers=3)

    try:
        output_features = encoder(dummy_batch)
        logging.info(">>> ✅ Encoder 前向传播测试成功！")
        logging.info(f"输入张量形状: {dummy_batch.shape} -> (Batch, TimeSteps, InputDim)")
        logging.info(f"输出隐特征形状: {output_features.shape} -> 此张量将专供 Decoder 计算 MSE 重构误差！")
        logging.info("设备与梯度安全校验通过。")
    except Exception as e:
        logging.error(f"❌ 测试失败，报错信息: {e}")