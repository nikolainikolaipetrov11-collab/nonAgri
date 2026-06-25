# model/decoder.py
# 解码器
import torch
import torch.nn as nn
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ConditionalTwinDecoder(nn.Module):
    """
    Twin-Expert MAE 解码器 (条件注入模式)
    核心逻辑：接收 Encoder 隐特征，安全注入 Cluster_ID 条件，重构 47 维特征。
    """

    def __init__(self, embed_dim=128, out_dim=47, max_cluster_id=10, num_layers=2, nhead=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.max_cluster_id = max_cluster_id

        # 1. 专家条件嵌入层
        self.condition_embed = nn.Embedding(num_embeddings=max_cluster_id, embedding_dim=embed_dim)

        # 引入条件注入后的归一化层，防止特征分布漂移引发的梯度震荡
        self.norm_cond = nn.LayerNorm(embed_dim)

        # 2. 解码器主干 (轻量级 2 层即可，工具人无需太深)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)

        # 3. 重构投影头
        self.reconstruction_head = nn.Linear(embed_dim, out_dim)

    def forward(self, x, cluster_ids):
        if x.dim() != 3:
            raise ValueError(f"❌ Decoder 期望输入特征形状 (Batch, 7, {self.embed_dim})，但得到 {x.dim()}D 张量。")
        if cluster_ids.dim() != 1 or cluster_ids.size(0) != x.size(0):
            raise ValueError(f"❌ cluster_ids 必须是与 Batch 匹配的 1D 张量，当前形状: {cluster_ids.shape}")

        cluster_ids = cluster_ids.long()

        #  防弹级安全锁，强制截断超界 ID，彻底消灭 CUDA Assert 炸弹！
        cluster_ids = torch.clamp(cluster_ids, min=0, max=self.max_cluster_id - 1)

        # 提取条件向量并广播
        cond = self.condition_embed(cluster_ids)
        x = x + cond.unsqueeze(1)

        # 稳定条件注入后的特征分布
        x = self.norm_cond(x)

        # 解码与重构
        x = self.transformer_decoder(x)
        reconstructed_x = self.reconstruction_head(x)

        return reconstructed_x


# ==========================================
# 内存级防弹测试模块
# ==========================================
if __name__ == "__main__":
    logging.info("🚀 开始内存级 Decoder 条件注入架构测试...")

    dummy_encoded = torch.randn(16, 7, 128)

    # 模拟包含脏数据的 ID (包含 -1 和越界 99)
    dummy_ids = torch.tensor([1, 3, -1, 99] * 4)

    decoder = ConditionalTwinDecoder(embed_dim=128, out_dim=47, max_cluster_id=10)

    try:
        output_features = decoder(dummy_encoded, dummy_ids)
        logging.info(">>> ✅ Decoder 前向传播测试成功！")
        logging.info(">>> 🛡️ 越界脏数据 ID 已被成功拦截并安全截断！")
        logging.info(f"重构输出形状: {output_features.shape} (严格对齐 47 维底座)")
    except Exception as e:
        logging.error(f"❌ 测试失败，报错: {e}")