"""
    阶段零：基于时序协同重构的抗噪学习 (架构师重构版)
    核心机制：Co-Teaching 交叉验证，提纯优质主粮时序规律。
    model/co_teaching_train.py
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch

from torch.utils.data import DataLoader, Subset
import pandas as pd
import logging
from pathlib import Path
from tqdm import tqdm

from dataset import PhenologyMAEDataset
from encoder import SerialLocalGlobalEncoder
from decoder import ConditionalTwinDecoder


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================================
# 配置中心
# ==========================================
class CoTeachingConfig:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MASK_RATIO = 0.25         # MAE 遮挡率
    NOISE_RATE = 0.35         # 预估伪标签最大噪音率
    EPOCH_GRADUAL = 30        # 达到最大遗忘率的预热 Epoch
    NUM_EPOCHS = 100
    BATCH_SIZE = 256
    LR = 1e-4
    DATA_ROOT = Path("E:/test/date/out")
    CHECKPOINT_DIR = Path("E:/test/model/checkpoints")

# ==========================================
# 抗噪交叉教学损失函数
# ==========================================
def loss_coteaching(pred_A, pred_B, targets, masks, forget_rate):
    """计算重构误差，独立排序并执行交叉更新。"""
    mse_A = ((pred_A - targets) ** 2) * masks
    mse_B = ((pred_B - targets) ** 2) * masks

    loss_A = mse_A.sum(dim=(1, 2)) / (masks.sum(dim=(1, 2)) + 1e-8)
    loss_B = mse_B.sum(dim=(1, 2)) / (masks.sum(dim=(1, 2)) + 1e-8)

    # 必须使用 detach() 斩断计算图，防止显存泄漏与图纠缠
    ind_A = torch.argsort(loss_A.detach())
    ind_B = torch.argsort(loss_B.detach())

    remember_rate = 1.0 - forget_rate
    num_remember = max(1, int(remember_rate * len(loss_A)))

    pure_ind_A = ind_A[:num_remember]
    pure_ind_B = ind_B[:num_remember]

    final_loss_A = loss_A[pure_ind_B].mean()
    final_loss_B = loss_B[pure_ind_A].mean()

    return final_loss_A, final_loss_B

# ==========================================
# 🚀 主管线
# ==========================================
def run_coteaching_pipeline():
    cfg = CoTeachingConfig()
    logging.info(f"🌌 启动 Co-Teaching 抗噪双子星训练引擎，计算设备: {cfg.DEVICE}")

    cfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = cfg.CHECKPOINT_DIR / "co_teaching_encoder_best.pth"

    # 1. 加载数据
    logging.info("正在加载含噪数据集并提取 K-Means 伪主粮 (包含干扰项)...")
    full_dataset = PhenologyMAEDataset(data_root=str(cfg.DATA_ROOT), mode='inference', mask_ratio=cfg.MASK_RATIO)
    all_labeled_df = pd.read_csv(cfg.DATA_ROOT / "sample_select_ms" / "All_Labeled_Parcels.csv")

    df_id_col = all_labeled_df.columns[0]
    pseudo_crops = all_labeled_df[all_labeled_df['Cluster_ID'].isin([0, 2])]
    valid_ids = set(pseudo_crops[df_id_col].astype(str).str.replace(r'\.0$', '', regex=True).str.strip())

    valid_indices = [idx for idx, pid in enumerate(full_dataset.meta_data[full_dataset.wide_id_col])
                     if str(pid).replace('.0', '').strip() in valid_ids]

    train_loader = DataLoader(Subset(full_dataset, valid_indices),
                              batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

    logging.info(f"-> 喂入双子星数据量: {len(valid_indices)} 块 (预设噪率峰值: {cfg.NOISE_RATE * 100}%)")

    # 2. 实例化双子星架构
    encoder_A = SerialLocalGlobalEncoder(input_dim=47, time_steps=7, d_model=128, nhead=4, num_layers=3).to(cfg.DEVICE)
    decoder_A = ConditionalTwinDecoder(embed_dim=128, out_dim=47, max_cluster_id=10).to(cfg.DEVICE)

    encoder_B = SerialLocalGlobalEncoder(input_dim=47, time_steps=7, d_model=128, nhead=4, num_layers=3).to(cfg.DEVICE)
    decoder_B = ConditionalTwinDecoder(embed_dim=128, out_dim=47, max_cluster_id=10).to(cfg.DEVICE)

    opt_A = torch.optim.AdamW(list(encoder_A.parameters()) + list(decoder_A.parameters()), lr=cfg.LR, weight_decay=1e-5)
    opt_B = torch.optim.AdamW(list(encoder_B.parameters()) + list(decoder_B.parameters()), lr=cfg.LR, weight_decay=1e-5)

    scaler_A = torch.amp.GradScaler('cuda')
    scaler_B = torch.amp.GradScaler('cuda')

    def get_forget_rate(epoch):
        return cfg.NOISE_RATE if epoch >= cfg.EPOCH_GRADUAL else (epoch / cfg.EPOCH_GRADUAL) * cfg.NOISE_RATE

    best_loss = float('inf')

    logging.info("⚔️ 交叉教学 (Co-Teaching) 引擎正式点火！")
    for epoch in range(cfg.NUM_EPOCHS):
        encoder_A.train(); decoder_A.train()
        encoder_B.train(); decoder_B.train()

        current_forget_rate = get_forget_rate(epoch)
        total_loss_A, total_loss_B = 0.0, 0.0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{cfg.NUM_EPOCHS}] (Drop {current_forget_rate * 100:.1f}%)")
        for masked_tensor, orig_tensor, mask_tensor, _, c_ids in pbar:
            inputs, targets, masks = masked_tensor.to(cfg.DEVICE), orig_tensor.to(cfg.DEVICE), mask_tensor.to(cfg.DEVICE)
            c_ids = c_ids.to(cfg.DEVICE)

            opt_A.zero_grad()
            opt_B.zero_grad()

            with torch.amp.autocast('cuda'):
                pred_A = decoder_A(encoder_A(inputs), c_ids)
                pred_B = decoder_B(encoder_B(inputs), c_ids)
                loss_A, loss_B = loss_coteaching(pred_A, pred_B, targets, masks, current_forget_rate)

            scaler_A.scale(loss_A).backward()
            scaler_A.step(opt_A)
            scaler_A.update()

            scaler_B.scale(loss_B).backward()
            scaler_B.step(opt_B)
            scaler_B.update()

            total_loss_A += loss_A.item()
            total_loss_B += loss_B.item()
            pbar.set_postfix({'Loss_A': f"{loss_A.item():.4f}", 'Loss_B': f"{loss_B.item():.4f}"})

        avg_loss_A = total_loss_A / len(train_loader)
        avg_loss_B = total_loss_B / len(train_loader)

        logging.info(f"Epoch {epoch + 1} | Drop: {current_forget_rate:.3f} | L_A: {avg_loss_A:.4f} | L_B: {avg_loss_B:.4f}")

        if avg_loss_A < best_loss:
            best_loss = avg_loss_A
            # ❗ 架构师 Fix: 救回 Decoder！将“连体婴儿”一起打包保存！
            torch.save({
                'encoder_state_dict': encoder_A.state_dict(),
                'decoder_state_dict': decoder_A.state_dict(),
            }, save_path)
            logging.info(f"   --> 🌟 优质主粮提取器 (Encoder+Decoder) 已完整保存 (Loss: {best_loss:.4f})")

    logging.info("🎉 Co-Teaching 降噪训练完美收官！")
    logging.info("下一步：我们将使用此网络对全县地块进行推理，获取 MSE 重构误差进行【可信置信度切割】！")

if __name__ == "__main__":
    run_coteaching_pipeline()