"""
    阶段一：两阶段混合路由管道
    核心任务：算 MSE -> KDE 自动寻拐点 -> Sigmoid 异常置信度映射 -> 输出绝对物理隔离台账
    model/hybrid_anomaly_routing.py
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from tqdm import tqdm
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parents[1]))
from project_config import CONFIG

from dataset import PhenologyMAEDataset
from encoder import SerialLocalGlobalEncoder
from decoder import ConditionalTwinDecoder


plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def auto_find_elbow_threshold(mse_array):
    logging.info("🤖 启动 Elbow/Knee Point 动态自适应寻拐点算法...")
    clean_mse = mse_array[mse_array < np.percentile(mse_array, 98)]

    # 最多只抽取 10000 个代表点去拟合曲线
    if len(clean_mse) > 10000:
        kde_input = np.random.choice(clean_mse, size=10000, replace=False)
    else:
        kde_input = clean_mse
    kde = gaussian_kde(kde_input, bw_method='scott')

    grid = np.linspace(np.min(clean_mse), np.max(clean_mse), 500)
    density = kde(grid)

    peak_idx = np.argmax(density)
    tail_grid = grid[peak_idx:]
    tail_density = density[peak_idx:]

    p1 = np.array([tail_grid[0], tail_density[0]])
    p2 = np.array([tail_grid[-1], tail_density[-1]])

    distances = []
    line_norm = np.linalg.norm(p2 - p1) + 1e-8

    for x, y in zip(tail_grid, tail_density):
        p3 = np.array([x, y])
        dist = np.abs(np.cross(p2 - p1, p3 - p1)) / line_norm
        distances.append(dist)

    optimal_threshold = tail_grid[np.argmax(distances)]
    logging.info(f"🎯 算法自动锁定 L 型曲线完美拐点阈值: {optimal_threshold:.4f}")
    return optimal_threshold


# ==========================================
# 分段非对称自适应置信度映射引擎
# ==========================================
def calculate_confidence_scores(mse_array, threshold):
    logging.info("📈 启动【分段非对称自解释】置信度映射引擎...")

    # 初始化输出概率阵列
    p_anomaly = np.zeros_like(mse_array)

    # 提取边界特征，消灭硬编码
    min_mse = np.min(mse_array)
    safe_mask = mse_array <= threshold
    anomaly_mask = mse_array > threshold

    # ----------------------------------------------------
    # 通道一：安全主粮侧 (MSE <= Threshold)
    # 物理心智：强迫最完美的地块(min_mse)置信度收敛到 0.99 (即 p_anomaly = 0.01)
    # 逆推公式：1 / (1 + e^(-alpha * (min_mse - T))) = 0.01  =>  alpha = ln(99) / (T - min_mse)
    # ----------------------------------------------------
    if safe_mask.any():
        delta_safe = threshold - min_mse
        # 防止极端情况下分母为 0 导致崩溃
        if delta_safe < 1e-6:
            alpha_safe = 10.0
        else:
            alpha_safe = np.log(99.0) / delta_safe

        # 局部 Sigmoid 映射
        p_anomaly[safe_mask] = 1.0 / (1.0 + np.exp(-alpha_safe * (mse_array[safe_mask] - threshold)))

    # ----------------------------------------------------
    # 通道二：疑似非粮侧 (MSE > Threshold)
    # 物理心智：摆脱主粮池的引力，只根据异常子集内部的变异系数调整长尾舒展度
    # ----------------------------------------------------
    if anomaly_mask.any():
        # 只计算异常点内部的标准差，实现局部特征解耦
        std_anomaly = np.std(mse_array[anomaly_mask])
        alpha_anomaly = 3.0 / (std_anomaly + 1e-8)

        # 局部 Sigmoid 映射
        p_anomaly[anomaly_mask] = 1.0 / (1.0 + np.exp(-alpha_anomaly * (mse_array[anomaly_mask] - threshold)))

    return np.round(p_anomaly, 4)

def run_hybrid_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"🚀 启动异常初筛路由引擎 (可信AI版)，计算设备: {device}")

    DATA_ROOT = CONFIG.date_out_dir
    CO_TEACHING_CKPT = CONFIG.checkpoint_dir / "co_teaching_encoder_best.pth"

    REPORT_DIR = DATA_ROOT / "split_reports"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SAFE_CSV = REPORT_DIR / "01_Safe_Grain_Parcels.csv"
    ANOMALY_CSV = REPORT_DIR / "02_Non_Grain_Anomalies.csv"
    OUTPUT_IMG = CONFIG.evaluation_dir / "MSE_Distribution.png"
    OUTPUT_IMG.parent.mkdir(parents=True, exist_ok=True)

    full_dataset = PhenologyMAEDataset(data_root=str(DATA_ROOT), mode='inference', mask_ratio=0.0)
    full_loader = DataLoader(full_dataset, batch_size=CONFIG.get("inference", "batch_size"), shuffle=False, num_workers=0)

    encoder = SerialLocalGlobalEncoder(input_dim=47, time_steps=7, d_model=128, nhead=4, num_layers=3).to(device)
    decoder = ConditionalTwinDecoder(embed_dim=128, out_dim=47, max_cluster_id=10).to(device)

    # 从同一个包中解包恢复
    if not CO_TEACHING_CKPT.exists():
        raise FileNotFoundError(f"❌ 未找到最新权重 {CO_TEACHING_CKPT}，请先运行 co_teaching_train.py！")

    checkpoint = torch.load(CO_TEACHING_CKPT, map_location=device, weights_only=False)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    encoder.eval(); decoder.eval()

    all_parcel_ids = []
    all_mse_errors = []

    TEXTURE_START_IDX = CONFIG.get("inference", "texture_start_idx")
    TEXTURE_END_IDX = CONFIG.get("inference", "texture_end_idx")

    logging.info("⚖️ 正在进行全域扫描，计算纯正的 MAE 重构误差 (MSE)...")
    with torch.no_grad():
        for _, orig_tensor, _, parcel_ids, cluster_ids in tqdm(full_loader, desc="计算 MSE"):
            inputs = orig_tensor.to(device)
            c_ids = cluster_ids.to(device)

            # 手术刀屏蔽纹理
            inputs_pure = inputs.clone()
            inputs_pure[:, :, TEXTURE_START_IDX:TEXTURE_END_IDX] = 0.0

            features = encoder(inputs_pure)
            reconstructed = decoder(features, c_ids)

            mse = ((reconstructed - inputs) ** 2).mean(dim=(1, 2))

            all_mse_errors.extend(mse.cpu().numpy())
            all_parcel_ids.extend(parcel_ids)

    mse_array = np.array(all_mse_errors)

    # 1. 寻找拐点
    TEMP_THRESHOLD = auto_find_elbow_threshold(mse_array)

    # 2. 计算全局异常概率置信度 P_anomaly
    p_anomaly_array = calculate_confidence_scores(mse_array, TEMP_THRESHOLD)

    # 渲染分布图
    plot_max = np.percentile(mse_array, 98)
    clean_mse_for_plot = mse_array[mse_array <= plot_max]

    plt.figure(figsize=(12, 6))
    sns.histplot(clean_mse_for_plot, bins=100, kde=True, color="royalblue")
    plt.title('MAE 重构误差 (MSE) 分布与动态阈值切割', fontsize=16)
    plt.xlabel('Reconstruction Error (MSE)', fontsize=14)
    plt.ylabel('地块数量', fontsize=14)
    plt.axvline(x=TEMP_THRESHOLD, color='red', linestyle='--', linewidth=2, label=f'自动拐点拦截线: {TEMP_THRESHOLD:.4f}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 纯净路由分流落盘
    safe_grain_mask = mse_array <= TEMP_THRESHOLD
    anomaly_mask = mse_array > TEMP_THRESHOLD

    df_all = pd.DataFrame({
        'parcel_id': all_parcel_ids,
        'MSE_Score': np.round(mse_array, 4),
        'P_Anomaly': p_anomaly_array
    })

    # [安全池] 记录主粮置信度
    df_safe = df_all[safe_grain_mask].copy()
    df_safe['Grain_Confidence'] = np.round(1.0 - df_safe['P_Anomaly'], 4)
    df_safe = df_safe.drop(columns=['P_Anomaly']).sort_values(by='Grain_Confidence', ascending=False) # 置信度降序

    # [靶点池] 记录异常置信度
    df_anomaly = df_all[anomaly_mask].copy()
    df_anomaly = df_anomaly.rename(columns={'P_Anomaly': 'Anomaly_Confidence'}).sort_values(by='Anomaly_Confidence', ascending=False)

    df_safe.to_csv(SAFE_CSV, index=False)
    df_anomaly.to_csv(ANOMALY_CSV, index=False)

    logging.info(f"🌾 路由成功：{len(df_safe)} 块安全主粮已存入 -> {SAFE_CSV.name} (携带 Grain_Confidence)")
    logging.info(f"🚨 拦截成功：{len(df_anomaly)} 块疑似非粮靶点已存入 -> {ANOMALY_CSV.name} (携带 Anomaly_Confidence)")
    logging.info("🎉 第一阶段概率化改造完成！请检查输出的 CSV 台账，确认置信度梯度是否符合预期！")

if __name__ == "__main__":
    run_hybrid_pipeline()
