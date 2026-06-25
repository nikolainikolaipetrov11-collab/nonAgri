"""
    阶段二：生成阶段二伪标签
    核心任务：提取 4 维物理灵魂特征 -> BIC 动态聚类 -> 专家规则判决 -> 渲染全景看板
    model/nongrain_gmm.py
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import sys
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parents[1]))
from project_config import CONFIG

# 直接用 Dataset 底层接口加载数据
from dataset import PhenologyMAEDataset

# 设置中文字体与负号正常显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================================
# 统一配置
# ==========================================
NDVI_IDX = CONFIG.get("gmm", "ndvi_idx")
GLCM_IDX = CONFIG.get("gmm", "glcm_idx")
MIN_CLUSTERS = CONFIG.get("gmm", "min_clusters")
MAX_CLUSTERS = CONFIG.get("gmm", "max_clusters")

def run_nongrain_physical_gmm():
    logging.info("🌌 启动阶段二：非粮靶点【宏观物理特征】直聚类引擎")

    # 1. 路径与目标加载
    DATA_ROOT = CONFIG.date_out_dir
    TARGET_CSV = DATA_ROOT / "split_reports/02_Non_Grain_Anomalies.csv"
    OUTPUT_CSV = DATA_ROOT / "split_reports/03_Non_Grain_GMM_Final_Semantic.csv"
    OUTPUT_IMG = CONFIG.evaluation_dir / "Non_Grain_Physical_Clustering.png"
    OUTPUT_IMG.parent.mkdir(parents=True, exist_ok=True)

    if not TARGET_CSV.exists(): raise FileNotFoundError("❌ 未找到非粮化清单！")

    target_df = pd.read_csv(TARGET_CSV)
    target_df['parcel_id'] = target_df['parcel_id'].astype(str)
    target_ids_set = set(target_df['parcel_id'])
    logging.info(f"🎯 靶向锁定：已读取 {len(target_ids_set)} 块异常/非粮靶点。")

    # 2. 毫秒级提取绝对物理值
    logging.info("🧬 正在通过 Dataset 物理后门提取真实特征...")
    full_dataset = PhenologyMAEDataset(data_root=str(DATA_ROOT), mode='inference', mask_ratio=0.0)

    # 获取底座中匹配靶点 ID 的绝对索引
    meta_df = full_dataset.meta_data
    wide_id_col = full_dataset.wide_id_col
    valid_indices = meta_df.index[meta_df[wide_id_col].astype(str).isin(target_ids_set)].tolist()

    # 按照索引拉取 parcel_id 以保证顺序严格对齐
    nongrain_ids = meta_df.loc[valid_indices, wide_id_col].astype(str).tolist()

    # 直接调用底层原始数据，无需用 StandardScaler
    real_physics_matrix = full_dataset.get_raw_physical_features(valid_indices) # 形状: (N, 7, 47)

    # 3. 构建 4 维“灵魂宏观特征”
    logging.info("🔬 正在降维构建四大农学核心指标特征空间...")
    N = real_physics_matrix.shape[0]
    macro_features = np.zeros((N, 4))

    ndvi_matrix = real_physics_matrix[:, :, NDVI_IDX]
    glcm_matrix = real_physics_matrix[:, :, GLCM_IDX]

    macro_features[:, 0] = np.max(ndvi_matrix[:, 1:6], axis=1)  # 盛夏峰值绿度 (Peak NDVI)
    macro_features[:, 1] = ndvi_matrix[:, 0]                    # 开春返青绿度 (Spring NDVI)
    macro_features[:, 2] = np.std(ndvi_matrix, axis=1)          # 生命周期波动率 (NDVI Std)
    macro_features[:, 3] = np.mean(glcm_matrix, axis=1)         # 平均空间纹理 (Mean GLCM)

    # 仅对聚类空间做无量纲化
    macro_scaler = StandardScaler()
    macro_features_scaled = macro_scaler.fit_transform(macro_features)

    # 4. BIC 动态寻优
    logging.info("📡 启动宏观物理空间的 BIC 雷达...")
    bic_scores = []
    for k in range(MIN_CLUSTERS, MAX_CLUSTERS + 1):
        gmm_search = GaussianMixture(n_components=k, covariance_type='full', random_state=42, n_init=5)
        gmm_search.fit(macro_features_scaled)
        bic_scores.append(gmm_search.bic(macro_features_scaled))

    N_COMPONENTS = MIN_CLUSTERS + np.argmin(bic_scores)
    logging.info(f"🎯 雷达锁定！最优簇数确定为 {N_COMPONENTS} 个。")

    # 5. GMM 聚类
    gmm = GaussianMixture(n_components=N_COMPONENTS, covariance_type='full', random_state=42, n_init=10)
    gmm.fit(macro_features_scaled)
    cluster_labels = gmm.predict(macro_features_scaled)
    probs = gmm.predict_proba(macro_features_scaled)

    # 计算均值用于规则与绘图
    cluster_mean_real = np.zeros((N_COMPONENTS, 7, 47))
    for i in range(N_COMPONENTS):
        cluster_mean_real[i] = real_physics_matrix[cluster_labels == i].mean(axis=0)

    # 6. 专家规则引擎
    logging.info("启动专家规则引擎：锚定四大类与亚种...")
    cluster_semantics = {}
    category_counter = {"撂荒与裸地": 0, "果树与林地": 0, "经济作物": 0, "长势衰弱主粮": 0}
    global_glcm_mean = np.mean(macro_features[:, 3])

    for i in range(N_COMPONENTS):
        peak_ndvi = np.max(cluster_mean_real[i, 1:6, NDVI_IDX])
        spring_ndvi = cluster_mean_real[i, 0, NDVI_IDX]
        mean_glcm = np.mean(cluster_mean_real[i, :, GLCM_IDX])
        ndvi_std = np.std(cluster_mean_real[i, :, NDVI_IDX])

        if peak_ndvi < 0.55:
            base_label = "撂荒与裸地"
        elif spring_ndvi >= 0.18 and ndvi_std < 0.20 and mean_glcm > global_glcm_mean:
            base_label = "果树与林地"
        elif peak_ndvi >= 0.60 and mean_glcm < global_glcm_mean:
            base_label = "经济作物"
        else:
            base_label = "长势衰弱主粮"

        count = category_counter[base_label]
        category_counter[base_label] += 1
        final_label = base_label if count == 0 else f"{base_label}_亚种{count}"
        cluster_semantics[i] = final_label
        logging.info(f"   -> Cluster {i} 判定为: 【{final_label}】 (峰值NDVI: {peak_ndvi:.2f}, 纹理: {mean_glcm:.0f})")

    # 7. 结果落盘
    result_df = pd.DataFrame({
        'parcel_id': nongrain_ids,
        'GMM_Cluster_ID': cluster_labels,
        'GMM_Semantic_Label': [cluster_semantics[lbl] for lbl in cluster_labels],
        'GMM_Confidence': np.round(probs.max(axis=1), 4)
    })

    # 修复KeyError，确保合并的是 MSE_Score
    final_df = pd.merge(result_df, target_df[['parcel_id', 'MSE_Score']], on='parcel_id', how='left')
    final_df = final_df[['parcel_id', 'MSE_Score', 'GMM_Cluster_ID', 'GMM_Semantic_Label', 'GMM_Confidence']]
    final_df = final_df.sort_values(by='MSE_Score', ascending=False)

    final_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    logging.info(f"🎉 最终安全细分台账已保存至: {OUTPUT_CSV.name}")

    # 8. 看板渲染 (加入 Cluster ID 增强显示)
    logging.info("🎨 正在渲染物理特征聚类看板...")
    fig, axes = plt.subplots(2, 2, figsize=(28, 16))
    raw_colors = sns.color_palette("tab20", N_COMPONENTS)

    # 核心修改：生成带有 GMM_Cluster_ID 的绘图专用标签
    display_labels = {i: f"Cluster {i}: {cluster_semantics[i]}" for i in range(N_COMPONENTS)}
    color_dict = {display_labels[i]: raw_colors[i] for i in range(N_COMPONENTS)}
    months = ['4月', '5月', '6月', '7月', '8月', '9月', '10月']

    # 图1：条形图
    ax1 = axes[0, 0]
    counts = pd.Series(cluster_labels).value_counts()
    sorted_counts = pd.Series({display_labels[k]: v for k, v in counts.items()}).sort_values(ascending=True)

    sns.barplot(x=sorted_counts.values, y=sorted_counts.index, ax=ax1, palette=color_dict, hue=sorted_counts.index,
                legend=False)
    ax1.set_title('非粮类别数量分布', fontsize=20, pad=15)

    # 图2：t-SNE
    ax2 = axes[0, 1]
    tsne_res = TSNE(n_components=2, init='pca', random_state=42).fit_transform(macro_features_scaled)
    for i in range(N_COMPONENTS):
        mask = cluster_labels == i
        lbl = display_labels[i]
        ax2.scatter(tsne_res[mask, 0], tsne_res[mask, 1], color=color_dict[lbl], label=lbl, alpha=0.7, s=30)
    ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=14)
    ax2.set_title('4维宏观特征流形空间 t-SNE', fontsize=20, pad=15)

    # 图3：NDVI 曲线
    ax3 = axes[1, 0]
    for i in range(N_COMPONENTS):
        lbl = display_labels[i]
        ax3.plot(months, cluster_mean_real[i, :, NDVI_IDX], marker='o', lw=2.5, color=color_dict[lbl], label=lbl)
    ax3.set_title('NDVI平均生长曲线', fontsize=20, pad=15)
    ax3.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=14)
    ax3.grid(True, ls='--', alpha=0.6)

    # 图4：GLCM 曲线
    ax4 = axes[1, 1]
    for i in range(N_COMPONENTS):
        lbl = display_labels[i]
        ax4.plot(months, cluster_mean_real[i, :, GLCM_IDX], marker='s', lw=2.5, color=color_dict[lbl], ls='--', label=lbl)
    ax4.set_title('GLCM 空间纹理', fontsize=20, pad=15)
    ax4.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=14)
    ax4.grid(True, ls='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    run_nongrain_physical_gmm()
