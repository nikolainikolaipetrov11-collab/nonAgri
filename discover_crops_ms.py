"""
    用于阶段一非粮化正样本自举
    discover_crops_ms.py
"""
import os
import glob
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 0. 基础路径与专属收纳仓配置
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, 'date', 'out')
INPUT_CSV = os.path.join(OUT_DIR, 'Dynamic_TimeSeries_WideTable.csv')

# 建立多维聚类专属输出收纳仓
SELECT_DIR = os.path.join(OUT_DIR, 'sample_select_ms')
os.makedirs(SELECT_DIR, exist_ok=True)


def perform_multidimensional_clustering():
    print("=" * 70)
    print("启动升级版：多维空间物候协同聚类 与 前30%核心样本提纯引擎")
    print("=" * 70)

    if not os.path.exists(INPUT_CSV):
        print(f"  [致命错误] 未找到动态特征宽表: {INPUT_CSV}")
        return

    print("1. 正在加载全县地块特征大宽表并抽取立体特征...")
    df = pd.read_csv(INPUT_CSV)

    # 动态感知并提取三大核心协同特征列
    ndvi_cols = [col for col in df.columns if col.startswith('NDVI_mean_M')]
    gcvi_cols = [col for col in df.columns if col.startswith('GCVI_mean_M')]
    glcm_cols = [col for col in df.columns if col.startswith('GLCM_Contrast_M')]

    if not ndvi_cols or not gcvi_cols or not glcm_cols:
        print("  [致命错误] 特征宽表中缺少 NDVI, GCVI 或 GLCM 纹理特征，请检查总装管线！")
        return

    # 严格按照月份数字排序，确保时间轴完美对齐
    ndvi_cols.sort(key=lambda x: int(x.split('_M')[-1]))
    gcvi_cols.sort(key=lambda x: int(x.split('_M')[-1]))
    glcm_cols.sort(key=lambda x: int(x.split('_M')[-1]))

    available_months = [col.split('_M')[-1] for col in ndvi_cols]
    print(f"  [系统感知] 探测到 {len(available_months)} 个有效月份节点。正在熔铸立体特征矩阵...")

    # 抽取三个维度的物理特征数据 (形状分别为 N x 7)
    X_ndvi = df[ndvi_cols].fillna(0).values
    X_gcvi = df[gcvi_cols].fillna(0).values
    X_glcm = df[glcm_cols].fillna(0).values

    # 【核心重构】：横向拼接成一个 21 维的复合立体特征矩阵 (N x 21)
    X_combined = np.hstack([X_ndvi, X_gcvi, X_glcm])
    print(f"  -> 原始矩阵物理组合成功，总形状 (Shape): {X_combined.shape}")

    print("\n2. 正在执行多维特征空间【标准化去量纲 (StandardScaler)】...")
    # 绝杀操作：消除纹理大数值对标准化 NDVI 的尺度压权，让所有指数在同一起跑线公平测距
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)
    print("  [√] 空间去量纲完成，高维几何距离计算准备就绪。")

    print("\n3. 正在启动多维高维 K-Means 聚类空间解算...")
    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    df['Cluster_ID'] = cluster_labels

    print("\n4. 正在执行多维高维几何测距，提取各簇前 30% 最纯净核心样本...")
    # 计算在【标准化尺度】下，每个样本到 4 个簇中心的真实欧氏距离
    all_distances = kmeans.transform(X_scaled)
    assigned_distances = all_distances[np.arange(len(X_scaled)), cluster_labels]
    df['Distance_To_Center'] = assigned_distances

    df['Is_Pure_Core'] = 0
    for c in range(n_clusters):
        cluster_mask = (df['Cluster_ID'] == c)
        cluster_dist = df.loc[cluster_mask, 'Distance_To_Center']

        if len(cluster_dist) > 0:
            threshold_30 = np.percentile(cluster_dist, 30)
            df.loc[cluster_mask & (df['Distance_To_Center'] <= threshold_30), 'Is_Pure_Core'] = 1

            raw_count = np.sum(cluster_mask)
            pure_count = np.sum(cluster_mask & (df['Distance_To_Center'] <= threshold_30))
            print(f"      -> 簇 Cluster {c}: 原始总量 {raw_count} -> 多维提纯后极品样本数: {pure_count}")

    # 落盘保存全量带距离与核心标记的Csv表
    out_labeled_csv = os.path.join(SELECT_DIR, 'All_Labeled_Parcels.csv')
    df.to_csv(out_labeled_csv, index=False, encoding='utf-8-sig')

    # 筛选纯正的小麦与玉米落盘（深度学习核心燃料库）
    pure_train_df = df[df['Is_Pure_Core'] == 1].copy()
    out_pure_csv = os.path.join(SELECT_DIR, 'Pure_Train_Dataset.csv')
    pure_train_df.to_csv(out_pure_csv, index=False, encoding='utf-8-sig')
    print(f"  [√] 多维核心提纯数据集已独立落盘 -> {out_pure_csv}")

    # ==========================================
    # 5. 【联动多流图】生成与对比可视化
    # ==========================================
    print("\n5. 正在生成高维【联动多流图】以便进行多特征物理学权衡对比...")

    # 创立 1行3列 的超级联动画布，清晰呈现各簇在不同物理指标上的本源分布
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # 重新计算每个簇在原始未缩放物理域中的真实均值剖面
    for i in range(n_clusters):
        c_mask = (cluster_labels == i)
        raw_count = np.sum(c_mask)
        pure_count = np.sum(c_mask & (df['Is_Pure_Core'] == 1))

        label_text = f'Cluster {i} [Raw={raw_count} | Core={pure_count}]'

        # 子图1：NDVI 曲线
        mean_ndvi = np.mean(X_ndvi[c_mask], axis=0)
        axes[0].plot(available_months, mean_ndvi, marker='o', linewidth=2.5, color=colors[i], label=label_text)

        # 子图2：GCVI 曲线
        mean_gcvi = np.mean(X_gcvi[c_mask], axis=0)
        axes[1].plot(available_months, mean_gcvi, marker='s', linewidth=2.5, color=colors[i])

        # 子图3：GLCM_Contrast 纹理曲线
        mean_glcm = np.mean(X_glcm[c_mask], axis=0)
        axes[2].plot(available_months, mean_glcm, marker='^', linewidth=2.5, color=colors[i])

    # 润色子图1 (NDVI)
    axes[0].set_title('Temporal NDVI Profile', fontsize=12)
    axes[0].set_xlabel('Month', fontsize=11)
    axes[0].set_ylabel('Mean NDVI', fontsize=11)
    axes[0].set_ylim(-0.05, 1.0)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(fontsize=8, loc='upper right')

    # 润色子图2 (GCVI)
    axes[1].set_title('Temporal GCVI Profile (Chlorophyll)', fontsize=12)
    axes[1].set_xlabel('Month', fontsize=11)
    axes[1].set_ylabel('Mean GCVI', fontsize=11)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    # 润色子图3 (GLCM_Contrast)
    axes[2].set_title('Temporal GLCM Contrast Profile (Texture)', fontsize=12)
    axes[2].set_xlabel('Month', fontsize=11)
    axes[2].set_ylabel('Mean Contrast', fontsize=11)
    axes[2].grid(True, linestyle='--', alpha=0.5)

    plt.suptitle('Multidimensional Phenology & Texture Clustering Comparison', fontsize=14, y=0.98)

    # 高清落盘
    plot_path = os.path.join(SELECT_DIR, 'Multidimensional_Clusters_Comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  [√] 高清多特征权衡对比联动图已保存 -> {plot_path}")

    plt.show()

    print("\n" + "=" * 70)
    print("架构师解读说明（如何与上一次对比）：")
    print(
        "  1. 观察 NDVI 子图：看原本的小麦簇和玉米簇在加入了纹理和 GCVI 后，其波峰和夏收断崖是否变得更利落、混淆点是否减少。")
    print(
        "  2. 观察 GCVI 与 Texture 子图：真正的玉米簇在 7-8月 应该会呈现‘GCVI极高’且‘纹理Contrast极高（行距阴影大）’的物理特征。")
    print("  3. 此时筛选出来的 'Pure_Train_Dataset.csv' 是经过时序、叶绿素、空间结构三重交叉过滤的绝佳样本！")
    print("=" * 70)


if __name__ == "__main__":
    perform_multidimensional_clustering()