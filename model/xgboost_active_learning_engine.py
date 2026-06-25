"""
    阶段二（进化版）：集成了动态定额抽样与单体归因的有监督 XGBoost 引擎
    model/xgboost_active_learning_engine.py
    核心特性：
    1. 软伪标签 (GMM) + 黄金样本 (Human) 混合加权微调
    2. 单体 SHAP 翻译机，生成 "人话" 解释并完美兼容新版 3D 张量
    3. [动态重构] 不确定性主动抽样：支持全局配置总靶点数，系统自动按类平摊配额
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import pandas as pd
import numpy as np
import geopandas as gpd
import xgboost as xgb
import shap
import logging
from pathlib import Path
from dataset import PhenologyMAEDataset


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================================
# 统一配置
# ==========================================
NDVI_IDX = 0
GLCM_IDX = 43
LABEL_MAPPING = {0: "经济作物/其他", 1: "果树与林地", 2: "撂荒与裸地"}


# 配置抽样最大数量
ACTIVE_LEARNING_BATCH_SIZE = 20

def build_interpretable_features(real_physics_matrix):
    """构建 18 维具有高度农学可解释性的特征矩阵"""
    ndvi_matrix = real_physics_matrix[:, :, NDVI_IDX]
    glcm_matrix = real_physics_matrix[:, :, GLCM_IDX]
    df_feats = pd.DataFrame()

    df_feats['Peak_NDVI_盛夏峰值'] = np.max(ndvi_matrix[:, 1:6], axis=1)
    df_feats['Spring_NDVI_开春绿度'] = ndvi_matrix[:, 0]
    df_feats['NDVI_Std_生命周期波动'] = np.std(ndvi_matrix, axis=1)
    df_feats['Mean_GLCM_平均空间纹理'] = np.mean(glcm_matrix, axis=1)

    for i, month in enumerate([4, 5, 6, 7, 8, 9, 10]):
        df_feats[f'NDVI_{month}月'] = ndvi_matrix[:, i]
        df_feats[f'GLCM_{month}月'] = glcm_matrix[:, i]

    return df_feats

def generate_shap_human_reason(shap_values, feature_names, predicted_class, instance_idx):
    """单体 SHAP 翻译机：将冰冷的 SHAP 矩阵转化为业务园能看懂的解释"""
    if isinstance(shap_values, list):
        instance_shap = shap_values[predicted_class][instance_idx]
    elif len(shap_values.shape) == 3:
        instance_shap = shap_values[instance_idx, :, predicted_class]
    else:
        instance_shap = shap_values[instance_idx]

    top_indices = np.argsort(instance_shap)[-2:]

    top1_feat = feature_names[top_indices[1]].split('_')[-1]
    top1_val = instance_shap[top_indices[1]]
    top2_feat = feature_names[top_indices[0]].split('_')[-1]
    top2_val = instance_shap[top_indices[0]]

    if top1_val <= 0: return "依据不足，模型置信度偏低"
    reason = f"首要依据: [{top1_feat}]特征显著"
    if top2_val > 0: reason += f"；次要依据: [{top2_feat}]协同判定"
    return reason

def run_trustworthy_active_learning():
    logging.info("🚀 启动 XGBoost 定向抽样闭环与单体归因引擎...")

    DATA_ROOT = Path("E:/test")
    GMM_CSV = DATA_ROOT / "date/out/split_reports/03_Non_Grain_GMM_Final_Semantic.csv"
    ORIGINAL_SHP = DATA_ROOT / "shp/huocheng_dk_260605.shp"

    GOLDEN_SHP = DATA_ROOT / "shp/human_check/true_label.shp"

    FINAL_CSV = DATA_ROOT / "date/out/split_reports/04_Final_Trustworthy_NonGrain.csv"
    NEXT_BATCH_SHP = DATA_ROOT / "shp/human_check/Next_Batch_To_Check.shp"

    NEXT_BATCH_SHP.parent.mkdir(parents=True, exist_ok=True)

    # ==========================================
    # 1. 混合保真度数据装载
    # ==========================================
    logging.info("正在融合预训练伪标签与人类黄金样本...")
    gmm_df = pd.read_csv(GMM_CSV, dtype={'parcel_id': str})

    pseudo_df = gmm_df[gmm_df['GMM_Confidence'] > 0.95].copy()

    train_ids, train_labels, train_weights = [], [], []

    for _, row in pseudo_df.iterrows():
        train_ids.append(str(row['parcel_id']))
        lbl = 1 if "果树" in str(row['GMM_Semantic_Label']) else (2 if "撂荒" in str(row['GMM_Semantic_Label']) else 0)
        train_labels.append(lbl)
        train_weights.append(0.1)

    golden_ids_set = set()
    if GOLDEN_SHP.exists():
        logging.info("发现人类黄金样本库！正在执行强力微调介入...")
        golden_gdf = gpd.read_file(GOLDEN_SHP)

        cols_lower = {c.lower(): c for c in golden_gdf.columns}
        shp_id_col = cols_lower.get('parcel_id', cols_lower.get('id', golden_gdf.columns[0]))
        golden_gdf[shp_id_col] = golden_gdf[shp_id_col].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()

        label_col = 'true_label' if 'true_label' in golden_gdf.columns else 'ture_label'

        if label_col in golden_gdf.columns:
            golden_clean = golden_gdf[golden_gdf[label_col].notna()].copy()
            for _, row in golden_clean.iterrows():
                pid = row[shp_id_col]
                golden_ids_set.add(pid)
                if pid in train_ids:
                    idx = train_ids.index(pid)
                    train_labels[idx] = int(row[label_col])
                    train_weights[idx] = 10.0
                else:
                    train_ids.append(pid)
                    train_labels.append(int(row[label_col]))
                    train_weights.append(10.0)
            logging.info(f"-> 成功融合 {len(golden_clean)} 个黄金样本！")
    else:
        logging.warning("未检测到 true_label.shp，当前为 [冷启动模式]，完全依赖 GMM 伪标签进行预训练。")

    # ==========================================
    # 2. 提取物理特征并训练微调
    # ==========================================
    full_dataset = PhenologyMAEDataset(data_root=str(DATA_ROOT / "date/out"), mode='inference', mask_ratio=0.0)
    meta_df = full_dataset.meta_data
    wide_id_col = full_dataset.wide_id_col

    train_indices = meta_df.index[meta_df[wide_id_col].astype(str).str.replace(r'\.0$', '', regex=True).isin(train_ids)].tolist()
    train_extracted_ids = meta_df.loc[train_indices, wide_id_col].astype(str).str.replace(r'\.0$', '', regex=True).tolist()

    label_dict = dict(zip(train_ids, zip(train_labels, train_weights)))
    y_train = np.array([label_dict[pid][0] for pid in train_extracted_ids])
    w_train = np.array([label_dict[pid][1] for pid in train_extracted_ids])

    X_train = build_interpretable_features(full_dataset.get_raw_physical_features(train_indices))

    logging.info(" 正在执行 XGBoost 加权微调 (Fine-tuning)...")
    xgb_model = xgb.XGBClassifier(
        max_depth=3, learning_rate=0.05, n_estimators=100,
        objective='multi:softprob', num_class=3, random_state=42
    )
    xgb_model.fit(X_train, y_train, sample_weight=w_train)

    # ==========================================
    # 3. 全域靶点推演与单体 SHAP 解释
    # ==========================================
    logging.info("⚖️ 正在对全县异常靶点进行推演与 SHAP 局部归因...")
    anomaly_df = pd.read_csv(DATA_ROOT / "date/out/split_reports/02_Non_Grain_Anomalies.csv", dtype={'parcel_id': str})
    anomaly_df['parcel_id'] = anomaly_df['parcel_id'].str.replace(r'\.0$', '', regex=True).str.strip()

    infer_indices = meta_df.index[meta_df[wide_id_col].astype(str).str.replace(r'\.0$', '', regex=True).isin(set(anomaly_df['parcel_id']))].tolist()
    infer_extracted_ids = meta_df.loc[infer_indices, wide_id_col].astype(str).str.replace(r'\.0$', '', regex=True).tolist()

    X_infer = build_interpretable_features(full_dataset.get_raw_physical_features(infer_indices))
    feature_names = X_infer.columns.tolist()

    probs = xgb_model.predict_proba(X_infer)
    pred_classes = np.argmax(probs, axis=1)
    max_probs = np.max(probs, axis=1)

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_infer)

    reasons, margins = [], []
    for i in range(len(X_infer)):
        sorted_p = np.sort(probs[i])
        margins.append(sorted_p[-1] - sorted_p[-2])

        if max_probs[i] < 0.60:
            pred_classes[i] = 0
            max_probs[i] = probs[i, 0]
            reasons.append("模型置信度偏低，执行安全兜底")
        else:
            reasons.append(generate_shap_human_reason(shap_values, feature_names, pred_classes[i], i))

    infer_result_df = pd.DataFrame({
        'parcel_id': infer_extracted_ids,
        'XGB_Class': pred_classes,
        'XGB_Semantic': [LABEL_MAPPING[c] for c in pred_classes],
        'XGB_Confidence': np.round(max_probs, 4),
        'Prob_Orchard': probs[:, 1],
        'Prob_Abandoned': probs[:, 2],
        'Reason': reasons,
        'Margin': margins
    })

    final_df = pd.merge(anomaly_df, infer_result_df, on='parcel_id', how='left')
    final_df['Joint_Conf'] = np.round(final_df['Anomaly_Confidence'] * final_df['XGB_Confidence'], 4)
    final_df = final_df.sort_values(by='Joint_Conf', ascending=False)
    final_df.to_csv(FINAL_CSV, index=False, encoding='utf-8-sig')
    logging.info(f"🎉 终极台账已生成: {FINAL_CSV.name}")

    # ==========================================
    # 4. 不确定性抽样：根据全局配置平摊配额
    # ==========================================
    logging.info(
        f"🗺️ 正在执行动态配额不确定性抽样（目标总数 {ACTIVE_LEARNING_BATCH_SIZE} 块，严格过滤经济作物）并挂载至 GIS...")

    # 动态计算配额
    quota_orchard = ACTIVE_LEARNING_BATCH_SIZE // 2
    quota_abandoned = ACTIVE_LEARNING_BATCH_SIZE - quota_orchard

    # 提取未验证的异常靶点池
    unlabeled_mask = ~infer_result_df['parcel_id'].isin(golden_ids_set)
    confused_pool = infer_result_df[unlabeled_mask].copy()

    # 业务前置强过滤！
    # 果树抽样池：必须是模型最终预测为果树(1)的地块
    orchard_pool = confused_pool[confused_pool['XGB_Class'] == 1].copy()
    # 撂荒抽样池：必须是模型最终预测为撂荒(2)的地块
    abandoned_pool = confused_pool[confused_pool['XGB_Class'] == 2].copy()

    # 在已经定性的池子里，按 Margin 升序排列（Margin 越小，说明第一名和第二名差距越小）
    orchard_batch = orchard_pool.sort_values(by='Margin', ascending=True).head(quota_orchard)
    abandoned_batch = abandoned_pool.sort_values(by='Margin', ascending=True).head(quota_abandoned)

    # 合并去重
    next_batch = pd.concat([orchard_batch, abandoned_batch], ignore_index=True).drop_duplicates(subset=['parcel_id'])
    target_ids = next_batch['parcel_id'].tolist()

    # GIS 矩阵空间几何合并
    gdf = gpd.read_file(ORIGINAL_SHP)
    cols_lower = {c.lower(): c for c in gdf.columns}
    orig_shp_id = cols_lower.get('parcel_id', cols_lower.get('id', gdf.columns[0]))
    gdf[orig_shp_id] = gdf[orig_shp_id].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()

    export_gdf = gdf[gdf[orig_shp_id].isin(target_ids)].copy()
    export_gdf = export_gdf.merge(next_batch, left_on=orig_shp_id, right_on='parcel_id', how='inner')

    # 移除冗余主键
    if 'parcel_id' in export_gdf.columns and orig_shp_id != 'parcel_id':
        export_gdf = export_gdf.drop(columns=['parcel_id'], errors='ignore')

    # 预留审核标签真值
    export_gdf['true_label'] = -9999  # 农业常用值
    export_gdf = export_gdf.fillna(-9999)

    # 物理安全映射：全量控制在 10 字符内，消灭所有垃圾警告
    mapping = {
        'XGB_Class': 'xgb_cls', 'XGB_Semantic': 'xgb_label', 'XGB_Confidence': 'xgb_conf',
        'Prob_Orchard': 'p_orchard', 'Prob_Abandoned': 'p_abandon',
        'Reason': 'xgb_reason', 'Margin': 'xgb_margin'
    }
    export_gdf = export_gdf.rename(columns=mapping)

    # 物理强制落盘
    export_gdf.to_file(NEXT_BATCH_SHP, driver="ESRI Shapefile", encoding="utf-8")

    print("\n" + "=" * 65)
    print("🎯 【架构师重构指令】不确定性有监督抽样飞轮已就绪：")
    print("=" * 65)
    print(f"1. 严格过滤：剔除了所有经济作物。当前仅在预测为果树和撂荒的地块中抽取最不确定的样本。")
    print(f"2. 核查专用图层已生成覆盖: {NEXT_BATCH_SHP.name} (最终输出 {len(export_gdf)} 块)")
    print("3. 请在 ArcGIS 里查看，此时的 'xgb_label' 有【果树与林地】和【撂荒与裸地】的边缘样本")
    print("4. 核查完毕后，将该图层数据合并入 -> true_label.shp 驱动下次模型微调")
    print("=" * 65)


if __name__ == "__main__":
    run_trustworthy_active_learning()
