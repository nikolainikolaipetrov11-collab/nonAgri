"""
    把所有指数以及统计特征共47维特征汇总到一个表中，用于之后哦使用
    build_tensor.py
"""
import os
import glob
import pandas as pd
import numpy as np
import warnings

from project_config import CONFIG

warnings.filterwarnings("ignore")

OUT_DIR = str(CONFIG.date_out_dir)

FULL_MONTHS = CONFIG.get("processing", "full_months")

FEATURE_ORDER = [
    # 1. 四大植被指数 (28维)
    'NDVI_mean', 'NDVI_median', 'NDVI_cv', 'NDVI_skew', 'NDVI_kurt', 'NDVI_q25', 'NDVI_q75',
    'EVI_mean', 'EVI_median', 'EVI_cv', 'EVI_skew', 'EVI_kurt', 'EVI_q25', 'EVI_q75',
    'GCVI_mean', 'GCVI_median', 'GCVI_cv', 'GCVI_skew', 'GCVI_kurt', 'GCVI_q25', 'GCVI_q75',
    'SAVI_mean', 'SAVI_median', 'SAVI_cv', 'SAVI_skew', 'SAVI_kurt', 'SAVI_q25', 'SAVI_q75',

    # 2. 建筑/不透水指数系 ISI + UBI (10维)
    'ISI_mean', 'ISI_median', 'ISI_cv', 'ISI_q25', 'ISI_q75',
    'UBI_mean', 'UBI_median', 'UBI_cv', 'UBI_q25', 'UBI_q75',

    # 3. 水体/坑塘养殖指数 NDWI (5维)
    'NDWI_mean', 'NDWI_median', 'NDWI_cv', 'NDWI_q25', 'NDWI_q75',

    # 4. NIR 物理纹理 (3维)
    'GLCM_Contrast', 'GLCM_Correlation', 'LBP_Variance',

    # 5. 时序突变 (1维)
    'delta_NDVI'
]


def _impute_feature_columns_for_tensor(df, feature_columns):
    """
    张量不能保存 NaN，否则后续神经网络、GMM 和规则判断都会不稳定。
    这里仅在写入张量前补值：优先用同一列的中位数，整列都缺失时再退回 0。
    CSV 宽表仍保留原始 NaN，方便区分“真实 0”和“没有有效像元”。
    """
    df_for_tensor = df.copy()
    for col in feature_columns:
        if col not in df_for_tensor.columns:
            continue

        values = pd.to_numeric(df_for_tensor[col], errors='coerce')
        values = values.replace([np.inf, -np.inf], np.nan)
        median_value = values.median()
        if pd.isna(median_value):
            median_value = 0.0
        df_for_tensor[col] = values.fillna(median_value)

    return df_for_tensor


def _clip_quality(values, min_quality):
    return np.clip(np.nan_to_num(values, nan=1.0, posinf=1.0, neginf=min_quality), min_quality, 1.0)


def _get_manual_month_weight(month):
    month_weights = CONFIG.get("quality", "month_weights", {}) or {}
    return float(month_weights.get(str(month), month_weights.get(month, 1.0)))


def _build_quality_matrix(global_df, full_months):
    """
    为每个“地块-月份”计算影像质量分。
    分数越接近 1，说明该月数据越可靠；越接近 0，后续训练和推理越应降低它的影响。
    """
    min_quality = float(CONFIG.get("quality", "min_quality", 0.05))
    quality_matrix = np.ones((len(global_df), len(full_months)), dtype=np.float32)

    for month_index, month in enumerate(full_months):
        manual_score = _get_manual_month_weight(month)

        valid_ratio_cols = [c for c in global_df.columns if c.endswith(f"_valid_ratio_M{month}")]
        if valid_ratio_cols:
            valid_score = global_df[valid_ratio_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1).to_numpy()
        else:
            valid_score = np.ones(len(global_df), dtype=np.float32)

        core_cols = [f"{name}_M{month}" for name in ["NDVI_mean", "EVI_mean", "GCVI_mean", "SAVI_mean", "NDWI_mean"]]
        existing_core_cols = [c for c in core_cols if c in global_df.columns]
        if existing_core_cols:
            completeness_score = 1.0 - global_df[existing_core_cols].isna().mean(axis=1).to_numpy()
        else:
            completeness_score = np.ones(len(global_df), dtype=np.float32)

        cv_cols = [f"{name}_M{month}" for name in ["NDVI_cv", "EVI_cv", "GCVI_cv", "SAVI_cv", "NDWI_cv"]]
        existing_cv_cols = [c for c in cv_cols if c in global_df.columns]
        if existing_cv_cols:
            cv_values = global_df[existing_cv_cols].apply(pd.to_numeric, errors='coerce')
            median_cv = cv_values.median(axis=1).to_numpy()
            # CV 越大，地块内部越乱，越可能受云影、拼接或混合像元影响。
            dispersion_score = 1.0 / (1.0 + np.clip(median_cv, 0.0, 3.0))
        else:
            dispersion_score = np.ones(len(global_df), dtype=np.float32)

        auto_score = (
            0.50 * _clip_quality(valid_score, min_quality)
            + 0.25 * _clip_quality(completeness_score, min_quality)
            + 0.25 * _clip_quality(dispersion_score, min_quality)
        )
        quality_matrix[:, month_index] = _clip_quality(manual_score * auto_score, min_quality)

    ndvi_cols = [f"NDVI_mean_M{m}" for m in full_months]
    if all(c in global_df.columns for c in ndvi_cols):
        ndvi_table = global_df[ndvi_cols].apply(pd.to_numeric, errors='coerce')
        for month_index in range(1, len(full_months) - 1):
            prev_values = ndvi_table.iloc[:, month_index - 1]
            curr_values = ndvi_table.iloc[:, month_index]
            next_values = ndvi_table.iloc[:, month_index + 1]
            neighbor_mean = pd.concat([prev_values, next_values], axis=1).mean(axis=1)
            jump = (curr_values - neighbor_mean).abs().to_numpy()
            temporal_score = np.where(np.isnan(jump), 1.0, 1.0 / (1.0 + np.clip(jump / 0.25, 0.0, 4.0)))
            quality_matrix[:, month_index] *= _clip_quality(temporal_score, min_quality)

    return _clip_quality(quality_matrix, min_quality).astype(np.float32)


def build_universal_tensor():
    print("=" * 70)
    print("启动流 B 总装车间：47维多模态时序张量构建引擎 ")
    print("=" * 70)

    available_months = []
    opt_files = glob.glob(os.path.join(OUT_DIR, 'Optical_Features_out_*.csv'))
    for f in opt_files:
        if os.path.getsize(f) == 0:
            print(f"  [-] 跳过空特征文件: {os.path.basename(f)}，请先重新运行 fenqutongji.py 生成有效 CSV。")
            continue
        month_str = os.path.basename(f).replace('Optical_Features_out_', '').replace('.csv', '')
        if month_str.isdigit(): available_months.append(int(month_str))

    available_months.sort()
    if not available_months:
        print("  [致命错误] 未在 date/out 目录下检测到特征文件！")
        return

    global_df = None

    for current_month in available_months:
        opt_csv = os.path.join(OUT_DIR, f"Optical_Features_out_{current_month}.csv")
        tex_csv = os.path.join(OUT_DIR, f"Texture_Features_out_{current_month}.csv")

        if not os.path.exists(tex_csv):
            print(f"  [-] 警告: 缺失 {current_month} 月的物理纹理特征，本月以 0.0 填补。")
            df_opt = pd.read_csv(opt_csv)
            df_opt['parcel_id'] = df_opt['parcel_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            df_month = df_opt.drop_duplicates(subset=['parcel_id'], keep='first').copy()
            for col in ['GLCM_Contrast', 'GLCM_Correlation', 'LBP_Variance']:
                df_month[col] = 0.0
        else:
            df_opt = pd.read_csv(opt_csv)
            df_tex = pd.read_csv(tex_csv)
            df_opt['parcel_id'] = df_opt['parcel_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            if 'df_tex' in locals():
                df_tex['parcel_id'] = df_tex['parcel_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()

            df_opt = df_opt.drop_duplicates(subset=['parcel_id'], keep='first')
            df_tex = df_tex.drop_duplicates(subset=['parcel_id'], keep='first')
            df_month = pd.merge(df_opt, df_tex, on='parcel_id', how='left')

        prev_month = current_month - 1
        if prev_month in available_months and global_df is not None:
            prev_col_name = f"NDVI_mean_M{prev_month}"
            if prev_col_name in global_df.columns:
                safe_global = global_df.drop_duplicates(subset=['parcel_id'])
                prev_map = safe_global.set_index('parcel_id')[prev_col_name]
                df_month['delta_NDVI'] = df_month['NDVI_mean'] - df_month['parcel_id'].map(prev_map)
            else:
                df_month['delta_NDVI'] = 0.0
        else:
            df_month['delta_NDVI'] = 0.0

        rename_cols = {col: f"{col}_M{current_month}" for col in df_month.columns if col != 'parcel_id'}
        df_month.rename(columns=rename_cols, inplace=True)

        if global_df is None:
            global_df = df_month
        else:
            global_df = pd.merge(global_df, df_month, on='parcel_id', how='outer')

    parcel_count = len(global_df)

    print(f"  [*] 正在执行立体空间拓扑重塑 -> 目标 3D 张量形状: ({parcel_count}, 7, 47)...")
    tensor_3d = np.zeros((parcel_count, len(FULL_MONTHS), len(FEATURE_ORDER)), dtype=np.float32)
    tensor_feature_cols = [f"{feat}_M{m}" for m in FULL_MONTHS for feat in FEATURE_ORDER]
    tensor_df = _impute_feature_columns_for_tensor(global_df, tensor_feature_cols)
    quality_matrix = _build_quality_matrix(global_df, FULL_MONTHS)

    for month_index, month in enumerate(FULL_MONTHS):
        global_df[f"Quality_M{month}"] = quality_matrix[:, month_index]

    for m in available_months:
        try:
            t_idx = FULL_MONTHS.index(m)
        except ValueError:
            continue

        current_month_features = [f"{feat}_M{m}" for feat in FEATURE_ORDER]
        # 防御性列写入
        for f_idx, feat_col in enumerate(current_month_features):
            if feat_col in tensor_df.columns:
                tensor_3d[:, t_idx, f_idx] = tensor_df[feat_col].values

    wide_csv_path = os.path.join(OUT_DIR, 'Dynamic_TimeSeries_WideTable.csv')
    global_df.to_csv(wide_csv_path, index=False, encoding='utf-8-sig')
    tensor_npy_path = os.path.join(OUT_DIR, 'Dynamic_TimeSeries_Tensor.npy')
    np.save(tensor_npy_path, tensor_3d)
    quality_npy_path = os.path.join(OUT_DIR, 'Dynamic_TimeSeries_Quality.npy')
    np.save(quality_npy_path, quality_matrix)

    print("\n" + "=" * 70)
    print("  [大功告成] 数据管线全线通车！47 维张量底座已稳固落盘！")
    print(f"  -> 最终 3D 张量形状: {tensor_3d.shape}")
    print(f"  -> 质量矩阵形状: {quality_matrix.shape}")
    print("=" * 70)


if __name__ == "__main__":
    build_universal_tensor()
