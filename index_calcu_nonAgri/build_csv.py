"""
    非农化突变检测引擎
    build_tensor.py
"""
import os
import glob
import pandas as pd
import numpy as np
import warnings
import logging
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(message)s')

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = str(BASE_DIR / 'date' / 'out_nonAgri')


FULL_MONTHS = [4, 5, 6, 7, 8, 9, 10]

# ==========================================
# 核心特征罗盘 (38维)
# ==========================================
FEATURE_ORDER = [
    # 1. 四大植被指数 (20维) - 关注局部骤降 (q05, q10)
    'NDVI_mean', 'NDVI_median', 'NDVI_cv', 'NDVI_q05', 'NDVI_q10',
    'EVI_mean', 'EVI_median', 'EVI_cv', 'EVI_q05', 'EVI_q10',
    'GCVI_mean', 'GCVI_median', 'GCVI_cv', 'GCVI_q05', 'GCVI_q10',
    'SAVI_mean', 'SAVI_median', 'SAVI_cv', 'SAVI_q05', 'SAVI_q10',

    # 2. 建筑/不透水指数系 ISI + UBI (10维) - 关注局部激增 (q90, q95)
    'ISI_mean', 'ISI_median', 'ISI_cv', 'ISI_q90', 'ISI_q95',
    'UBI_mean', 'UBI_median', 'UBI_cv', 'UBI_q90', 'UBI_q95',

    # 3. 水体/坑塘养殖指数 NDWI (5维) - 关注局部激增 (q90, q95)
    'NDWI_mean', 'NDWI_median', 'NDWI_cv', 'NDWI_q90', 'NDWI_q95',

    # 4. NIR 物理纹理 (3维) - 关注空间异质性突变
    'GLCM_Contrast', 'GLCM_Correlation', 'LBP_Variance'
]


def build_mutation_tensor():
    logging.info("=" * 70)
    logging.info("🚀 启动流 B 总装车间：时序极值张量与动态突变矩阵构建引擎")
    logging.info("=" * 70)

    available_months = []
    opt_files = glob.glob(os.path.join(OUT_DIR, 'Optical_Features_out_*.csv'))
    for f in opt_files:
        month_str = os.path.basename(f).replace('Optical_Features_out_', '').replace('.csv', '')
        if month_str.isdigit(): available_months.append(int(month_str))

    available_months.sort()
    if not available_months:
        logging.error("  ⚠️ [致命错误] 未在 date/out 目录下检测到特征文件！")
        return

    global_df = None

    # ---------------------------------------------------------
    # 阶段 1：基础极值切片无缝拼接
    # ---------------------------------------------------------
    logging.info("💡 [1/3] 正在缝合各月切片，组装抗稀释时序底座...")
    for current_month in available_months:
        opt_csv = os.path.join(OUT_DIR, f"Optical_Features_out_{current_month}.csv")
        tex_csv = os.path.join(OUT_DIR, f"Texture_Features_out_{current_month}.csv")

        df_opt = pd.read_csv(opt_csv)
        # 防弹级地块 ID 处理
        df_opt['parcel_id'] = df_opt['parcel_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        df_opt = df_opt.drop_duplicates(subset=['parcel_id'], keep='first')

        if not os.path.exists(tex_csv):
            logging.warning(f"  [-] 警告: 缺失 {current_month} 月物理纹理，本月以 0.0 填补。")
            df_month = df_opt.copy()
            for col in ['GLCM_Contrast', 'GLCM_Correlation', 'LBP_Variance']:
                df_month[col] = 0.0
        else:
            df_tex = pd.read_csv(tex_csv)
            df_tex['parcel_id'] = df_tex['parcel_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            df_tex = df_tex.drop_duplicates(subset=['parcel_id'], keep='first')
            df_month = pd.merge(df_opt, df_tex, on='parcel_id', how='left').fillna(0.0)

        # 列名附带月份标记
        rename_cols = {col: f"{col}_M{current_month}" for col in df_month.columns if col != 'parcel_id'}
        df_month.rename(columns=rename_cols, inplace=True)

        if global_df is None:
            global_df = df_month
        else:
            global_df = pd.merge(global_df, df_month, on='parcel_id', how='outer')

    global_df.fillna(0.0, inplace=True)
    parcel_count = len(global_df)

    # ---------------------------------------------------------
    # 阶段 2：构建 3D 纯净时序张量 (用于备用深度学习/时序模型)
    # ---------------------------------------------------------
    logging.info(f"💡 [2/3] 正在执行立体空间拓扑重塑 -> 目标 3D 张量形状: ({parcel_count}, 7, {len(FEATURE_ORDER)})...")
    tensor_3d = np.zeros((parcel_count, len(FULL_MONTHS), len(FEATURE_ORDER)), dtype=np.float32)

    for m in available_months:
        try:
            t_idx = FULL_MONTHS.index(m)
        except ValueError:
            continue

        current_month_features = [f"{feat}_M{m}" for feat in FEATURE_ORDER]
        for f_idx, feat_col in enumerate(current_month_features):
            if feat_col in global_df.columns:
                tensor_3d[:, t_idx, f_idx] = global_df[feat_col].values

    # ---------------------------------------------------------
    # 阶段 3：突变特征工程 (一阶差分与极值锁定)
    # ---------------------------------------------------------
    logging.info("🚀 [3/3] 激活突变引擎：计算时序一阶差分与极端跃变点...")
    for i in range(1, len(available_months)):
        curr_m = available_months[i]
        prev_m = available_months[i-1]

        # 1. 抓取局部建筑/水体暴涨 (使用 q95 防空间稀释)
        if f'UBI_q95_M{curr_m}' in global_df.columns and f'UBI_q95_M{prev_m}' in global_df.columns:
            global_df[f'delta_UBI_q95_M{curr_m}_M{prev_m}'] = global_df[f'UBI_q95_M{curr_m}'] - global_df[f'UBI_q95_M{prev_m}']
        if f'NDWI_q95_M{curr_m}' in global_df.columns and f'NDWI_q95_M{prev_m}' in global_df.columns:
            global_df[f'delta_NDWI_q95_M{curr_m}_M{prev_m}'] = global_df[f'NDWI_q95_M{curr_m}'] - global_df[f'NDWI_q95_M{prev_m}']

        # 2. 抓取局部植被锐减 (使用 q05 防空间稀释)
        if f'NDVI_q05_M{curr_m}' in global_df.columns and f'NDVI_q05_M{prev_m}' in global_df.columns:
            global_df[f'delta_NDVI_q05_M{curr_m}_M{prev_m}'] = global_df[f'NDVI_q05_M{curr_m}'] - global_df[f'NDVI_q05_M{prev_m}']

        # 3. 抓取纹理剧烈变化 (动土物理印记)
        if f'GLCM_Contrast_M{curr_m}' in global_df.columns and f'GLCM_Contrast_M{prev_m}' in global_df.columns:
            global_df[f'delta_Contrast_M{curr_m}_M{prev_m}'] = global_df[f'GLCM_Contrast_M{curr_m}'] - global_df[f'GLCM_Contrast_M{prev_m}']

    # 提取全时段最大突变振幅 (寻找非农化确凿断点)
    ubi_cols = [c for c in global_df.columns if c.startswith('delta_UBI_q95')]
    ndwi_cols = [c for c in global_df.columns if c.startswith('delta_NDWI_q95')]
    ndvi_cols = [c for c in global_df.columns if c.startswith('delta_NDVI_q05')]

    if ubi_cols: global_df['MAX_JUMP_UBI_q95'] = global_df[ubi_cols].max(axis=1)
    if ndwi_cols: global_df['MAX_JUMP_NDWI_q95'] = global_df[ndwi_cols].max(axis=1)
    if ndvi_cols: global_df['MAX_DROP_NDVI_q05'] = global_df[ndvi_cols].min(axis=1) # 负值越小代表植被破坏越惨烈

    # 落盘保存
    wide_csv_path = os.path.join(OUT_DIR, 'Dynamic_TimeSeries_WideTable_Mutation.csv')
    global_df.to_csv(wide_csv_path, index=False, encoding='utf-8-sig')

    tensor_npy_path = os.path.join(OUT_DIR, 'Dynamic_TimeSeries_Tensor.npy')
    np.save(tensor_npy_path, tensor_3d)

    logging.info("\n" + "=" * 70)
    logging.info(f"  [大功告成] 数据管线全线通车！目标地块数: {parcel_count}")
    logging.info(f"  -> 3D 张量底座已落盘: {tensor_3d.shape}")
    logging.info(f"  -> 包含极值差分的宽表已落盘: {wide_csv_path}")
    logging.info("=" * 70)

if __name__ == "__main__":
    build_mutation_tensor()