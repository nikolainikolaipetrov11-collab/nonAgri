"""
    把所有指数以及统计特征共47维特征汇总到一个表中，用于之后哦使用
    build_tensor.py
"""
import os
import glob
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, 'date', 'out')

FULL_MONTHS = [4, 5, 6, 7, 8, 9, 10]

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


def build_universal_tensor():
    print("=" * 70)
    print("启动流 B 总装车间：47维多模态时序张量构建引擎 ")
    print("=" * 70)

    available_months = []
    opt_files = glob.glob(os.path.join(OUT_DIR, 'Optical_Features_out_*.csv'))
    for f in opt_files:
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
            df_month = pd.merge(df_opt, df_tex, on='parcel_id', how='left').fillna(0.0)

        prev_month = current_month - 1
        if prev_month in available_months and global_df is not None:
            prev_col_name = f"NDVI_mean_M{prev_month}"
            if prev_col_name in global_df.columns:
                safe_global = global_df.drop_duplicates(subset=['parcel_id'])
                prev_map = safe_global.set_index('parcel_id')[prev_col_name]
                df_month['delta_NDVI'] = df_month['NDVI_mean'] - df_month['parcel_id'].map(prev_map).fillna(0.0)
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

    global_df.fillna(0.0, inplace=True)
    parcel_count = len(global_df)

    print(f"  [*] 正在执行立体空间拓扑重塑 -> 目标 3D 张量形状: ({parcel_count}, 7, 47)...")
    tensor_3d = np.zeros((parcel_count, len(FULL_MONTHS), len(FEATURE_ORDER)), dtype=np.float32)

    for m in available_months:
        try:
            t_idx = FULL_MONTHS.index(m)
        except ValueError:
            continue

        current_month_features = [f"{feat}_M{m}" for feat in FEATURE_ORDER]
        # 防御性列写入
        for f_idx, feat_col in enumerate(current_month_features):
            if feat_col in global_df.columns:
                tensor_3d[:, t_idx, f_idx] = global_df[feat_col].values

    wide_csv_path = os.path.join(OUT_DIR, 'Dynamic_TimeSeries_WideTable.csv')
    global_df.to_csv(wide_csv_path, index=False, encoding='utf-8-sig')
    tensor_npy_path = os.path.join(OUT_DIR, 'Dynamic_TimeSeries_Tensor.npy')
    np.save(tensor_npy_path, tensor_3d)

    print("\n" + "=" * 70)
    print("  [大功告成] 数据管线全线通车！47 维张量底座已稳固落盘！")
    print(f"  -> 最终 3D 张量形状: {tensor_3d.shape}")
    print("=" * 70)


if __name__ == "__main__":
    build_universal_tensor()