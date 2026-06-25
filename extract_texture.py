"""
    用于计算纹理指数
    extract_texture.py
"""
import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from concurrent.futures import ProcessPoolExecutor
import warnings

from project_config import CONFIG

warnings.filterwarnings("ignore")

PROCESS_DIR = str(CONFIG.date_process_dir)
OUT_DIR = str(CONFIG.date_out_dir)
SHP_PATH = str(CONFIG.parcel_shp)

MAX_WORKERS = CONFIG.get("processing", "max_workers")


def calculate_texture(img_2d):
    valid_pixels = img_2d[~np.isnan(img_2d)]
    if valid_pixels.size < 9:
        return {'GLCM_Contrast': 0.0, 'GLCM_Correlation': 0.0, 'LBP_Variance': 0.0}

    p2, p98 = np.percentile(valid_pixels, (2, 98))

    if p98 - p2 < 1e-5:
        img_8bit = np.zeros_like(img_2d, dtype=np.uint8)
    else:
        img_clipped = np.clip(img_2d, p2, p98)
        img_8bit = np.uint8((img_clipped - p2) / (p98 - p2) * 255)

    img_8bit[np.isnan(img_2d)] = 0

    try:
        g = graycomatrix(img_8bit, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                         levels=256, symmetric=True, normed=True)
        # 用 np.nan_to_num 兜底极少数情况下的纹理异常值
        contrast = float(np.nan_to_num(graycoprops(g, 'contrast').mean()))
        correlation = float(np.nan_to_num(graycoprops(g, 'correlation').mean()))

        radius = 1
        n_points = 8 * radius
        lbp = local_binary_pattern(img_8bit, n_points, radius, method='uniform')
        lbp_var = float(np.nan_to_num(np.var(lbp[~np.isnan(img_2d)])))

        return {'GLCM_Contrast': contrast, 'GLCM_Correlation': correlation, 'LBP_Variance': lbp_var}
    except Exception:
        return {'GLCM_Contrast': 0.0, 'GLCM_Correlation': 0.0, 'LBP_Variance': 0.0}


def _extract_texture_chunk(args):
    gdf_chunk, tif_path = args
    results = []
    with rasterio.Env(NUM_THREADS='all_cpus', GDAL_CACHEMAX=512, GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
                      CPL_DEBUG='OFF'):
        with rasterio.open(tif_path) as src:
            for _, row in gdf_chunk.iterrows():
                geom, pid = row.geometry, row.parcel_id
                try:
                    out_image, _ = mask(src, [geom], crop=True, filled=False)
                    # 严防波段越界
                    if out_image.shape[0] < 8:
                        results.append((pid, 0.0, 0.0, 0.0))
                        continue

                    img_2d = out_image[7]  # 提取第 8 波段 (NIR_Raw)
                    if np.all(img_2d.mask):
                        results.append((pid, 0.0, 0.0, 0.0))
                    else:
                        img_data = img_2d.filled(np.nan)
                        tex = calculate_texture(img_data)
                        results.append((pid, tex['GLCM_Contrast'], tex['GLCM_Correlation'], tex['LBP_Variance']))
                except Exception:
                    results.append((pid, 0.0, 0.0, 0.0))
    return pd.DataFrame(results, columns=['parcel_id', 'GLCM_Contrast', 'GLCM_Correlation', 'LBP_Variance'])


def extract_monthly_textures():
    print("=" * 65)
    print(f"启动高阶空间纹理提取引擎 (全域防撞版 | 进程数: {MAX_WORKERS})")
    print("=" * 65)

    gdf = gpd.read_file(SHP_PATH)
    mosaic_tifs = sorted(glob.glob(os.path.join(PROCESS_DIR, 'Optical_Mosaic_out_*.tif')))

    if not mosaic_tifs: return

    for tif_path in mosaic_tifs:
        month_str = os.path.basename(tif_path).replace('Optical_Mosaic_', '').replace('.tif', '')
        out_csv = os.path.join(OUT_DIR, f"Texture_Features_{month_str}.csv")

        # 断点续传拦截
        if os.path.exists(out_csv):
            print(f"\n>>> [跳过] 月份 {month_str} 纹理 CSV 已存在，秒过！")
            continue

        print(f"\n>>> 正在挖掘纹理特征: 月份 {month_str}...")
        with rasterio.open(tif_path) as src:
            if gdf.crs != src.crs:
                gdf_aligned = gdf.to_crs(src.crs)
            else:
                gdf_aligned = gdf

        num_chunks = MAX_WORKERS * 4
        chunks = np.array_split(gdf_aligned, num_chunks)
        tasks = [(chunk, tif_path) for chunk in chunks]

        df_list = []
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for i, df_res in enumerate(executor.map(_extract_texture_chunk, tasks), 1):
                df_list.append(df_res)

        df_texture = pd.concat(df_list, ignore_index=True)
        df_texture.to_csv(out_csv, index=False, encoding='utf-8-sig')
        print(f"  [√] 物理纹理特征已成功落盘 -> {out_csv}")


if __name__ == "__main__":
    extract_monthly_textures()
