
"""
    用于计算光学指数
    fenqutongji.py
"""
import os
import glob
import rasterio
from rasterio.merge import merge
from rasterio.windows import Window
from rasterio.mask import mask
import geopandas as gpd
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import gc

from project_config import CONFIG

warnings.filterwarnings("ignore")

# ==========================================
# 0. 基础路径与性能配置
# ==========================================
RAW_DIR = str(CONFIG.date_raw_dir)
PROCESS_DIR = str(CONFIG.date_process_dir)
OUT_DIR = str(CONFIG.date_out_dir)
SHP_PATH = str(CONFIG.parcel_shp)

os.makedirs(PROCESS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

CHUNK_SIZE = CONFIG.get("processing", "chunk_size")
MAX_WORKERS = CONFIG.get("processing", "max_workers")
MOSAIC_CHUNK_SIZE = CONFIG.get("processing", "mosaic_chunk_size")
REFLECTANCE_SCALE = 10000.0


# ==========================================
# 1. 核心解算与镶嵌逻辑
# ==========================================
def calculate_gf1_indices(input_dat, output_tif):
    try:
        with rasterio.open(input_dat) as src:
            meta = src.meta.copy()
            img_height, img_width = src.height, src.width
            meta.update({"driver": "GTiff", "count": 8, "dtype": 'float32', "nodata": np.nan,
                         "compress": "lzw", "tiled": True, "blockxsize": 256, "blockysize": 256, "BIGTIFF": "YES"})

            with rasterio.open(output_tif, 'w', **meta) as dst:
                for row_off in range(0, img_height, CHUNK_SIZE):
                    for col_off in range(0, img_width, CHUNK_SIZE):
                        curr_h = min(CHUNK_SIZE, img_height - row_off)
                        curr_w = min(CHUNK_SIZE, img_width - col_off)
                        window = Window(col_off, row_off, curr_w, curr_h)

                        # 原始影像是 uint16，nodata=0，像元值约为真实反射率 * 10000。
                        # masked=True 会先把 nodata 排除，filled(np.nan) 让后续统计能识别缺失值。
                        blue = src.read(1, window=window, masked=True).astype(np.float32).filled(np.nan)
                        green = src.read(2, window=window, masked=True).astype(np.float32).filled(np.nan)
                        red = src.read(3, window=window, masked=True).astype(np.float32).filled(np.nan)
                        nir = src.read(4, window=window, masked=True).astype(np.float32).filled(np.nan)

                        blue_norm = blue / REFLECTANCE_SCALE
                        green_norm = green / REFLECTANCE_SCALE
                        red_norm = red / REFLECTANCE_SCALE
                        nir_norm = nir / REFLECTANCE_SCALE

                        ndvi = np.where((nir_norm + red_norm) == 0, np.nan,
                                        (nir_norm - red_norm) / (nir_norm + red_norm))
                        ndvi = np.where((ndvi >= -0.2) & (ndvi <= 1.0), ndvi, np.nan)

                        evi_denom = (nir_norm + 6.0 * red_norm - 7.5 * blue_norm + 1.0)
                        evi = np.where(evi_denom == 0, np.nan, 2.5 * (nir_norm - red_norm) / evi_denom)
                        evi = np.where((evi >= -0.2) & (evi <= 1.5), evi, np.nan)

                        gcvi = np.where(green_norm == 0, np.nan, (nir_norm / green_norm) - 1.0)
                        gcvi = np.where((gcvi >= -1.0) & (gcvi <= 5.0), gcvi, np.nan)

                        L = 0.5
                        savi_denom = (nir_norm + red_norm + L)
                        savi = np.where(savi_denom == 0, np.nan,
                                        ((nir_norm - red_norm) / savi_denom) * (1.0 + L))
                        savi = np.where((savi >= -0.2) & (savi <= 1.0), savi, np.nan)

                        isi = 0.8192 * blue_norm - 0.5735 * nir_norm + 0.0750
                        isi = np.where((isi >= -1.0) & (isi <= 1.0), isi, np.nan)

                        ubi = 1.34 * blue_norm - 2.24 * green_norm
                        ubi = np.where((ubi >= -5.0) & (ubi <= 5.0), ubi, np.nan)

                        ndwi = np.where((green_norm + nir_norm) == 0, np.nan,
                                        (green_norm - nir_norm) / (green_norm + nir_norm))
                        ndwi = np.where((ndwi >= -1.0) & (ndwi <= 1.0), ndwi, np.nan)

                        dst.write(ndvi, 1, window=window)
                        dst.write(evi, 2, window=window)
                        dst.write(gcvi, 3, window=window)
                        dst.write(savi, 4, window=window)
                        dst.write(isi, 5, window=window)
                        dst.write(ubi, 6, window=window)
                        dst.write(ndwi, 7, window=window)
                        dst.write(nir, 8, window=window)

                        del blue, green, red, nir, blue_norm, green_norm, red_norm, nir_norm
                        del ndvi, evi, gcvi, savi, isi, ubi, ndwi
        return output_tif, True, ""
    except Exception as e:
        return output_tif, False, str(e)


def memory_safe_mosaic(src_paths, out_path, chunk_size=4096):
    srcs = [rasterio.open(fp) for fp in src_paths]
    lefts, bottoms = [src.bounds.left for src in srcs], [src.bounds.bottom for src in srcs]
    rights, tops = [src.bounds.right for src in srcs], [src.bounds.top for src in srcs]

    global_left, global_bottom = min(lefts), min(bottoms)
    global_right, global_top = max(rights), max(tops)
    res_x, res_y = srcs[0].res
    width = int(np.ceil((global_right - global_left) / res_x))
    height = int(np.ceil((global_top - global_bottom) / res_y))

    global_transform = rasterio.transform.from_bounds(global_left, global_bottom, global_right, global_top, width,
                                                      height)
    meta = srcs[0].meta.copy()
    meta.update({"height": height, "width": width, "transform": global_transform,
                 "compress": "lzw", "tiled": True, "blockxsize": 256, "blockysize": 256, "BIGTIFF": "YES",
                 "nodata": np.nan})

    bands_count = srcs[0].count
    print(f"      [*] 正在开辟 {width} x {height} 巨幅基底...")

    with rasterio.open(out_path, 'w', **meta) as dst:
        for row_off in range(0, height, chunk_size):
            for col_off in range(0, width, chunk_size):
                curr_h, curr_w = min(chunk_size, height - row_off), min(chunk_size, width - col_off)
                window = Window(col_off, row_off, curr_w, curr_h)
                win_bounds = rasterio.windows.bounds(window, global_transform)

                intersecting_srcs = [src for src in srcs if
                                     not (src.bounds.right <= win_bounds[0] or src.bounds.left >= win_bounds[2] or
                                          src.bounds.top <= win_bounds[1] or src.bounds.bottom >= win_bounds[3])]
                if not intersecting_srcs: continue

                chunk_mosaic, _ = merge(intersecting_srcs, bounds=win_bounds, res=(res_x, res_y), nodata=np.nan,
                                        method='max')
                write_h, write_w = min(curr_h, chunk_mosaic.shape[1]), min(curr_w, chunk_mosaic.shape[2])

                block = np.full((bands_count, curr_h, curr_w), np.nan, dtype=np.float32)
                block[:, :write_h, :write_w] = chunk_mosaic[:, :write_h, :write_w]
                dst.write(block, window=window)
                del chunk_mosaic, block
    for src in srcs: src.close()


# ==========================================
# 2. 多进程内存切片提取特征
# ==========================================
def _extract_chunk(args):
    gdf_chunk, tif_path, bands_names = args
    results = []

    with rasterio.Env(NUM_THREADS='all_cpus', GDAL_CACHEMAX=1024, GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
                      CPL_DEBUG='OFF'):
        with rasterio.open(tif_path) as src:
            for _, row in gdf_chunk.iterrows():
                geom, pid = row.geometry, row.parcel_id
                res = {'parcel_id': pid}

                try:
                    out_image, _ = mask(src, [geom], crop=True, filled=False)
                    for i, b_name in enumerate(bands_names):
                        data = out_image[i]
                        valid_data = data.compressed()
                        valid_data = valid_data[~np.isnan(valid_data)]
                        total_count = int(data.count()) if hasattr(data, "count") else int(data.size)
                        valid_count = int(valid_data.size)
                        valid_ratio = valid_count / total_count if total_count > 0 else 0.0
                        res[f"{b_name}_valid_count"] = valid_count
                        res[f"{b_name}_valid_ratio"] = float(valid_ratio)

                        if valid_data.size < 3:
                            for stat in ['mean', 'median', 'cv', 'skew', 'kurt', 'q25', 'q75']:
                                res[f"{b_name}_{stat}"] = np.nan
                        else:
                            mean_val = float(np.mean(valid_data))
                            res[f"{b_name}_mean"] = mean_val
                            res[f"{b_name}_median"] = float(np.median(valid_data))
                            res[f"{b_name}_cv"] = float(np.std(valid_data) / abs(mean_val)) if abs(mean_val) > 1e-6 else 0.0
                            res[f"{b_name}_skew"] = float(np.nan_to_num(skew(valid_data)))
                            res[f"{b_name}_kurt"] = float(np.nan_to_num(kurtosis(valid_data)))
                            res[f"{b_name}_q25"] = float(np.percentile(valid_data, 25))
                            res[f"{b_name}_q75"] = float(np.percentile(valid_data, 75))

                    del out_image, valid_data
                except Exception:
                    for b_name in bands_names:
                        res[f"{b_name}_valid_count"] = 0
                        res[f"{b_name}_valid_ratio"] = 0.0
                        for stat in ['mean', 'median', 'cv', 'skew', 'kurt', 'q25', 'q75']:
                            res[f"{b_name}_{stat}"] = np.nan

                results.append(res)

    gc.collect()
    return pd.DataFrame(results)


def extract_optical_features(shp_path, tif_path, output_csv):
    print(f"      [*] 正在加载矢量并投递至 {MAX_WORKERS} 核计算引擎...")
    gdf = gpd.read_file(shp_path)

    with rasterio.open(tif_path) as src:
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

    bands = ['NDVI', 'EVI', 'GCVI', 'SAVI', 'ISI', 'UBI', 'NDWI']

    num_chunks = MAX_WORKERS * 4
    chunks = np.array_split(gdf, num_chunks)
    tasks = [(chunk, tif_path, bands) for chunk in chunks]

    df_list = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for i, df_res in enumerate(executor.map(_extract_chunk, tasks), 1):
            df_list.append(df_res)
            print(f"        -> 批次 {i}/{num_chunks} 并发提取完成")

    final_df = pd.concat(df_list, ignore_index=True)
    stats = ['mean', 'median', 'cv', 'skew', 'kurt', 'q25', 'q75', 'valid_count', 'valid_ratio']
    final_df = final_df[['parcel_id'] + [f"{b}_{s}" for b in bands for s in stats]]
    final_df.to_csv(output_csv, index=False, encoding='utf-8-sig')


# ==========================================
# 3. 主引擎
# ==========================================
def main_pipeline():
    print("=" * 65)
    print(f"启动流 B: 高分一号特征提取 | (进程数: {MAX_WORKERS})")
    print("=" * 65)

    month_folders = sorted([f for f in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, f))])
    for month_folder in month_folders:
        print(f"\n>>> 正在处理时序节点: {month_folder}")
        folder_path = os.path.join(RAW_DIR, month_folder)

        # 巨型镶嵌图依然存放在 process 根目录
        mosaic_tif_path = os.path.join(PROCESS_DIR, f"Optical_Mosaic_{month_folder}.tif")
        out_csv_path = os.path.join(OUT_DIR, f"Optical_Features_{month_folder}.csv")

        # 解析月份名称并建立独立的临时收纳仓
        month_str = month_folder.replace('out_', '')
        month_temp_dir = os.path.join(PROCESS_DIR, f"out_tif_{month_str}")

        if os.path.exists(out_csv_path) and os.path.getsize(out_csv_path) > 0:
            print(f"  [跳过] 发现 {month_folder} 月特征 CSV 已存在，秒速越过该节点！")
            continue
        if os.path.exists(out_csv_path):
            print(f"  [重算] 发现 {month_folder} 月特征 CSV 为空文件，将重新生成。")

        if os.path.exists(mosaic_tif_path):
            print(f"  [系统检测] 发现 {month_folder} 月巨型镶嵌矩阵已存在，直达特征并发提取！")
        else:
            dat_files = glob.glob(os.path.join(folder_path, '*.dat'))
            if not dat_files: continue

            # 创建独立收纳仓
            os.makedirs(month_temp_dir, exist_ok=True)

            month_tif_list = []
            print(f"  [1/3] 启动多进程解算，临时文件将收纳至: {month_temp_dir} ...")
            futures_map = {}
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                for dat in dat_files:
                    basename = os.path.basename(dat).replace('.dat', '')
                    # 临时文件全部定向生成到专属子文件夹中
                    out_tif = os.path.join(month_temp_dir, f"Temp_{basename}_indices.tif")
                    future = executor.submit(calculate_gf1_indices, dat, out_tif)
                    futures_map[future] = basename

                for future in as_completed(futures_map):
                    basename = futures_map[future]
                    out_tif, success, error_msg = future.result()
                    if success:
                        month_tif_list.append(out_tif)
                        print(f"      [√] 解算完成: {basename[:25]}...")
                    else:
                        print(f"      [X] 解算失败: {error_msg}")

            print(f"  [2/3] 启动内存安全型 MVC 镶嵌 (合并 {len(month_tif_list)} 景影像)...")
            memory_safe_mosaic(month_tif_list, mosaic_tif_path, chunk_size=MOSAIC_CHUNK_SIZE)
            print(f"      [√] 巨幅镶嵌完成！")

            # 阅后即焚：清理临时文件
            for tif in month_tif_list:
                if os.path.exists(tif): os.remove(tif)

            # 把临时文件夹也删了，保持绝对的整洁
            try:
                os.rmdir(month_temp_dir)
            except OSError:
                pass  # 如果因其他原因文件夹非空，则保留

        print(f"  [3/3] 正在全速提取 7 大光学指数的 47 维局部特征矩阵...")
        extract_optical_features(SHP_PATH, mosaic_tif_path, out_csv_path)
        print(f"  [√] {month_folder} 节点特征矩阵全线提存！")


if __name__ == "__main__":
    main_pipeline()
