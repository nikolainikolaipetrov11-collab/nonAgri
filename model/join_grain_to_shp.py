"""
    model/join_grain_to_shp.py
    阶段一：把初筛的安全粮食地块隔离并写回 Shapefile (架构师重构版)
    核心任务：向量化 ID 清洗 -> Inner Join 严格几何过滤 -> DBF 字段截断 -> 纯净 SHP 落盘
"""
import os
import pandas as pd
import geopandas as gpd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def join_csv_to_shapefile():
    logging.info("🚀 启动 GIS 空间属性级联连接管线 (严格几何隔离模式)...")

    # ==========================================
    # 1. 路径路由配置
    # ==========================================
    DATA_ROOT = Path("E:/test")
    CSV_PATH = DATA_ROOT / "date/out/split_reports/01_Safe_Grain_Parcels.csv"
    ORIGINAL_SHP = DATA_ROOT / "shp/huocheng_dk_260605.shp"

    OUTPUT_DIR = DATA_ROOT / "shp/safe_grain"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_SHP = OUTPUT_DIR / "Safe_Grain_Parcels.shp"

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"❌ 未找到 CSV 属性表: {CSV_PATH}")
    if not ORIGINAL_SHP.exists():
        raise FileNotFoundError(f"❌ 未找到原始 Shapefile: {ORIGINAL_SHP}")

    # ==========================================
    # 2. 空间与属性双大数据底座安全装载
    # ==========================================
    logging.info("📦 正在加载 Safe Grain 属性台账...")
    csv_df = pd.read_csv(CSV_PATH, dtype={'parcel_id': str})

    logging.info("🗺️ 正在读取原始 Shapefile（这可能需要一些时间，请保持耐心）...")
    gdf = gpd.read_file(ORIGINAL_SHP)
    logging.info(f"-> 成功加载原始矢量底座，全县总地块: {len(gdf)} 块")

    # ==========================================
    # 3. 毫秒级双端 ID 类型强力清洗与对齐
    # ==========================================
    def find_shp_id_col(columns):
        cols_lower = {c.lower(): c for c in columns}
        for tgt in ['parcel_id', 'id', 'objectid', 'fid', 'dkm']:
            if tgt in cols_lower: return cols_lower[tgt]
        return columns[0]

    shp_id_col = find_shp_id_col(gdf.columns)
    logging.info(f"-> 检测到原始 SHP 关键连接列为: '{shp_id_col}'")

    # 彻底抛弃低效的 apply，使用强大的向量化正则强杀浮点幻影
    logging.info("🧹 正在执行毫秒级向量化 ID 护城河清洗...")
    gdf[shp_id_col] = gdf[shp_id_col].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    csv_df['parcel_id'] = csv_df['parcel_id'].str.replace(r'\.0$', '', regex=True).str.strip()

    # ==========================================
    # 4. 执行内连接 (Inner Join) - 真正的空间隔离
    # ==========================================
    logging.info("🔗 正在执行 Inner Join 空间矩阵 Hash 连接...")
    # 使用 inner join！只保留在 CSV 中存在的安全粮食几何体，剔除全县无用多边形！
    merged_gdf = gdf.merge(csv_df, left_on=shp_id_col, right_on='parcel_id', how='inner')

    logging.info(f"-> 空间裁切完成！纯净安全主粮地块保留: {len(merged_gdf)} 块 (占比 {len(merged_gdf) / len(gdf) * 100:.2f}%)")

    # ==========================================
    # 5. 突破 SHP 物理限制：DBF 字段安全映射
    # ==========================================
    # (使用 errors='ignore' 防止报错)
    if shp_id_col != 'parcel_id':
        merged_gdf = merged_gdf.drop(columns=['parcel_id'], errors='ignore')

    # 10 字符映射字典，卡死 DBF 边界，避开保留字
    shp_column_mapping = {
        'MSE_Score': 'mse_score',
        'Grain_Confidence': 'grain_conf'
    }
    merged_gdf = merged_gdf.rename(columns=shp_column_mapping)

    # 以防万一 CSV 里存在空置信度，强填 -9999 防止 DBF 崩溃
    merged_gdf = merged_gdf.fillna(-9999)

    # ==========================================
    # 6. 新生成 Shapefile 隔离落盘
    # ==========================================
    logging.info(f"💾 正在向新安全区写出纯净版 Shapefile 图层...")
    # 写入文件，利用 utf-8 保护中文属性
    merged_gdf.to_file(OUTPUT_SHP, driver="ESRI Shapefile", encoding="utf-8")

    logging.info("🎉 地理决策资产构建成功！")
    logging.info(f"   --> 纯净安全主粮图层已生成: {OUTPUT_SHP.name}")

if __name__ == "__main__":
    join_csv_to_shapefile()