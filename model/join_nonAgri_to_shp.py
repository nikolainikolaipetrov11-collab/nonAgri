# model/join_nonAgri_to_shp.py
import os
import pandas as pd
import geopandas as gpd  # 工业级空间矢量处理核心库
import logging
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from project_config import CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def join_nonagri_csv_to_shapefile():
    logging.info("🚀 启动 GIS 空间属性级联连接管线 (【纯非农】底库提取专项)...")

    # ==========================================
    # 1. 路径路由配置
    # ==========================================
    CSV_PATH = CONFIG.split_report("00_Pure_Non_Agri.csv")
    ORIGINAL_SHP = CONFIG.agri_parcels_shp

    OUTPUT_DIR = CONFIG.non_agri_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_SHP = OUTPUT_DIR / "00_Pure_Non_Agri.shp"

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"未找到 CSV 属性表: {CSV_PATH}")
    if not ORIGINAL_SHP.exists():
        logging.error(f"❌ 未找到原始 Shapefile: {ORIGINAL_SHP}")
        return

    # ==========================================
    # 2. 空间与属性双大数据底座
    # ==========================================
    logging.info("📥 正在加载纯非农靶点属性表...")
    csv_df = pd.read_csv(CSV_PATH)

    logging.info("🗺️ 正在读取原始 Shapefile（这可能需要一些时间）...")
    gdf = gpd.read_file(ORIGINAL_SHP)
    logging.info(f"-> 成功加载原始矢量，全县地块总数: {len(gdf)} 块")

    # ==========================================
    # 3. 双端 ID 类型强力清洗与对齐
    # ==========================================
    def clean_id(val):
        s = str(val).strip()
        if s.endswith('.0'): s = s[:-2]
        return s

    def find_shp_id_col(columns):
        cols_lower = {c.lower(): c for c in columns}
        for tgt in ['parcel_id', 'id', 'objectid', 'fid', 'dkm']:
            if tgt in cols_lower: return cols_lower[tgt]
        return columns[0]

    shp_id_col = find_shp_id_col(gdf.columns)
    logging.info(f"-> 检测到原始 SHP 关键连接列为: '{shp_id_col}'")

    # 强制统一 ID 格式
    gdf[shp_id_col] = gdf[shp_id_col].apply(clean_id)
    csv_df['parcel_id'] = csv_df['parcel_id'].apply(clean_id)

    # ==========================================
    # 4. 执行内连接 (Inner Join) - 纯净提纯！
    # ==========================================
    logging.info("🔗 正在执行全局空间矩阵 Hash 连接 (Inner Join)...")
    merged_gdf = gdf.merge(csv_df, left_on=shp_id_col, right_on='parcel_id', how='inner')

    logging.info(f"-> 提纯成功！当前非农底库图层共成功匹配保留: {len(merged_gdf)} 块核心地块。")

    # ==========================================
    # 5. 突破 SHP 物理限制与关键字避让
    # ==========================================
    if 'parcel_id' in merged_gdf.columns and shp_id_col != 'parcel_id':
        merged_gdf = merged_gdf.drop(columns=['parcel_id'])

    # 🌟 核心修正：将 'class' 字段重命名为 'is_nonagri'
    # 既避开了代码保留字，又完美符合 10 字符 DBF 限制
    shp_column_mapping = {
        'class': 'is_nonagri'
    }

    merged_gdf = merged_gdf.rename(columns=shp_column_mapping)

    # ==========================================
    # 6. 新生成 Shapefile 隔离落盘
    # ==========================================
    logging.info("💾 正在向隔离区写出最终【非农黑名单】专题 Shapefile 图层...")

    merged_gdf.to_file(OUTPUT_SHP, driver="ESRI Shapefile", encoding="utf-8")

    logging.info("🎉 纯净非农专题图层构建成功！")
    logging.info(f"   --> 衍生专题图层位置: {OUTPUT_SHP}")


if __name__ == "__main__":
    join_nonagri_csv_to_shapefile()
