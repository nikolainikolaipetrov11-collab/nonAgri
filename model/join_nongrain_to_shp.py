"""
    阶段二：把二阶段非粮地块精细图层写回 Shapefile (架构师重构版)
    核心任务：防弹级读取 -> 向量化清洗 -> 绝对纯净 Inner Join -> 全字段 DBF 安全映射
    model/join_nongrain_to_shp.py
"""
import pandas as pd
import geopandas as gpd
import logging
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from project_config import CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def join_nongrain_csv_to_shapefile():
    logging.info("🚀 启动 GIS 空间属性级联连接管线 (非粮化精细图层专项)...")

    # ==========================================
    # 1. 路径路由配置
    # ==========================================
    CSV_PATH = CONFIG.split_report("03_Non_Grain_GMM_Final_Semantic.csv")
    ORIGINAL_SHP = CONFIG.parcel_shp

    OUTPUT_DIR = CONFIG.non_grain_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_SHP = OUTPUT_DIR / "Non_Grain_Anomalies.shp"

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"❌ 未找到非粮台账 CSV: {CSV_PATH}")
    if not ORIGINAL_SHP.exists():
        raise FileNotFoundError(f"❌ 未找到原始 Shapefile: {ORIGINAL_SHP}")

    # ==========================================
    # 2. 空间与属性双大数据底座安全装载
    # ==========================================
    # 坚守 ID 鄙视链，读取时强制锁定 String，防止 Pandas 瞎猜类型
    logging.info("📥 正在加载非粮定性属性表 (已开启类型保护)...")
    csv_df = pd.read_csv(CSV_PATH, dtype={'parcel_id': str})

    logging.info("🗺️ 正在读取原始 Shapefile（这可能需要一些时间）...")
    gdf = gpd.read_file(ORIGINAL_SHP)
    logging.info(f"-> 成功加载原始矢量，全县地块总数: {len(gdf)} 块")

    # ==========================================
    # 3. 毫秒级双端 ID 强力清洗与对齐
    # ==========================================
    def find_shp_id_col(columns):
        cols_lower = {c.lower(): c for c in columns}
        for tgt in ['parcel_id', 'id', 'objectid', 'fid', 'dkm']:
            if tgt in cols_lower: return cols_lower[tgt]
        return columns[0]

    shp_id_col = find_shp_id_col(gdf.columns)
    logging.info(f"-> 检测到原始 SHP 关键连接列为: '{shp_id_col}'")

    # 使用极速向量化正则替换
    logging.info("🧹 正在执行毫秒级向量化 ID 清洗...")
    gdf[shp_id_col] = gdf[shp_id_col].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    csv_df['parcel_id'] = csv_df['parcel_id'].str.replace(r'\.0$', '', regex=True).str.strip()

    # ==========================================
    # 4. 执行内连接 (Inner Join) - 纯净提纯！
    # ==========================================
    logging.info("🔗 正在执行全局空间矩阵 Hash 连接 (Inner Join)...")
    merged_gdf = gdf.merge(csv_df, left_on=shp_id_col, right_on='parcel_id', how='inner')
    logging.info(f"-> 提纯成功！当前非粮违规专题图层共保留: {len(merged_gdf)} 块核心地块。")

    # ==========================================
    # 5. 突破 SHP 物理限制：全量 DBF 字段映射
    # ==========================================
    if 'parcel_id' in merged_gdf.columns and shp_id_col != 'parcel_id':
        merged_gdf = merged_gdf.drop(columns=['parcel_id'])

    # 补全遗漏的 GMM_Cluster_ID，彻底卡死 10 字符上限
    shp_column_mapping = {
        'loss_MSE': 'loss_mse',
        'GMM_Cluster_ID': 'gmm_id',         # 补全：簇类ID
        'GMM_Semantic_Label': 'gmm_label',  # 非粮分类结果
        'GMM_Confidence': 'gmm_conf'        # 判定置信度
    }

    merged_gdf = merged_gdf.rename(columns=shp_column_mapping)

    # ==========================================
    # 6. 新生成 Shapefile 隔离落盘
    # ==========================================
    logging.info("💾 正在向非粮安全区写出最终专题 Shapefile 图层...")
    merged_gdf.to_file(OUTPUT_SHP, driver="ESRI Shapefile", encoding="utf-8")

    logging.info("🎉 霍城县非粮化专题地理资产构建成功！")
    logging.info(f"   --> 衍生专题图层位置: {OUTPUT_SHP.name}")
    logging.info("💡 架构师提示：现在可以直接将此 SHP 拖入 ArcGIS，根据 'gmm_label' 字段进行唯一值赋色进行执法调度！")

if __name__ == "__main__":
    join_nongrain_csv_to_shapefile()
