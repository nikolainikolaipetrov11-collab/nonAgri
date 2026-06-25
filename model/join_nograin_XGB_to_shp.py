"""
    阶段二：非粮可信地块精细图层写回 Shapefile
    核心任务：读取 XGB 可信判定 CSV → 向量化清洗 → Inner Join → 字段安全映射与写出
    model/join_nograin_XGB_to_shp.py
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
    logging.info("🚀 启动非粮可信地块空间连接管线 (XGB 判定专项)...")

    # ==========================================
    # 1. 路径路由配置
    # ==========================================
    # 改为使用 04 号最终可信非粮 CSV
    CSV_PATH = CONFIG.split_report("04_Final_Trustworthy_NonGrain.csv")
    ORIGINAL_SHP = CONFIG.parcel_shp

    OUTPUT_DIR = CONFIG.non_grain_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_SHP = OUTPUT_DIR / "NonGrain_Trusted_Anomalies.shp"

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"❌ 未找到可信非粮台账 CSV: {CSV_PATH}")
    if not ORIGINAL_SHP.exists():
        raise FileNotFoundError(f"❌ 未找到原始 Shapefile: {ORIGINAL_SHP}")

    # ==========================================
    # 2. 空间与属性双大数据底座安全装载
    # ==========================================
    logging.info("📥 正在加载非粮可信属性表 (类型保护已开启)...")
    csv_df = pd.read_csv(CSV_PATH, dtype={'parcel_id': str})

    logging.info("🗺️ 正在读取原始 Shapefile...")
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

    logging.info("🧹 正在执行毫秒级向量化 ID 清洗...")
    gdf[shp_id_col] = gdf[shp_id_col].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    csv_df['parcel_id'] = csv_df['parcel_id'].str.replace(r'\.0$', '', regex=True).str.strip()

    # ==========================================
    # 4. 执行内连接 - 防字段冲突版
    # ==========================================
    # 提前列出 CSV 中我们关心的字段（除去连接键 parcel_id）
    csv_target_fields = ['XGB_Semantic', 'Reason', 'Margin']

    # 检查原始 SHP 是否有与这些字段重名的列，若有，先删除或重命名
    conflict_cols = [c for c in csv_target_fields if c in gdf.columns]
    if conflict_cols:
        logging.warning(f"⚠️ 原始 SHP 中存在与 CSV 冲突的字段: {conflict_cols}，将自动删除以避免混淆。")
        gdf = gdf.drop(columns=conflict_cols)

    logging.info("🔗 正在执行全局空间矩阵 Hash 连接 (Inner Join)...")
    merged_gdf = gdf.merge(csv_df, left_on=shp_id_col, right_on='parcel_id', how='inner')
    logging.info(f"-> 提纯成功！当前可信非粮专题图层共保留: {len(merged_gdf)} 块核心地块。")

    # ==========================================
    # 5. 字段整理与最终筛选
    # ==========================================
    # 删除可能重复的原始连接键
    if shp_id_col != 'parcel_id' and shp_id_col in merged_gdf.columns:
        merged_gdf = merged_gdf.drop(columns=[shp_id_col])

    # 严格遵循 10 字符命名法！
    shp_column_mapping = {
        'XGB_Semantic': 'xgb_sem',  # 最终定性分类 (如：果树与林地)
        'XGB_Confidence': 'xgb_conf',  # XGB单边置信度
        'Joint_Conf': 'joint_conf',  # 终极双重融合置信度
        'Reason': 'xgb_reason',  # SHAP 可解释性归因
        'Margin': 'xgb_margin'  # 纠结度/主动学习指标
    }

    # 执行重命名
    merged_gdf = merged_gdf.rename(columns=shp_column_mapping)

    # 圈定最终输出列 (务必带上 geometry)
    keep_cols = ['parcel_id', 'xgb_sem', 'xgb_conf', 'joint_conf', 'xgb_reason', 'xgb_margin', 'geometry']

    # 防止上游某些列缺失导致的 KeyError
    existing_keep_cols = [c for c in keep_cols if c in merged_gdf.columns]
    merged_gdf = merged_gdf[existing_keep_cols]

    # 终极防弹衣：处理可能的 NaN，防止 GDAL 写入报错
    for col in merged_gdf.columns:
        if col == 'geometry':
            continue
        if merged_gdf[col].dtype == 'object':
            merged_gdf[col] = merged_gdf[col].fillna("")  # 字符串填空
        else:
            merged_gdf[col] = merged_gdf[col].fillna(-9999)  # 数值填 -9999


    # ==========================================
    # 6. 新生成 Shapefile 隔离落盘
    # ==========================================
    logging.info("💾 正在写出最终可信非粮专题 Shapefile 图层...")
    merged_gdf.to_file(OUTPUT_SHP, driver="ESRI Shapefile", encoding="utf-8")

    logging.info("🎉 霍城县非粮化可信地块地理资产构建成功！")
    logging.info(f"   --> 衍生专题图层位置: {OUTPUT_SHP.name}")
    logging.info("💡 提示：可将此 SHP 拖入 GIS 软件，使用 'xgb_sem' 或 'Reason' 字段进行可视化与分析。")

if __name__ == "__main__":
    join_nongrain_csv_to_shapefile()
