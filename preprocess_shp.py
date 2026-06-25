import os
import geopandas as gpd
import warnings

from project_config import CONFIG

warnings.filterwarnings("ignore")

# ==========================================
# 0. 基础路径配置
# ==========================================
INPUT_SHP = CONFIG.raw_landuse_shp
OUTPUT_SHP = CONFIG.agri_parcels_shp


def preprocess_shapefile():
    print("=" * 60)
    print("启动 Shp 预处理：纯净地类提纯与原生主键锁定")
    print("=" * 60)

    print(f"\n>>> 1. 正在读取原始全要素矢量文件: {INPUT_SHP}")
    try:
        gdf = gpd.read_file(INPUT_SHP)
        initial_count = len(gdf)
        print(f"  [成功] 共读取到 {initial_count} 个原始要素。")
    except Exception as e:
        print(f"  [致命错误] 读取 Shapefile 失败: {e}")
        return

    print("\n>>> 2. 正在执行流 A 先验过滤 (依据 g10_cls_id 提取耕地 2 和园地 8)...")
    if 'g10_cls_id' in gdf.columns:
        mask_target = gdf['g10_cls_id'].isin([2, 8])

        arable_count = (gdf['g10_cls_id'] == 2).sum()
        orchard_count = (gdf['g10_cls_id'] == 8).sum()

        gdf = gdf[mask_target].copy()
        target_count = len(gdf)

        print(f"  [清洗报告] 剔除了 {initial_count - target_count} 个无关非农图斑。")
        print(f"  [清洗报告] 成功锁定目标图斑 {target_count} 个。")
        print(f"      -> 其中 [耕地 ID=2] 数量: {arable_count} 个")
        print(f"      -> 其中 [园地 ID=8] 数量: {orchard_count} 个")
    else:
        print("  [警告] 未找到 g10_cls_id 字段，跳过地类过滤！请检查数据。")
        target_count = len(gdf)

    print("\n>>> 3. 正在进行几何拓扑清洗...")
    gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty].copy()
    valid_count = len(gdf)
    if target_count != valid_count:
        print(f"  [警告] 清除了 {target_count - valid_count} 个破损或无效的几何要素。")

    print("\n>>> 4. 正在验证并锁定原生主键 (parcel_id)...")
    if 'parcel_id' in gdf.columns:
        # 强制转换为 String 字符串类型，彻底杜绝 Pandas 后续 Merge 时的科学计数法越界问题
        gdf['parcel_id'] = gdf['parcel_id'].astype(str)
    else:
        print("  [致命错误] 未找到 parcel_id 字段，请检查原始数据结构！")
        return

    # 强制 parcel_id 置顶排在第一列，方便在 GIS 软件中查阅
    cols = gdf.columns.tolist()
    cols = ['parcel_id'] + [c for c in cols if c != 'parcel_id']
    gdf = gdf[cols]

    print(f"\n>>> 5. 正在固化并保存农业基准矢量文件: {OUTPUT_SHP}")
    try:
        gdf.to_file(OUTPUT_SHP, driver='ESRI Shapefile', encoding='utf-8')
        print("\n" + "=" * 60)
        print("  [大功告成] 预处理圆满结束！原生 parcel_id 已全线贯通。")
        print("=" * 60)
    except Exception as e:
        print(f"  [错误] 保存文件失败: {e}")


if __name__ == "__main__":
    preprocess_shapefile()
