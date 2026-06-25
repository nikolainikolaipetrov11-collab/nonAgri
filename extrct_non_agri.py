"""
    阶段0：用于把非农结果提取出来
"""
import pandas as pd
import logging
from pathlib import Path

# 配置工业级日志打印
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def extract_non_agri_parcels():
    logging.info("🚀 启动非农靶点极速提取管线...")

    # ==========================================
    # 1. 路径路由配置
    # ==========================================
    # 输入的带有非农标签的总表
    INPUT_CSV = Path("E:/test/date/out/non_agri/parcels_growth_season_detected.csv")

    # 提取后的输出路径
    OUTPUT_DIR = Path("E:/test/date/out/non_agri")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV = OUTPUT_DIR / "pure_non_agri_targets.csv"

    if not INPUT_CSV.exists():
        logging.error(f"❌ 未找到输入文件: {INPUT_CSV}")
        return

    # ==========================================
    # 2. 数据装载与强类型检查
    # ==========================================
    logging.info("加载原始检测结果表...")
    df = pd.read_csv(INPUT_CSV)

    # 检查核心字段是否存在
    required_cols = ['parcel_id', 'label']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logging.error(f"❌ 数据表缺失关键字段: {missing_cols}")
        logging.error(f"💡 当前表头为: {df.columns.tolist()}")
        return

    # ==========================================
    # 3. 核心提取逻辑 (Filter & Subset)
    # ==========================================
    logging.info("⚙️ 正在执行标签过滤 (label == 1)...")

    # 仅保留 label 为 1（非农）的地块
    filtered_df = df[df['label'] == 1].copy()

    # 仅保留 parcel_id 和 label 两个字段
    final_df = filtered_df[['parcel_id', 'label']]

    logging.info(f"-> 提取完成！共发现 {len(final_df)} 个纯非农地块 (原表总计: {len(df)} 块)。")

    # ==========================================
    # 4. 隔离落盘
    # ==========================================
    final_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    logging.info(f"💾 纯净非农白名单已安全落盘至: {OUTPUT_CSV}")


if __name__ == "__main__":
    extract_non_agri_parcels()