"""
    提取非农地块：筛选 label 为 1 和 2 的记录并单独保存
    输出：date/out/non_agri/非农_剔除列表.csv
"""
import pandas as pd
from pathlib import Path


def extract_non_agri_exclude():
    DATA_ROOT = Path("E:/test")
    NON_AGRI_PATH = DATA_ROOT / "date/out/non_agri/非农.csv"
    OUTPUT_PATH = DATA_ROOT / "date/out/non_agri/非农_剔除列表.csv"

    # 读取原始非农 CSV，锁定 parcel_id 为字符串
    df = pd.read_csv(NON_AGRI_PATH, dtype={'parcel_id': str})

    # 筛选 label 为 1 或 2
    exclude_df = df[df['label'].isin([1, 2])].copy()

    # 保存
    exclude_df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')

    print(f"✅ 已提取非农剔除地块 {len(exclude_df)} 条，保存至：{OUTPUT_PATH}")


if __name__ == "__main__":
    extract_non_agri_exclude()