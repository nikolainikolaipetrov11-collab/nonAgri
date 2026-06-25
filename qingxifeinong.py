"""
    把非农地块从非粮和主粮地块中剔除
"""
import pandas as pd
from pathlib import Path

from project_config import CONFIG

NON_AGRI_PATH = CONFIG.date_out("non_agri", "非农.csv")
TARGET_FILES = {
    "01": CONFIG.split_report("01_Safe_Grain_Parcels.csv"),
    "04": CONFIG.split_report("04_Final_Trustworthy_NonGrain.csv"),
}

# 1. 读取非农剔除名单
non_agri = pd.read_csv(NON_AGRI_PATH, dtype={'parcel_id': str})
exclude_ids = set(non_agri.loc[non_agri['label'].isin([1, 2]), 'parcel_id'])
print(f"非农剔除地块数：{len(exclude_ids)}")

# 2. 对每个目标文件进行过滤并保存
for key, path in TARGET_FILES.items():
    df = pd.read_csv(path, dtype={'parcel_id': str})
    before = len(df)
    df_filtered = df[~df['parcel_id'].isin(exclude_ids)]
    after = len(df_filtered)
    print(f"{key} 过滤：{before} → {after}")
    # 保存到原位置（可改为新文件名）
    df_filtered.to_csv(path.with_stem(path.stem + "_filtered"), index=False, encoding='utf-8-sig')
