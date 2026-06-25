# model/dataset.py
# 数据加载器
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import logging
from pathlib import Path

# 解决 Windows 下 PyTorch 与 NumPy (MKL) 的 OpenMP 多线程冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 配置基础日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PhenologyMAEDataset(Dataset):
    def __init__(self, data_root: str, mode: str = 'train', mask_ratio: float = 0.25):
        """
        Twin-Expert MAE 时序数据集加载器 (架构师重构版)
        - 坚守 ID 鄙视链：全量强转 String，抹除浮点幻影
        - 坚守物理红线：内置原始物理值备份，供阶段二使用
        """
        super().__init__()
        self.mode = mode
        self.mask_ratio = mask_ratio

        # 1. 统一路径路由
        root_dir = Path(data_root)
        self.tensor_path = root_dir / "Dynamic_TimeSeries_Tensor.npy"
        self.wide_table_path = root_dir / "Dynamic_TimeSeries_WideTable.csv"
        self.scaler_path = root_dir / "feature_scaler.pt"

        for path in [self.tensor_path, self.wide_table_path]:
            if not path.exists():
                raise FileNotFoundError(f"❌ 致命错误: 在 {root_dir} 下未找到核心数据文件 {path.name}")

        # 2. 安全读取与异常清洗
        try:
            logging.info(f"📦 正在从 {root_dir} 加载全量张量到底层内存...")
            # 保留原始物理值备份，防止下游阶段二取不到真值！
            self._raw_physical_tensor = np.load(self.tensor_path)
            full_wide_table = pd.read_csv(self.wide_table_path)
        except Exception as e:
            raise RuntimeError(f"💥 数据加载崩溃，底层报错: {e}")

        if len(self._raw_physical_tensor) != len(full_wide_table):
            raise ValueError(
                f"⚖️ 数据不对齐: 张量有 {len(self._raw_physical_tensor)} 样本，宽表有 {len(full_wide_table)} 行！")

        if np.isnan(self._raw_physical_tensor).any() or np.isinf(self._raw_physical_tensor).any():
            logging.warning("⚠️ 检测到内存张量中包含异常值！已自动执行安全清洗并置零。")
            self._raw_physical_tensor = np.nan_to_num(self._raw_physical_tensor, nan=0.0, posinf=0.0, neginf=0.0)

        # 工作张量 (用于神经网络的 Z-score 副本)
        full_tensor_normalized = self._raw_physical_tensor.copy()

        # 3. 全局 Z-Score 归一化
        if self.mode == 'train':
            mean = np.mean(full_tensor_normalized, axis=(0, 1), keepdims=True)
            std = np.std(full_tensor_normalized, axis=(0, 1), keepdims=True)
            std[std == 0] = 1e-8

            torch.save({'mean': mean, 'std': std}, self.scaler_path)
            logging.info(f"📈 已计算并保存全局归一化参数至: {self.scaler_path.name}")
            full_tensor_normalized = (full_tensor_normalized - mean) / std
        else:
            if not self.scaler_path.exists():
                raise FileNotFoundError(f"🔍 推理模式下未找到 {self.scaler_path.name}，请先运行 'train' 模式！")

            scaler_dict = torch.load(self.scaler_path, weights_only=False)
            full_tensor_normalized = (full_tensor_normalized - scaler_dict['mean']) / scaler_dict['std']
            logging.info(f"📏 已成功应用 {self.scaler_path.name} 进行一致性特征缩放。")

        # 4. 自适应查找 ID 列 & 强制破除 ID 鄙视链
        def find_id_column(df_columns):
            cols_lower = {col.lower(): col for col in df_columns}
            for target in ['parcel_id', 'id', 'objectid', 'fid']:
                if target in cols_lower:
                    return cols_lower[target]
            return df_columns[0]

        self.wide_id_col = find_id_column(full_wide_table.columns)

        # ID 全部转 String 并抹杀 '.0' 浮点幻影
        # 只抹除末尾的 .0
        full_wide_table[self.wide_id_col] = full_wide_table[self.wide_id_col].astype(str).str.replace(r'\.0$', '',
                                                                                                      regex=True).str.strip()

        # 5. 架构解耦：将过滤逻辑剥离，让 Dataset 只做数据承载，Subset 逻辑交由业务脚本决定
        self.tensor_data = full_tensor_normalized
        self.meta_data = full_wide_table.reset_index(drop=True)
        self.num_samples, self.time_steps, self.num_features = self.tensor_data.shape

        logging.info(f"✅ 数据底座加载完成：共 {self.num_samples} 个地块。核心主键列锁定为: {self.wide_id_col}")

    # ==========================================
    # 提取未经污染的绝对物理值
    # ==========================================
    def get_raw_physical_features(self, indices: list = None) -> np.ndarray:
        """
        供阶段二 (GMM/专家规则) 调用的安全出口，返回 100% 真实的物理张量！
        """
        if indices is not None:
            return self._raw_physical_tensor[indices]
        return self._raw_physical_tensor

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        original_tensor = torch.tensor(self.tensor_data[idx], dtype=torch.float32)
        parcel_id = self.meta_data.loc[idx, self.wide_id_col]
        cluster_id = int(self.meta_data.loc[idx, 'Cluster_ID']) if 'Cluster_ID' in self.meta_data.columns else 0

        mask = torch.zeros((self.time_steps, self.num_features), dtype=torch.float32)
        masked_tensor = original_tensor.clone()

        num_mask_steps = max(1, int(self.time_steps * self.mask_ratio))

        if num_mask_steps > 0 and self.mask_ratio > 0:
            masked_indices = np.random.choice(self.time_steps, num_mask_steps, replace=False)
            mask[masked_indices, :] = 1.0
            masked_tensor[masked_indices, :] = 0.0  # Z-score 后的 0 就是均值，完美掩码

        return masked_tensor, original_tensor, mask, parcel_id, cluster_id