# -*- coding = utf-8 -*-
# @Time: 2025/6/13 下午2:07
# @Author: 手可摘星辰
# @File: create_data.py
# @Email: 1090461393@qq.com
# @SoftWare: PyCharm

import os
import shutil
import kagglehub

# 1. 下载数据集
dataset_path = kagglehub.dataset_download("asadullahprl/fruits-ripeness-classification-dataset")

print("原始下载路径:", dataset_path)

# 2. 你要移动到的目标目录
target_dir = "/Volumes/柯影数智/研发部/番茄成熟检测/dataset/fruits_ripeness"

# 如果目标目录存在就清空（可选）
if os.path.exists(target_dir):
    shutil.rmtree(target_dir)

# 创建目标目录
os.makedirs(target_dir, exist_ok=True)

# 3. 复制整个数据集内容到目标路径
shutil.copytree(dataset_path, target_dir, dirs_exist_ok=True)

print("已成功移动到指定目录:", target_dir)
