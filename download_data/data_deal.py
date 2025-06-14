# -*- coding = utf-8 -*-
# @Time: 2025/6/13 下午2:29
# @Author: 手可摘星辰
# @File: data_deal.py
# @Email: 1090461393@qq.com
# @SoftWare: PyCharm

import os
import shutil
from PIL import Image

# 类别关键词
keywords = ['overripe', 'ripe', 'unripe']
class2id = {k: i for i, k in enumerate(keywords)}

source_root = '../dataset'
target_root = '../dataset_yolo'

# 创建 YOLOv8 格式目录结构
for split in ['train', 'test']:
    os.makedirs(os.path.join(target_root, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(target_root, 'labels', split), exist_ok=True)

def detect_class_from_filename(filename: str):
    fname = filename.lower()
    for keyword in keywords:
        if keyword in fname:
            return keyword
    return None

def create_yolo_label(image_path, class_id):
    try:
        with Image.open(image_path) as img:
            w, h = img.size
        return f"{class_id} 0.5 0.5 1.0 1.0\n"
    except Exception as e:
        print(f"❌ 无法读取图像：{image_path}，错误：{e}")
        return None

# 遍历 train 和 test
for split in ['train', 'test']:
    split_path = os.path.join(source_root, split)
    for root, _, files in os.walk(split_path):
        for fname in files:
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            cls_keyword = detect_class_from_filename(fname)
            if cls_keyword is None:
                print(f"⚠️ 跳过未识别文件: {fname}")
                continue

            class_id = class2id[cls_keyword]
            src_path = os.path.join(root, fname)

            # 统一输出文件名（避免子目录干扰）
            clean_name = f"{split}_{cls_keyword}_{fname}"
            dst_img_path = os.path.join(target_root, 'images', split, clean_name)
            dst_lbl_path = os.path.join(target_root, 'labels', split, os.path.splitext(clean_name)[0] + '.txt')

            shutil.copy2(src_path, dst_img_path)

            label = create_yolo_label(src_path, class_id)
            if label:
                with open(dst_lbl_path, 'w') as f:
                    f.write(label)

print("✅ 数据集已成功转换为 YOLOv8 所需格式，存放于 dataset_yolo/")