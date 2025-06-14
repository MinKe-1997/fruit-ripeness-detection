# -*- coding = utf-8 -*-
# @Time: 2025/6/13 下午1:08
# @Author: 手可摘星辰
# @File: utils.py
# @Email: 1090461393@qq.com
# @SoftWare: PyCharm

import os
import yaml

import os
import yaml


def create_data_yaml(
        save_path='../dataset/tomato_ripeness.yaml',
        train_path='../dataset/images/train',
        val_path='../dataset/images/val',
        nc=3,
        class_names={0: 'unripe', 1: 'semi_ripe', 2: 'ripe'}
):
    """
    创建 YOLOv8 格式的数据集配置 data.yaml 文件。

    参数：
    - save_path: 保存 data.yaml 的完整路径
    - train_path: 训练图像目录路径
    - val_path: 验证图像目录路径
    - nc: 类别数量
    - class_names: 类别字典，键为整数索引，值为类别名称
    """
    data = {
        'train': train_path,
        'val': val_path,
        'nc': nc,
        'names': list(class_names.values())
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as f:
        yaml.dump(data, f, allow_unicode=True)

    print(f"[✔] 已成功生成 data.yaml 文件于: {save_path}")


if __name__ == '__main__':
    create_data_yaml()
