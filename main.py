# -*- coding = utf-8 -*-
# @Time: 2025/6/13 下午1:25
# @Author: 手可摘星辰
# @File: main.py
# @Email: 1090461393@qq.com
# @SoftWare: PyCharm

from app.utils import create_data_yaml
from app.train import train_tomato_ripeness_model
from app.predict import detect_tomato_ripeness


def main():
    # Step 1: 创建 data.yam
    create_data_yaml(
        save_path='dataset_yolo/tomato_ripeness.yaml',
        train_path='images/train',
        val_path='images/test',
        nc=3,
        class_names={0: 'unripe', 1: 'ripe', 2: 'overripe'}

    )
    # Step 2: 初始化模型并训练
    train_tomato_ripeness_model(data_yaml='dataset_yolo/tomato_ripeness.yaml',
                                epochs=5)
    # 推理检测，支持图像或目录
    detect_tomato_ripeness(
        model_path='runs/tomato_ripeness/weights/best.pt',
        source='source/test_1.jpeg')


if __name__ == '__main__':
    main()
