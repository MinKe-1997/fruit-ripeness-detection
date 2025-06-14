# -*- coding = utf-8 -*-
# @Time: 2025/6/13 下午1:07
# @Author: 手可摘星辰
# @File: predict.py
# @Email: 1090461393@qq.com
# @SoftWare: PyCharm

# predict.py
from ultralytics import YOLO


def detect_tomato_ripeness(model_path: str, source: str, conf_threshold: float = 0.3, save_result: bool = True,
                           show_result: bool = True):
    """
    使用训练好的 YOLOv8 模型进行推理检测。

    参数：
    - model_path: 模型权重路径，例如 'runs/tomato_ripeness/weights/best.pt'
    - source: 图片或图片目录路径，例如 'test.jpg' 或 'test_images/'
    - conf_threshold: 置信度阈值，默认为0.3
    - save_result: 是否保存检测结果，默认为True
    - show_result: 是否显示检测结果窗口，默认为True
    """
    model = YOLO(model_path)
    results = model.predict(
        source=source,
        conf=conf_threshold,
        save=save_result,
        show=show_result
    )
    return results
