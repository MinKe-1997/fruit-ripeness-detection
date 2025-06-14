# -*- coding = utf-8 -*-
# @Time: 2025/6/13 下午1:07
# @Author: 手可摘星辰
# @File: train.py
# @Email: 1090461393@qq.com
# @SoftWare: PyCharm

from ultralytics import YOLO
import os

def train_tomato_ripeness_model(
        model_name='yolov8n.pt',
        data_yaml='dataset/data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        project='runs',
        name='tomato_ripeness',
        exist_ok=True,
        save_dir='saved_models'
):
    """
    使用 YOLOv8 模型训练番茄成熟度检测模型，并保存训练好的权重。

    参数：
    - model_name: 预训练模型权重，如 yolov8n.pt、yolov8s.pt 等
    - data_yaml: 数据集配置文件路径
    - epochs: 训练轮数
    - imgsz: 输入图像尺寸
    - batch: 批次大小
    - project: 保存结果的项目文件夹（训练过程生成文件夹）
    - name: 本次训练名称（用于结果文件夹命名）
    - exist_ok: 是否覆盖已存在结果文件夹
    - save_dir: 训练完成后保存模型权重的目录
    """
    model = YOLO(model_name)
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        exist_ok=exist_ok
    )

    # 训练完成后，模型权重路径（默认best.pt）
    trained_weights_path = os.path.join(project, name, 'weights', 'best.pt')
    print(f"训练完成，权重文件位置: {trained_weights_path}")

    # 调用保存模型函数，拷贝到自定义目录
    save_model(trained_weights_path, save_dir, name)


def save_model(source_path, save_dir, model_name):
    """
    保存训练好的模型权重到指定目录。

    参数：
    - source_path: 源权重文件路径（训练结果best.pt路径）
    - save_dir: 目标保存目录
    - model_name: 模型名称，作为文件夹或文件名区分
    """
    import shutil

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dest_path = os.path.join(save_dir, f"{model_name}_best.pt")

    try:
        shutil.copy2(source_path, dest_path)
        print(f"模型权重已保存到: {dest_path}")
    except Exception as e:
        print(f"保存模型时出错: {e}")
