import os
import shutil
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from PIL.Image import Image
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from ultralytics import YOLO, settings


# def log_dataset_info(X_train, X_test, classes):
#     """记录数据集元数据到MLflow"""
#     import pandas as pd
#     from PIL import Image
#
#     # 创建数据集描述
#     dataset_stats = {
#         "train_samples": len(X_train),
#         "test_samples": len(X_test),
#         "num_classes": len(classes),
#         "class_distribution": {
#             cls: (sum(1 for l in y_train if l == idx),
#                   sum(1 for l in y_test if l == idx))
#             for idx, cls in enumerate(classes)
#         }
#     }
#
#     # 记录基本统计信息
#     mlflow.log_params({
#         "dataset.train_size": dataset_stats["train_samples"],
#         "dataset.test_size": dataset_stats["test_samples"],
#         "dataset.num_classes": dataset_stats["num_classes"]
#     })
#
#     # 记录类别分布图表
#     fig, ax = plt.subplots()
#     pd.DataFrame.from_dict(dataset_stats["class_distribution"], orient="index").plot(
#         kind="bar", ax=ax, title="Class Distribution"
#     )
#     mlflow.log_figure(fig, "artifacts/class_distribution.png")
#
#     # 记录样本元数据
#     sample_meta = []
#     for path in X_train[:100] + X_test[:100]:  # 记录前200个样本的元数据
#         with Image.open(path) as img:
#             sample_meta.append({
#                 "file_name": Path(path).name,
#                 "split": "train" if path in X_train else "test",
#                 "width": img.width,
#                 "height": img.height,
#                 "channels": len(img.getbands()),
#                 "format": img.format
#             })
#
#     pd.DataFrame(sample_meta).to_csv("dataset_sample_metadata.csv", index=False)
#     mlflow.log_artifact("dataset_sample_metadata.csv", "artifacts")
#
#     # 记录数据集版本指纹
#     import hashlib
#     combined_hash = hashlib.sha256()
#     for path in sorted(X_train + X_test):
#         with open(path, "rb") as f:
#             combined_hash.update(f.read())
#     mlflow.log_param("dataset.fingerprint", combined_hash.hexdigest())


# def generate_data_quality_report(X_train, X_test):
#     from pandas_profiling import ProfileReport
#
#     # 创建分析报告
#     df = pd.DataFrame({
#         "path": X_train + X_test,
#         "split": ["train"] * len(X_train) + ["test"] * len(X_test)
#     })
#     profile = ProfileReport(df, title="Data Quality Report")
#
#     # 保存并记录
#     profile.to_file("data_quality_report.html")
#     mlflow.log_artifact("data_quality_report.html")


def log_sample_previews(X_train, X_test, num_samples=5):
    """记录样本预览图"""
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))

    # 训练集样本
    for i, path in enumerate(X_train[:num_samples]):
        img = Image.open(path)
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Train: {Path(path).name}")

    # # 测试集样本
    # for i, path in enumerate(X_test[:num_samples]):
    #     img = Image.open(path)
    #     axes[1, i].imshow(img)
    #     axes[1, i].set_title(f"Test: {Path(path).name")
    #     plt.tight_layout()
    #     mlflow.log_figure(fig, "artifacts/data_samples.png")


def prepare_yolo_dataset(X_train, X_test, y_train, y_test, classes, data_dir):
    """创建符合YOLO分类标准的数据集结构"""
    from pathlib import Path
    import shutil
    import yaml

    # 创建根目录
    dataset_dir = Path("yolo_classification")
    splits = ["train", "val"]

    # 创建分类目录结构
    for split in splits:
        for cls in classes:
            (dataset_dir / split / cls).mkdir(parents=True, exist_ok=True)

    # 创建标签编码映射
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    def copy_class_files(file_list, labels, split):
        """按类别复制文件到对应目录"""
        for img_path, label_idx in zip(file_list, labels):
            src = Path(img_path)
            class_name = classes[label_idx]

            # 目标路径：数据集目录/分割类型/类别/文件名
            dst = dataset_dir / split / class_name / src.name

            if not dst.exists():
                shutil.copy2(src, dst)

    # 处理训练集和验证集
    copy_class_files(X_train, y_train, "train")
    copy_class_files(X_test, y_test, "val")

    # 生成YAML配置文件
    yolo_config = {
        "path": str(dataset_dir.resolve()),
        "train": "train",  # 相对路径
        "val": "val",
        "nc": len(classes),
        "names": class_to_idx  # 自动生成类别映射
    }

    config_path = dataset_dir / "dataset.yaml"
    with open(config_path, "w") as f:
        yaml.dump(yolo_config, f)

    return str(dataset_dir)


def log_dataset_info(X_train, X_test, data_dir, classes, class_dist):
    """新增数据集记录专用函数"""
    with mlflow.start_run(nested=True, description="Dataset Logging"):
        # 记录数据集信息
        mlflow.log_param("dataset_path", data_dir)
        mlflow.log_param("num_classes", len(classes))
        mlflow.log_metric("train_samples", len(X_train))
        mlflow.log_metric("test_samples", len(X_test))
        mlflow.log_dict(class_dist, "class_distribution.json")
        mlflow.log_artifact("data_split.csv")


def load_data(data_dir, test_size=0.2):
    # 获取图像路径和标签
    classes = sorted(os.listdir(data_dir))
    file_paths = []
    labels = []

    for label_idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        for img_file in os.listdir(class_dir):
            file_paths.append(os.path.join(class_dir, img_file))
            labels.append(label_idx)

    # 划分数据集（保持类别分布）
    X_train, X_test, y_train, y_test = train_test_split(
        file_paths,  # 返回文件路径而不是加载全部图像
        labels,
        test_size=test_size,
        stratify=labels,  # 保持类别分布
        random_state=42
    )

    # 记录类别分布
    class_dist = {cls: (labels.count(i), y_train.count(i), y_test.count(i))
                  for i, cls in enumerate(classes)}

    # 保存划分结果到CSV
    import pandas as pd
    df = pd.DataFrame({
        "path": X_train + X_test,
        "split": ["train"] * len(X_train) + ["test"] * len(X_test),
        "label": y_train + y_test
    })
    df.to_csv("data_split.csv", index=False)

    return X_train, X_test, y_train, y_test, classes, class_dist


def train_model(data_dir, test_size=0.2):
    # 自动记录参数、指标和模型
    # mlflow.autolog()
    X_train, X_test, y_train, y_test, classes, class_dist = load_data(data_dir)
    # 使用嵌套运行记录数据集
    # log_dataset_info(X_train, X_test, data_dir, classes, class_dist)
    # mlflow.log_param("dataset.version", "v1.2")
    yaml_path = prepare_yolo_dataset(X_train, X_test, y_train, y_test, classes, data_dir)
    # 模型参数（可配置化）
    params = {
        "data": yaml_path,
        "epochs": 100,
        "close_mosaic": 5,
        "optimizer": 42,
        "seed": 20250214,
        "lr0": 0.02,
        "batch": 32,
        "val": False,
        "imgsz": 320,
        "dropout": 0.2
    }
    # mlflow.log_params(params)
    # 训练模型

    model = YOLO('pts/yolov8s-cls.yaml').load('pts/yolov8s-cls.pt')
    results = model.train(data=params.get("data"),
                          epochs=100,
                          close_mosaic=0,
                          optimizer='SGD',
                          seed=20240722,
                          lr0=0.02,
                          batch=32,
                          val=False,
                          imgsz=320,
                          dropout=0.2)
    return model


if __name__ == "__main__":
    # 初始化MLflow实验
    mlflow.set_experiment("Tube_Classification")
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")
    settings.update({"mlflow": True})
    print(f"当前活动运行: {mlflow.active_run()}")  # 调试运行状态
    train_model("dataset", 0.2)
