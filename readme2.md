# MLOps & MLflow 实践指南 🚀

![MLOps Lifecycle](docs/images/mlops-lifecycle.png) <!-- 请替换为实际图片 -->
![MLflow Architecture](docs/images/mlflow-arch.png) <!-- 请替换为实际图片 -->
![YOLO Integration](docs/images/yolo-integration.png) <!-- 请替换为实际图片 -->

[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 目录
- [MLOps 成熟度模型](#-mlops-成熟度模型)
- [✨ MLflow 核心功能](#-mlflow-核心功能)
- [🚀 快速开始](#-快速开始)
- [🔧 功能详解](#-功能详解)
- [🤝 贡献指南](#-贡献指南)

## 🌟 MLOps 成熟度模型

| 等级 | 核心能力 | 迭代速度 | 关键组件 |
|------|----------|----------|----------|
| **0️⃣ 手动阶段** | 全手动流程 | <5次/年 | 无自动化体系 |
| **1️⃣ 自动化CT** | 持续训练 | 周级迭代 | 特征库/元数据管理 |
| **2️⃣ CI/CD自动化** | 分钟级部署 | 按需发布 | 模型注册中心/全链路监控 |

## ✨ MLflow 核心功能

### 🎯 实验追踪（Tracking）
import mlflow
with mlflow.start_run():
mlflow.log_param("learning_rate", 0.01) # 记录超参数
mlflow.log_metric("accuracy", 0.92) # 跟踪指标
mlflow.log_artifact("model.pkl") # 保存产出物

### 📦 项目打包（Projects）

MLproject
name: 房价预测
conda_env: conda.yaml
entry_points:
main:
parameters:
data_file: path
command: "python train.py {data_file}"

### 🧩 模型管理（Models）
mlflow.pyfunc.log_model(
"model",
python_model=SklearnModelWrapper(model),
conda_env=conda_env,
registered_model_name="房价预测模型"
)

## 🚀 快速开始

1. 安装依赖
pip install mlflow ultralytics
2. 启动MLflow服务
mlflow server --backend-store-uri runs/mlflow --host 0.0.0.0 --port 5000
3. 运行YOLO实验
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model.train(data='coco128.yaml', epochs=100)

## 🔧 功能详解

### 实验对比分析
- 多维度指标对比（准确率、F1值、训练时长）
- 超参数组合筛选
- 实验版本差异对比

### 模型部署流程
1. 模型验证 → 2. 打包容器 → 3. A/B测试 → 4. 生产发布

## 🤝 贡献指南
欢迎通过 Issue 提交建议或 PR 贡献代码！请遵循：
1. 保持代码风格统一
2. 添加必要的单元测试
3. 更新相关文档

[📚 查看完整文档](https://mlflow.org/docs/latest/index.html) | [🐛 提交问题](https://github.com/mlflow/mlflow/issues)
