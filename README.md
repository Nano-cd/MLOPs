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

#  用法示例：
# 工业参数拟合与预测模型

本项目实现了一个基于线性回归和最小二乘法的工业参数预测模型，集成MLflow进行实验跟踪和模型管理。

## 主要功能
- 多维度工业参数拟合分析
- 线性回归模型训练与评估
- 最小二乘法参数优化
- MLflow实验跟踪管理
- 可视化结果输出

## 依赖环境
- Python 3.8+
- 主要依赖库：
  ```bash
  pip install numpy scikit-learn matplotlib mlflow scipy
  ```

## 快速开始
1. 克隆仓库
   ```bash
   git clone https://github.com/yourusername/industrial-parameter-fitting.git
   cd industrial-parameter-fitting
   ```

2. 运行主程序
   ```bash
   python main.py
   ```

3. 查看MLflow结果
   ```bash
   mlflow ui
   ```
   在浏览器中访问 `http://localhost:5000`

## 数据配置
修改代码中的原始数据部分：

## 结果输出
- 自动保存两种可视化结果：
  -------------------------|---------------------------
  线性回归结果            | 最小二乘拟合结果

- MLflow跟踪记录：
  - 模型参数 (test_size, random_state)
  - 评估指标 (MSE, MAE)
  - 训练模型文件
  - 可视化图表



# 工业图像分类训练系统

基于YOLOv8和MLflow的智能分类训练框架，支持完整的实验跟踪和数据分析流程。

![训练过程可视化](assets/training_visualization.png)  
*MLflow跟踪界面示例*

## 核心功能

- 🖼️ 自动化图像数据集加载与预处理
- 📊 智能数据分布分析与可视化
- 🧠 YOLOv8分类模型训练支持
- 🔍 MLflow实验全流程跟踪
- 📁 自动生成YOLO格式数据集配置
- 📈 训练指标实时监控

## 环境要求

- Python 3.8+
- CUDA 11.7+ (推荐GPU环境)
- 依赖安装：pip install ultralytics mlflow pandas scikit-learn


## 快速开始

1. 数据准备（按以下结构组织）：
dataset/
class_1/
img1.jpg
img2.png
class_2/
img3.webp
...

2. 启动训练：
python main.py --data_dir ./dataset --imgsz 640 --epochs 100

3. 监控训练过程：
mlflow ui --port 5000

## 核心参数配置

通过`train_model`函数配置训练参数：
params = {
"data": yaml_path, # 自动生成的YAML配置
"epochs": 100, # 训练总轮次
"imgsz": 320, # 输入图像尺寸
"batch": 32, # 批次大小
"lr0": 0.02, # 初始学习率
"dropout": 0.2, # Dropout概率
"optimizer": "SGD", # 优化器选择
"seed": 20240722 # 随机种子
}

## 数据管理特性

- 自动生成数据集分析报告：
  - 类别分布直方图
  - 训练/测试集比例
  - 图像路径映射表
- 智能数据分割：
  ```python
  X_train, X_test = train_test_split(
      file_paths, 
      test_size=0.2,
      stratify=labels  # 保持类别分布
  )
  ```

## MLflow跟踪项

| 跟踪类型       | 记录内容                  |
|----------------|-------------------------|
| 参数           | 超参数/数据集路径/模型配置 |
| 指标           | 准确率/Loss曲线/mAP      |
| 图表           | 混淆矩阵/PR曲线          |
| 模型           | 最佳模型快照             |
| 数据集分析     | 类别分布/分割策略        |

## 项目结构

├── main.py # 主入口
├── dataset/ # 数据目录（示例）
├── pts/ # 模型配置
│ ├── yolov8s-cls.yaml # YOLO分类配置
│ └── yolov8s-cls.pt # 预训练权重
├── data_split.csv # 自动生成的数据分割
└── mlruns/ # MLflow记录目录
## 许可协议

本项目采用 [Apache 2.0 License](LICENSE)，允许商业使用但需保留版权声明。

主要特点说明：
1. 采用模块化结构展示不同功能组件
包含可视化的工作流程图和界面预览
突出MLflow的跟踪能力设计
强调工业级应用的可靠性
提供从数据准备到生产部署的完整链路说明
