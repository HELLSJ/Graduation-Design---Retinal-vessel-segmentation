# 视网膜血管分割 - 基于Attention U-Net的深度学习方法

## 项目简介

本项目实现了一个基于改进型Attention U-Net的视网膜血管分割系统，用于辅助糖尿病视网膜病变等眼部疾病的诊断。项目支持多个公开数据集（DRIVE、CHASE_DB1、HRF、STARE）的联合训练和评估。

## 创新点

### 1. 改进型网络架构
- **Attention机制**：在跳跃连接中引入注意力模块，让模型自动学习重要的特征区域
- **空洞卷积**：在瓶颈层使用空洞卷积扩大感受野，捕获多尺度信息
- **残差连接**：缓解深层网络的梯度消失问题

### 2. 组合损失函数
- **Dice Loss**：处理血管和背景的类别不平衡问题
- **Focal Loss**：聚焦难分类样本，提升模型对细血管的分割能力
- **Boundary Loss**：增强血管边界的分割精度

### 3. 可解释性分析
- **Grad-CAM**：可视化模型关注的区域，理解模型的决策过程
- **注意力图**：展示注意力模块的工作机制

## 项目结构

```
graduation thesis/
├── archive/                 # 数据集目录
│   ├── DRIVE/              # DRIVE数据集
│   ├── CHASE_DB1/          # CHASE_DB1数据集
│   ├── HRF/                # HRF数据集
│   └── STARE/              # STARE数据集
├── src/                     # 源代码目录
│   ├── config.py           # 配置文件
│   ├── data/               # 数据加载和预处理
│   │   ├── dataset.py
│   │   └── __init__.py
│   ├── models/             # 模型定义
│   │   ├── attention_unet.py
│   │   └── __init__.py
│   ├── losses/             # 损失函数
│   │   ├── losses.py
│   │   └── __init__.py
│   ├── metrics/            # 评估指标
│   │   ├── metrics.py
│   │   └── __init__.py
│   ├── utils/              # 工具函数
│   │   ├── visualization.py
│   │   ├── grad_cam.py
│   │   └── __init__.py
│   ├── train.py            # 训练脚本
│   └── test.py             # 测试脚本
├── checkpoints/             # 模型检查点保存目录
├── results/                 # 实验结果保存目录
├── logs/                    # 日志保存目录
├── main.py                  # 主入口文件
├── requirements.txt         # 依赖包列表
└── README.md               # 项目说明文档
```

## 环境配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 数据集准备

将数据集放置在 `archive/` 目录下，目录结构如下：

```
archive/
├── DRIVE/
│   ├── training/
│   │   ├── images/
│   │   └── mask/
│   └── test/
│       ├── images/
│       └── mask/
├── CHASE_DB1/
│   ├── Images/
│   └── Masks/
├── HRF/
│   ├── images/
│   └── mask/
└── STARE/
    └── *.ppm
```

## 使用方法

### 1. 训练模型

```bash
python main.py --mode train --batch_size 8 --epochs 100 --lr 0.0001
```

参数说明：
- `--mode`: 运行模式（train/test/grad_cam）
- `--batch_size`: 批次大小（默认8）
- `--epochs`: 训练轮数（默认100）
- `--lr`: 学习率（默认0.0001）
- `--seed`: 随机种子（默认42）
- `--data_root`: 数据集路径（默认archive）

### 2. 测试模型

```bash
python main.py --mode test --checkpoint checkpoints/checkpoint_epoch_100_best.pth
```

### 3. 生成Grad-CAM可视化

```bash
python main.py --mode grad_cam --checkpoint checkpoints/checkpoint_epoch_100_best.pth
```

## 配置说明

在 `src/config.py` 中可以修改以下配置：

```python
class Config:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DATA_ROOT = 'archive'
    IMG_SIZE = (512, 512)
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 100
    
    LOSS_WEIGHTS = {
        'dice': 0.5,
        'focal': 0.3,
        'boundary': 0.2
    }
```

## 评估指标

项目实现了以下评估指标：

- **Dice系数**：衡量分割的重叠程度
- **IoU（交并比）**：衡量预测和真实区域的交集与并集之比
- **敏感度**：正确识别血管的能力
- **特异度**：正确识别背景的能力
- **准确率**：整体分类正确率
- **精确率**：预测为血管的样本中真正是血管的比例
- **召回率**：真实血管中被正确识别的比例
- **F1分数**：精确率和召回率的调和平均
- **AUC**：ROC曲线下面积

## 实验结果

训练完成后，结果将保存在 `results/` 目录下，包括：

- 分割结果可视化图
- 训练历史曲线（Loss、Dice、IoU等）
- ROC曲线
- 精确率-召回率曲线
- 评估指标表格
- Grad-CAM可视化图

## 技术栈

- **深度学习框架**：PyTorch
- **图像处理**：OpenCV、PIL、Albumentations
- **数据处理**：NumPy
- **可视化**：Matplotlib
- **实验管理**：TensorBoard

## 论文撰写建议

基于本项目的毕业论文可以按照以下结构撰写：

1. **引言**
   - 研究背景和意义
   - 视网膜血管分割的应用价值
   - 研究目标和主要贡献

2. **相关工作**
   - 传统分割方法
   - 基于深度学习的分割方法
   - 注意力机制在医学图像中的应用

3. **方法**
   - 网络架构设计（Attention U-Net）
   - 损失函数设计（组合损失函数）
   - 训练策略（数据增强、多数据集联合训练）

4. **实验**
   - 数据集介绍
   - 实验设置
   - 评价指标
   - 实验结果与分析

5. **讨论**
   - 创新点分析
   - 与其他方法的对比
   - 局限性和未来工作

6. **结论**

## 常见问题

### 1. 显存不足怎么办？

可以减小 `batch_size` 或 `img_size`：

```python
BATCH_SIZE = 4  # 减小批次大小
IMG_SIZE = (256, 256)  # 减小图像尺寸
```

### 2. 训练速度慢怎么办？

- 使用GPU加速
- 减少 `num_workers`（如果CPU资源有限）
- 使用混合精度训练

### 3. 如何调整损失函数权重？

在 `config.py` 中修改 `LOSS_WEIGHTS`：

```python
LOSS_WEIGHTS = {
    'dice': 0.6,
    'focal': 0.2,
    'boundary': 0.2
}
```

## 联系方式

如有问题，请联系：1206817338lcx@gmail.com