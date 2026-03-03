# 快速开始指南

## 第一步：安装依赖

```bash
pip install -r requirements.txt
```

## 第二步：检查数据集

确保数据集已正确放置在 `archive/` 目录下：

```bash
ls archive/
```

应该看到：
- DRIVE/
- CHASE_DB1/
- HRF/
- STARE/

## 第三步：训练模型

```bash
python main.py --mode train --batch_size 8 --epochs 100
```

训练过程可能需要几个小时，具体取决于你的硬件配置。

## 第四步：测试模型

训练完成后，使用最佳模型进行测试：

```bash
python main.py --mode test
```

## 第五步：生成可视化结果

生成Grad-CAM可视化以理解模型的决策过程：

```bash
python main.py --mode grad_cam
```

## 查看结果

所有结果将保存在 `results/` 目录下：

- `prediction_*.png`: 分割结果可视化
- `training_history.png`: 训练历史曲线
- `roc_curve.png`: ROC曲线
- `pr_curve.png`: 精确率-召回率曲线
- `results_table.png`: 评估指标表格

## 使用TensorBoard监控训练

```bash
tensorboard --logdir checkpoints/logs
```

然后在浏览器中打开 `http://localhost:6006`

## 常用命令

```bash
# 训练模型（使用默认参数）
python main.py --mode train

# 训练模型（自定义参数）
python main.py --mode train --batch_size 4 --epochs 50 --lr 0.0001

# 测试模型（使用最佳检查点）
python main.py --mode test

# 测试模型（指定检查点）
python main.py --mode test --checkpoint checkpoints/checkpoint_epoch_50_best.pth

# 生成Grad-CAM可视化
python main.py --mode grad_cam
```

## 故障排除

### 问题1：找不到数据集

**错误信息**：`FileNotFoundError: [Errno 2] No such file or directory: 'archive'`

**解决方案**：确保数据集已正确放置在项目根目录下的 `archive/` 文件夹中。

### 问题2：CUDA out of memory

**错误信息**：`RuntimeError: CUDA out of memory`

**解决方案**：减小批次大小或图像尺寸：

```bash
python main.py --mode train --batch_size 4
```

或在 `src/config.py` 中修改：
```python
BATCH_SIZE = 4
IMG_SIZE = (256, 256)
```

### 问题3：导入错误

**错误信息**：`ModuleNotFoundError: No module named 'src'`

**解决方案**：确保在项目根目录下运行命令，或使用 `python -m` 方式运行：

```bash
cd "d:\graduation thesis"
python main.py --mode train
```

## 下一步

1. **调整超参数**：根据验证集结果调整学习率、批次大小等参数
2. **尝试不同模型**：在 `src/models/` 中尝试不同的网络架构
3. **添加新的损失函数**：在 `src/losses/` 中实现自定义损失函数
4. **分析结果**：使用生成的可视化结果分析模型性能
5. **撰写论文**：基于实验结果撰写毕业论文

## 需要帮助？

如果遇到问题，请检查：
1. 所有依赖包是否正确安装
2. 数据集路径是否正确
3. 配置文件中的参数是否合理
4. GPU驱动和CUDA版本是否兼容

祝你顺利完成毕业设计！