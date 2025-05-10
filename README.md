# 铣刀寿命预测系统

本项目基于PHM 2010数据集实现了铣刀寿命预测系统，采用多种机器学习和深度学习模型来预测铣刀的磨损程度。

## 项目概述

本系统使用铣削过程中采集的传感器数据（力传感器和振动传感器），提取特征，通过特征选择/降维，构建预测模型，实现对铣刀磨损量的准确预测。

## 主要功能

1. 数据预处理与特征提取
2. 特征选择与降维
3. 多模型铣刀寿命预测:
   - 随机森林 (RF)
   - BP神经网络 (BPNN)
   - 长短期记忆网络 (LSTM)
4. 模型性能评估与比较
5. 可视化分析

## 模型性能

| 模型 | R² | RMSE | MAE |
|------|-----|------|-----|
| BPNN | 0.9580 | 5.5984 | 4.4165 |
| LSTM | 0.9592 | 7.8483 | 6.2824 |
| 随机森林 | 0.8010 | 17.8817 | 14.7299 |

## 技术栈

- Python 3.x
- PyTorch (深度学习框架)
- Scikit-learn (机器学习框架)
- NumPy, Pandas (数据处理)
- Matplotlib, Seaborn (数据可视化)

## 代码结构

```
5_8tool/
├── data/               # 数据目录
│   ├── raw/            # 原始数据 (不包含在仓库中)
│   ├── processed/      # 处理后的数据 (不包含在仓库中)
│   └── selected_features/  # 特征选择后的数据
├── src/                # 源代码
│   ├── data_processing.py     # 数据预处理
│   ├── feature_extraction.py  # 特征提取
│   ├── feature_selection.py   # 特征选择与降维
│   ├── random_forest_model.py # 随机森林模型
│   ├── bpnn_final_model.py    # BP神经网络模型
│   ├── lstm_model.py          # LSTM模型
│   └── ...
├── results/            # 结果保存目录
│   ├── random_forest/  # 随机森林模型结果
│   ├── bpnn_final/     # BP神经网络模型结果
│   └── lstm/           # LSTM模型结果
└── README.md           # 项目说明文档
```

## 使用方法

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行模型

1. 随机森林模型:
```bash
python src/random_forest_model.py --features_path=data/selected_features --output_path=results/random_forest
```

2. BP神经网络模型:
```bash
python src/bpnn_final_model.py --features_path=data/selected_features --output_path=results/bpnn_final
```

3. LSTM模型:
```bash
python src/lstm_model.py --features_path=data/selected_features --output_path=results/lstm
```

## 结果示例

各模型生成结果包括:
- 模型权重文件
- 评估指标(R², RMSE, MAE等)
- 预测可视化图表
- 特征重要性分析(随机森林)
- 模型比较结果 
