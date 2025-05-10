# 铣刀寿命预测模型总结

本文档总结了在铣刀寿命预测项目中表现最优的随机森林（Random Forest, RF）和BP神经网络（BPNN）模型的关键信息，包括训练特征、参数设置、模型保存位置以及最终的性能评估结果。

## 1. 通用设置

- **数据来源**：所有模型均使用来自PHM2010挑战赛的铣削数据集。
- **特征工程**：采用基于相关系数的特征选择方法，从原始数据中筛选出24个与刀具磨损最相关的特征用于模型训练。这些特征主要包括原始信号（力、振动、声发射）的时域、频域及小波变换特征。
    - 特征数据保存路径：`/Users/xiaohudemac/cursor01/bishe/5_8tool/data/selected_features/`
- **数据划分**：
    - **训练集**：刀具 `c1` 和 `c4` 的数据。
    - **测试集**：刀具 `c6` 的数据。
- **目标变量**：平均后刀面磨损值 `wear_VB_avg`。

## 2. 最优随机森林 (RF) 模型

随机森林模型通过超参数调优获得了较好的预测性能。

- **训练特征**：使用上述筛选的24个特征。
- **超参数设置** (通过Grid Search获得):
    - `n_estimators`: 200
    - `max_depth`: 20
    - `min_samples_split`: 10
    - `min_samples_leaf`: 1
    - `max_features`: 'sqrt'
    - `bootstrap`: True
- **模型保存位置**：
    - 脚本：`src/random_forest_model.py` (应用了最佳参数并进行训练和评估)
    - 模型文件：`/Users/xiaohudemac/cursor01/bishe/5_8tool/results/random_forest/run_20250510_043000_selected/random_forest_model.joblib`
    - 评估结果：`/Users/xiaohudemac/cursor01/bishe/5_8tool/results/random_forest/run_20250510_043000_selected/evaluation.json`
- **训练结果 (测试集 c6)**:
    - **R² (决定系数)**: 0.8010
    - **RMSE (均方根误差)**: 17.8817
    - **MAE (平均绝对误差)**: 14.7299
- **重要特征分析**：
    根据先前的分析，最重要的特征主要来自振动信号的小波变换特征，特别是X方向的振动。前五重要特征包括：
    1.  `Vibration_X_wavelet_scale_16_ratio`
    2.  `Vibration_Y_wavelet_scale_16_ratio`
    3.  `Vibration_X_wavelet_entropy`
    4.  `Force_X_wavelet_scale_16_ratio`
    5.  `Vibration_X_wavelet_scale_32_ratio`

## 3. 最优BP神经网络 (BPNN) 模型

BP神经网络模型在经过超参数调优后，展现出本项目中最佳的预测性能。

- **训练特征**：与随机森林模型一致，使用筛选的24个特征。
    - 输入特征数量：24
- **超参数设置** (通过Grid Search获得，详情见 `/Users/xiaohudemac/cursor01/bishe/5_8tool/results/bpnn_tuning/run_20250510_044514/best_params.json`):
    - `hidden_sizes`: [64, 32] (两层隐藏层，分别有64和32个神经元)
    - `learning_rate`: 0.0001
    - `batch_size`: 64
    - `dropout_rate`: 0.1
    - `activation`: 'relu'
    - `optimizer`: 'adam'
- **模型架构与训练细节** (详情见 `/Users/xiaohudemac/cursor01/bishe/5_8tool/results/bpnn_final/run_20250510_052358/model_config.json`):
    - 网络结构包含输入层、两个隐藏层（ReLU激活，后接BatchNorm和Dropout）和输出层。
    - 训练周期 (epochs): 500 (实际训练中启用了早停机制，patience=50)
    - 损失函数: MSELoss
    - 学习率调度器: ReduceLROnPlateau (patience=20, factor=0.5)
- **模型保存位置**：
    - 调优脚本：`src/bpnn_hyperparameter_tuning.py`
    - 最终模型训练脚本：`src/bpnn_final_model.py`
    - 模型文件：`/Users/xiaohudemac/cursor01/bishe/5_8tool/results/bpnn_final/run_20250510_052358/best_model.pt`
    - 评估结果：`/Users/xiaohudemac/cursor01/bishe/5_8tool/results/bpnn_final/run_20250510_052358/evaluation.json`
- **训练结果 (测试集 c6)**:
    - **R² (决定系数)**: 0.9580
    - **RMSE (均方根误差)**: 5.5984
    - **MAE (平均绝对误差)**: 4.4165

## 4. LSTM神经网络模型

LSTM模型利用时间序列特性，能够捕捉刀具磨损数据中的时序依赖关系。

- **训练特征**：与其他模型一致，使用筛选的24个特征。
    - 输入特征数量：24
- **序列设置**：
    - 序列长度：5（使用前5个切削点的数据预测下一点）
    - 使用MinMaxScaler而非StandardScaler进行特征归一化
- **模型架构**：
    - 双层LSTM（hidden_size=64, num_layers=2）
    - Dropout率：0.3
    - 全连接输出层
- **训练参数**：
    - 学习率：0.001
    - 批处理大小：未明确指定（默认使用整个训练集）
    - 优化器：Adam（weight_decay=1e-5）
    - 训练周期 (epochs)：1000（实际训练中启用了早停机制，patience=100）
    - 学习率调度器：ReduceLROnPlateau（patience=50, factor=0.5）
- **模型保存位置**：
    - 脚本：`src/lstm_model.py`
    - 模型文件：`/Users/xiaohudemac/cursor01/bishe/5_8tool/results/lstm/best_model.pt`
    - 评估结果：`/Users/xiaohudemac/cursor01/bishe/5_8tool/results/lstm/evaluation.json`
- **训练结果 (测试集 c6)**:
    - **R² (决定系数)**: 0.9592
    - **RMSE (均方根误差)**: 7.8483
    - **MAE (平均绝对误差)**: 6.2824

## 5. 结论

通过特征选择和参数优化，我们对比了三种模型在铣刀磨损预测任务上的性能：

| 模型           | R²     | RMSE    | MAE     |
|---------------|--------|---------|---------|
| BPNN (最优超参数) | 0.9580 | 5.5984  | 4.4165  |
| LSTM          | 0.9592 | 7.8483  | 6.2824  |
| 随机森林        | 0.8010 | 17.8817 | 14.7299 |

* BPNN和LSTM模型在R²指标上表现相当（分别为0.9580和0.9592），但BPNN在误差指标上更优（RMSE和MAE分别低约29%和30%）。
* 深度学习模型（BPNN和LSTM）均显著优于随机森林模型。
* LSTM模型虽能捕捉时序特性，但在当前任务中并未带来明显优势，这可能是因为铣刀磨损更依赖于当前切削状态而非历史状态。
* BPNN模型结构更简单，计算效率更高，同时具有最低的误差，是当前最佳选择。

优化后的BPNN模型可作为后续部署和应用的主要候选模型，用于指导实际生产中的刀具更换决策，从而提高加工效率和降低成本。 