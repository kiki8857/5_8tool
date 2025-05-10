#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
铣刀寿命预测系统 - LSTM超参数调优 (GPU优化版)
功能：高效利用GPU加速LSTM模型超参数调优
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import argparse
import logging
import json
from datetime import datetime
import time
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置随机种子以便结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 配置CUDA以提高性能
torch.backends.cudnn.benchmark = True

class LSTMModel(nn.Module):
    """LSTM模型"""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 全连接层
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # 初始化隐藏状态
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))
        
        # 只取序列的最后一个时间步的输出
        out = out[:, -1, :]
        
        # Dropout和全连接层
        out = self.dropout(out)
        out = self.fc(out)
        
        return out

class GPUTuner:
    """GPU高效超参数调优器"""
    
    def __init__(self, features_path, output_path, train_cutters=['c1', 'c4'], test_cutter='c6', batch_size=128):
        """初始化GPU优化的LSTM超参数调优器"""
        self.features_path = features_path
        self.output_path = output_path
        self.train_cutters = train_cutters
        self.test_cutter = test_cutter
        self.batch_size = batch_size
        
        # 确保输出目录存在
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 超参数搜索空间
        self.param_grid = {
            'hidden_size': [32, 64, 128, 256, 512],
            'num_layers': [1, 2, 3],
            'dropout': [0.1, 0.2, 0.3, 0.4],
            'learning_rate': [0.01, 0.001, 0.0001, 0.00001],
            'weight_decay': [0, 1e-6, 1e-5, 1e-4]
        }
        
        # 序列长度固定为5
        self.seq_length = 5
        
        # 训练参数
        self.epochs = 300
        self.patience = 30
    
    def prepare_data(self):
        """预处理数据并将其移至GPU"""
        logger.info("加载和预处理数据...")
        
        # 1. 加载训练数据
        train_data = []
        for cutter in self.train_cutters:
            data_file = os.path.join(self.features_path, cutter, f"{cutter}_selected_feature_data.csv")
            data = pd.read_csv(data_file)
            data = data.sort_values('cut_num')
            train_data.append(data)
        
        # 2. 合并训练数据用于特征标准化
        all_train_data = pd.concat(train_data)
        X_all = all_train_data.drop(['cut_num', 'wear_VB_avg'], axis=1).values
        y_all = all_train_data['wear_VB_avg'].values.reshape(-1, 1)
        
        # 3. 标准化
        self.X_scaler = MinMaxScaler(feature_range=(0, 1))
        self.y_scaler = MinMaxScaler(feature_range=(0, 1))
        
        self.X_scaler.fit(X_all)
        self.y_scaler.fit(y_all)
        
        # 保存输入特征数量
        self.input_size = X_all.shape[1]
        
        # 4. 准备训练序列数据
        X_train_seqs = []
        y_train_seqs = []
        
        for df in train_data:
            X = df.drop(['cut_num', 'wear_VB_avg'], axis=1).values
            y = df['wear_VB_avg'].values.reshape(-1, 1)
            
            X_scaled = self.X_scaler.transform(X)
            y_scaled = self.y_scaler.transform(y)
            
            # 创建序列
            for i in range(len(X_scaled) - self.seq_length):
                X_train_seqs.append(X_scaled[i:i+self.seq_length])
                y_train_seqs.append(y_scaled[i+self.seq_length])
        
        # 5. 转换为PyTorch张量并移至GPU
        self.X_train = torch.tensor(np.array(X_train_seqs), dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(np.array(y_train_seqs), dtype=torch.float32).to(self.device)
        
        # 6. 加载测试数据
        test_data_file = os.path.join(self.features_path, self.test_cutter, f"{self.test_cutter}_selected_feature_data.csv")
        test_data = pd.read_csv(test_data_file)
        test_data = test_data.sort_values('cut_num')
        
        X_test = test_data.drop(['cut_num', 'wear_VB_avg'], axis=1).values
        y_test = test_data['wear_VB_avg'].values.reshape(-1, 1)
        self.test_cut_nums = test_data['cut_num'].values
        
        X_test_scaled = self.X_scaler.transform(X_test)
        y_test_scaled = self.y_scaler.transform(y_test)
        
        # 7. 创建测试序列
        X_test_seqs = []
        y_test_seqs = []
        self.test_cut_seqs = []
        
        for i in range(len(X_test_scaled) - self.seq_length):
            X_test_seqs.append(X_test_scaled[i:i+self.seq_length])
            y_test_seqs.append(y_test_scaled[i+self.seq_length])
            self.test_cut_seqs.append(self.test_cut_nums[i+self.seq_length])
        
        # 8. 转换为PyTorch张量并移至GPU
        self.X_test = torch.tensor(np.array(X_test_seqs), dtype=torch.float32).to(self.device)
        self.y_test = torch.tensor(np.array(y_test_seqs), dtype=torch.float32).to(self.device)
        
        # 9. 保存原始测试值用于评估
        self.y_test_orig = self.y_scaler.inverse_transform(y_test_scaled[self.seq_length:])
        
        logger.info(f"训练数据形状: {self.X_train.shape}, {self.y_train.shape}")
        logger.info(f"测试数据形状: {self.X_test.shape}, {self.y_test.shape}")
        logger.info(f"输入特征数量: {self.input_size}")
        
        return True

    def train_model(self, hidden_size, num_layers, dropout, learning_rate, weight_decay):
        """使用指定参数训练LSTM模型"""
        # 创建模型
        model = LSTMModel(
            input_size=self.input_size,
            hidden_size=hidden_size, 
            num_layers=num_layers,
            dropout=dropout
        ).to(self.device)
        
        # 损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # 训练跟踪变量
        best_val_loss = float('inf')
        early_stop_counter = 0
        best_model_state = None
        
        # 训练历史
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        # 训练循环
        for epoch in range(self.epochs):
            # 训练模式
            model.train()
            
            # 使用大批量训练，减少CPU-GPU传输
            optimizer.zero_grad(set_to_none=True)
            outputs = model(self.X_train)
            loss = criterion(outputs, self.y_train)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss = loss.item()
            
            # 验证模式
            model.eval()
            with torch.no_grad():
                val_outputs = model(self.X_test)
                val_loss = criterion(val_outputs, self.y_test).item()
            
            # 保存训练历史
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # 提前停止逻辑
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                best_model_state = model.state_dict()
            else:
                early_stop_counter += 1
                
            if early_stop_counter >= self.patience:
                break
                
        # 加载最佳模型进行评估
        model.load_state_dict(best_model_state)
        
        # 评估模型
        model.eval()
        with torch.no_grad():
            y_pred_scaled = model(self.X_test).cpu().numpy()
        
        # 反标准化预测值
        y_pred = self.y_scaler.inverse_transform(y_pred_scaled)
        
        # 计算评估指标
        mse = mean_squared_error(self.y_test_orig, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test_orig, y_pred)
        r2 = r2_score(self.y_test_orig, y_pred)
        
        # 创建结果字典
        result = {
            'params': {
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'dropout': dropout,
                'learning_rate': learning_rate,
                'weight_decay': weight_decay
            },
            'metrics': {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2),
                'epochs_trained': epoch + 1
            },
            'model_state': best_model_state
        }
        
        return result
    
    def tune_hyperparameters(self):
        """执行超参数调优过程"""
        logger.info("开始超参数调优...")
        start_time = time.time()
        
        # 准备数据
        self.prepare_data()
        
        # 评估所有参数组合
        all_results = []
        best_result = None
        best_r2 = -float('inf')
        
        # 生成所有参数组合
        param_combinations = []
        for hidden_size in self.param_grid['hidden_size']:
            for num_layers in self.param_grid['num_layers']:
                for dropout in self.param_grid['dropout']:
                    for learning_rate in self.param_grid['learning_rate']:
                        for weight_decay in self.param_grid['weight_decay']:
                            param_combinations.append((
                                hidden_size, num_layers, dropout, learning_rate, weight_decay
                            ))
        
        # 使用tqdm显示进度
        for params in tqdm(param_combinations, desc="超参数搜索"):
            hidden_size, num_layers, dropout, learning_rate, weight_decay = params
            
            logger.info(f"评估参数: hidden_size={hidden_size}, layers={num_layers}, "
                       f"dropout={dropout}, lr={learning_rate}, decay={weight_decay}")
            
            try:
                # 训练和评估
                result = self.train_model(hidden_size, num_layers, dropout, learning_rate, weight_decay)
                all_results.append(result)
                
                # 更新最佳结果
                if result['metrics']['r2'] > best_r2:
                    best_r2 = result['metrics']['r2']
                    best_result = result
                    
                    # 保存最佳模型
                    torch.save(result['model_state'], os.path.join(self.output_path, 'best_model.pt'))
                    
                    logger.info(f"找到更好的模型! R²: {best_r2:.4f}, RMSE: {result['metrics']['rmse']:.4f}")
                
            except Exception as e:
                logger.error(f"评估参数时出错: {e}")
                continue
        
        # 计算总耗时
        total_time = time.time() - start_time
        logger.info(f"超参数调优完成，总耗时: {total_time:.2f}秒")
        
        # 保存结果
        if best_result:
            # 保存最佳参数
            with open(os.path.join(self.output_path, 'best_params.json'), 'w') as f:
                json.dump(best_result['params'], f, indent=4)
            
            # 保存所有结果
            with open(os.path.join(self.output_path, 'all_results.json'), 'w') as f:
                # 移除模型状态以便保存
                serializable_results = []
                for result in all_results:
                    result_copy = result.copy()
                    if 'model_state' in result_copy:
                        del result_copy['model_state']
                    serializable_results.append(result_copy)
                json.dump(serializable_results, f, indent=4)
        
        return best_result
    
    def run(self):
        """运行完整调优流程"""
        # 创建唯一输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_path = os.path.join(self.output_path, f"run_{timestamp}")
        os.makedirs(self.output_path, exist_ok=True)
        
        # 记录GPU信息
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA版本: {torch.version.cuda}")
            logger.info(f"PyTorch版本: {torch.__version__}")
            logger.info(f"总显存: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB")
        
        # 计算总参数组合数
        total_combinations = (
            len(self.param_grid['hidden_size']) *
            len(self.param_grid['num_layers']) *
            len(self.param_grid['dropout']) *
            len(self.param_grid['learning_rate']) *
            len(self.param_grid['weight_decay'])
        )
        logger.info(f"将评估 {total_combinations} 个参数组合")
        
        # 执行超参数调优
        best_result = self.tune_hyperparameters()
        
        if best_result:
            logger.info("调优完成！最佳参数:")
            logger.info(f"  Hidden Size: {best_result['params']['hidden_size']}")
            logger.info(f"  Layers: {best_result['params']['num_layers']}")
            logger.info(f"  Dropout: {best_result['params']['dropout']}")
            logger.info(f"  Learning Rate: {best_result['params']['learning_rate']}")
            logger.info(f"  Weight Decay: {best_result['params']['weight_decay']}")
            logger.info(f"  R²: {best_result['metrics']['r2']:.6f}")
            logger.info(f"  RMSE: {best_result['metrics']['rmse']:.6f}")
            logger.info(f"  MAE: {best_result['metrics']['mae']:.6f}")
        
        return best_result

def main():
    """主函数"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='铣刀寿命预测系统 - LSTM超参数调优 (GPU优化版)')
    
    # 添加命令行参数
    parser.add_argument('--features_path', type=str, default='./data/selected_features',
                       help='特征数据路径')
    parser.add_argument('--output_path', type=str, default='./results/lstm_tuning_gpu',
                       help='输出根路径')
    parser.add_argument('--train_cutters', type=str, default='c1,c4',
                       help='训练集刀具列表，用逗号分隔')
    parser.add_argument('--test_cutter', type=str, default='c6',
                       help='测试集刀具')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='批处理大小')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 解析训练刀具列表
    train_cutters = args.train_cutters.split(',')
    
    # 初始化GPU高效调优器
    tuner = GPUTuner(
        features_path=args.features_path,
        output_path=args.output_path,
        train_cutters=train_cutters,
        test_cutter=args.test_cutter,
        batch_size=args.batch_size
    )
    
    # 运行超参数调优
    tuner.run()

if __name__ == "__main__":
    main() 