#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
铣刀寿命预测系统 - CNN模型超参数调优
功能：使用Optuna库对CNN模型进行超参数调优
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import matplotlib.pyplot as plt
import json
import logging
from datetime import datetime
import argparse
import joblib  # 添加joblib导入用于保存study对象

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"使用设备: {device}")

# 设置随机种子以确保可重复性
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 定义自注意力机制
class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.scale = torch.sqrt(torch.FloatTensor([input_dim])).to(device)
        
    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.size()
        
        q = self.query(x)  
        k = self.key(x)   
        v = self.value(x)  
        
        attn_scores = torch.matmul(q, k.transpose(1, 2)) / self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        return output, attn_weights

# 定义CNN模型，参数可调
class CNNModel(nn.Module):
    def __init__(self, params, input_channels, seq_length, num_features):
        super(CNNModel, self).__init__()
        
        # 从params中提取超参数
        conv_layers = params['conv_layers']
        first_conv_out = params['first_conv_out']
        kernel_size = params['kernel_size']
        dropout_rate1 = params['dropout_rate1']
        dropout_rate2 = params['dropout_rate2']
        fc_units1 = params['fc_units1']
        fc_units2 = params['fc_units2']
        use_attention = params['use_attention']
        attention_weight = params['attention_weight']
        
        # 计算每层的输出通道数
        channels = [first_conv_out]
        for i in range(1, conv_layers):
            if i <= 2:
                channels.append(channels[-1] * 2)  # 前两层翻倍
            else:
                channels.append(channels[-1] // 2)  # 后面的层减半
        
        # 构建卷积层
        self.conv_blocks = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        
        # 第一个卷积层
        self.conv_blocks.append(nn.Sequential(
            nn.Conv1d(input_channels, channels[0], kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(channels[0]),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(kernel_size=2, stride=2)
        ))
        self.shortcuts.append(nn.Conv1d(input_channels, channels[0], kernel_size=1))
        
        # 剩余卷积层
        for i in range(1, conv_layers):
            in_channels = channels[i-1]
            out_channels = channels[i]
            
            if i < conv_layers - 1:
                self.conv_blocks.append(nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
                    nn.BatchNorm1d(out_channels),
                    nn.LeakyReLU(0.1),
                    nn.MaxPool1d(kernel_size=2, stride=2)
                ))
            else:
                # 最后一层不使用池化
                self.conv_blocks.append(nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
                    nn.BatchNorm1d(out_channels),
                    nn.LeakyReLU(0.1)
                ))
            
            self.shortcuts.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
        
        # 计算展平后的特征维度
        pool_count = conv_layers - 1
        flattened_size = channels[-1] * ((seq_length * num_features) // (2**pool_count))
        
        # 注意力机制
        self.use_attention = use_attention
        self.attention_weight = attention_weight
        if use_attention:
            self.attention = SelfAttention(channels[-1])
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(flattened_size, fc_units1),
            nn.BatchNorm1d(fc_units1),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate1),
            
            nn.Linear(fc_units1, fc_units2),
            nn.BatchNorm1d(fc_units2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate2),
            
            nn.Linear(fc_units2, 1)  # 输出层
        )
    
    def forward(self, x):
        batch_size, channels, seq_length, num_features = x.shape
        x = x.view(batch_size, channels, seq_length * num_features)
        
        # 应用卷积层和残差连接
        for i, (conv, shortcut) in enumerate(zip(self.conv_blocks, self.shortcuts)):
            identity = x
            x = conv(x)
            
            # 对shortcut应用相同的池化
            if i < len(self.conv_blocks) - 1:  # 除了最后一层
                shortcut_x = shortcut(identity)
                shortcut_x = nn.functional.max_pool1d(shortcut_x, kernel_size=2, stride=2)
            else:
                shortcut_x = shortcut(identity)
            
            x = x + shortcut_x
        
        # 应用注意力机制
        if self.use_attention:
            batch_size, channels, feature_len = x.shape
            x_reshaped = x.permute(0, 2, 1)  # [batch_size, feature_len, channels]
            x_att, _ = self.attention(x_reshaped)
            x_att = x_att.permute(0, 2, 1)  # [batch_size, channels, feature_len]
            x = x + self.attention_weight * x_att
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.fc(x)
        
        return x

class DataProcessor:
    def __init__(self, data_path, target_column, train_cutters, test_cutter):
        self.data_path = data_path
        self.target_column = target_column
        self.train_cutters = train_cutters
        self.test_cutter = test_cutter
        self.feature_names = None
        self.scaler = None
    
    def load_data(self):
        logger.info("加载数据...")
        
        # 加载训练数据
        train_data = []
        for cutter in self.train_cutters:
            data_file = os.path.join(self.data_path, cutter, f"{cutter}_selected_feature_data.csv")
            try:
                cutter_data = pd.read_csv(data_file)
                train_data.append(cutter_data)
                logger.info(f"成功加载{cutter}数据: {data_file}")
            except Exception as e:
                logger.error(f"加载{cutter}数据失败: {e}")
        
        if not train_data:
            raise ValueError("未能加载任何训练数据")
        
        # 合并训练数据
        train_df = pd.concat(train_data, ignore_index=True)
        
        # 加载测试数据
        test_data_file = os.path.join(self.data_path, self.test_cutter, 
                                      f"{self.test_cutter}_selected_feature_data.csv")
        try:
            test_df = pd.read_csv(test_data_file)
            logger.info(f"成功加载{self.test_cutter}数据: {test_data_file}")
        except Exception as e:
            logger.error(f"加载{self.test_cutter}数据失败: {e}")
            raise ValueError(f"未能加载测试数据: {e}")
        
        # 提取特征和目标变量
        feature_cols = [col for col in train_df.columns if col != self.target_column and col != 'cut_num']
        X_train_raw = train_df[feature_cols].values
        y_train_raw = train_df[self.target_column].values
        
        X_test_raw = test_df[feature_cols].values
        y_test_raw = test_df[self.target_column].values
        test_cut_nums = test_df['cut_num'].values
        
        # 保存特征名称
        self.feature_names = feature_cols
        
        logger.info(f"数据加载完成，输入特征数量: {X_train_raw.shape[1]}")
        logger.info(f"训练数据形状: {X_train_raw.shape}, 测试数据形状: {X_test_raw.shape}")
        
        return X_train_raw, y_train_raw, X_test_raw, y_test_raw, test_cut_nums, len(feature_cols)
    
    def create_sequences(self, X, y, window_size, stride=1):
        sequences_X = []
        sequences_y = []
        
        for i in range(0, len(X) - window_size + 1, stride):
            sequences_X.append(X[i:i+window_size].reshape(1, window_size, -1))
            sequences_y.append(y[i+window_size-1])
        
        return np.array(sequences_X), np.array(sequences_y)

def objective(trial, data_processor, output_dir):
    """Optuna目标函数，用于超参数优化"""
    # 定义搜索空间
    params = {
        # 数据和训练参数
        'window_size': trial.suggest_int('window_size', 10, 25),
        'stride': trial.suggest_int('stride', 1, 3),
        'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32]),
        
        # 优化器参数
        'optimizer': trial.suggest_categorical('optimizer', ['Adam', 'AdamW']),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        
        # 学习率调度器
        'scheduler': trial.suggest_categorical('scheduler', ['OneCycleLR', 'CosineAnnealingLR']),
        'max_lr_factor': trial.suggest_float('max_lr_factor', 2.0, 10.0),
        
        # 网络结构参数
        'conv_layers': trial.suggest_int('conv_layers', 3, 5),
        'first_conv_out': trial.suggest_categorical('first_conv_out', [32, 64, 128]),
        'kernel_size': trial.suggest_categorical('kernel_size', [3, 5, 7]),
        'dropout_rate1': trial.suggest_float('dropout_rate1', 0.1, 0.5),
        'dropout_rate2': trial.suggest_float('dropout_rate2', 0.1, 0.4),
        'fc_units1': trial.suggest_categorical('fc_units1', [128, 256, 512]),
        'fc_units2': trial.suggest_categorical('fc_units2', [64, 128, 256]),
        
        # 注意力机制参数
        'use_attention': trial.suggest_categorical('use_attention', [True, False]),
        'attention_weight': trial.suggest_float('attention_weight', 0.1, 1.0)
    }
    
    # 处理数据
    X_train_raw, y_train_raw, X_test_raw, y_test_raw, _, num_features = data_processor.load_data()
    
    # 标准化数据
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    # 创建时序序列
    window_size = params['window_size']
    stride = params['stride']
    X_train_seq, y_train_seq = data_processor.create_sequences(
        X_train_scaled, y_train_raw, window_size, stride
    )
    X_test_seq, y_test_seq = data_processor.create_sequences(
        X_test_scaled, y_test_raw, window_size, stride
    )
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train_seq).to(device)
    y_train_tensor = torch.FloatTensor(y_train_seq).reshape(-1, 1).to(device)
    
    X_test_tensor = torch.FloatTensor(X_test_seq).to(device)
    y_test_tensor = torch.FloatTensor(y_test_seq).reshape(-1, 1).to(device)
    
    # 划分训练集和验证集
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    
    # 使用80%数据作为训练集，20%作为验证集
    train_size = int(0.8 * len(train_dataset))
    valid_size = len(train_dataset) - train_size
    
    train_subset, valid_subset = random_split(
        train_dataset, [train_size, valid_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    
    # 创建数据加载器
    batch_size = params['batch_size']
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False)
    
    # 构建模型
    model = CNNModel(params, input_channels=1, seq_length=window_size, num_features=num_features).to(device)
    
    # 设置损失函数
    criterion = nn.MSELoss()
    
    # 设置优化器
    if params['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    else:  # AdamW
        optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    
    # 设置学习率调度器
    if params['scheduler'] == 'OneCycleLR':
        steps_per_epoch = len(train_loader)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=params['learning_rate'] * params['max_lr_factor'], 
            epochs=100,  # 固定最大轮次为100
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3
        )
    else:  # CosineAnnealingLR
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, eta_min=params['learning_rate'] / 100
        )
    
    # 训练模型
    best_valid_loss = float('inf')
    patience = 20  # 提前停止耐心值
    patience_counter = 0
    
    for epoch in range(100):  # 最多训练100轮
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            # 前向传播
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()
            
            if params['scheduler'] == 'OneCycleLR':
                scheduler.step()  # OneCycleLR每个批次更新一次
            
            train_loss += loss.item() * X_batch.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # 验证阶段
        model.eval()
        valid_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                valid_loss += loss.item() * X_batch.size(0)
        
        valid_loss /= len(valid_loader.dataset)
        
        # CosineAnnealingLR每个轮次更新一次
        if params['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        
        # 打印训练进度
        if (epoch + 1) % 10 == 0:
            logger.info(f'Trial {trial.number}, Epoch {epoch+1}/100, '
                       f'Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')
        
        # 提前停止检查
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f'Trial {trial.number} 提前停止训练，已经{patience}个轮次没有改善')
                break
    
    # 计算最终验证损失
    model.eval()
    final_valid_loss = 0.0
    
    with torch.no_grad():
        for X_batch, y_batch in valid_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            final_valid_loss += loss.item() * X_batch.size(0)
    
    final_valid_loss /= len(valid_loader.dataset)
    
    # 计算测试集上的性能
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)
    
    y_test = y_test_tensor.cpu().numpy().flatten()
    y_pred = y_pred_tensor.cpu().numpy().flatten()
    
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # 记录结果
    trial.set_user_attr('MSE', float(mse))
    trial.set_user_attr('RMSE', float(rmse))
    trial.set_user_attr('MAE', float(mae))
    trial.set_user_attr('R2', float(r2))
    
    logger.info(f'Trial {trial.number} 完成: '
               f'验证损失: {final_valid_loss:.4f}, '
               f'测试MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}')
    
    # 如果是迄今为止最好的试验，保存模型
    if trial.should_prune():
        raise optuna.TrialPruned()
    
    # 保存每个试验的模型和参数
    logger.info(f"保存Trial {trial.number}的结果")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存模型
    torch.save(model.state_dict(), os.path.join(output_dir, f'model_trial_{trial.number}.pt'))
    
    # 保存参数
    with open(os.path.join(output_dir, f'params_trial_{trial.number}.json'), 'w') as f:
        json.dump(params, f, indent=4)
    
    # 保存指标
    metrics = {
        'MSE': float(mse),
        'RMSE': float(rmse),
        'MAE': float(mae),
        'R2': float(r2),
        'valid_loss': float(final_valid_loss)
    }
    with open(os.path.join(output_dir, f'metrics_trial_{trial.number}.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return final_valid_loss

def main():
    """主函数"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='CNN模型超参数调优')
    
    # 添加命令行参数
    parser.add_argument('--features_path', type=str, default='/Users/xiaohudemac/cursor01/bishe/5_8tool/data/selected_features',
                      help='特征选择后的数据路径')
    parser.add_argument('--output_dir', type=str, default='/Users/xiaohudemac/cursor01/bishe/5_8tool/results/cnn_tuning',
                      help='输出结果保存路径')
    parser.add_argument('--target_column', type=str, default='wear_VB_avg',
                      help='目标变量列名')
    parser.add_argument('--train_cutters', type=str, default='c1,c4',
                      help='用于训练的刀具列表，用逗号分隔')
    parser.add_argument('--test_cutter', type=str, default='c6',
                      help='用于测试的刀具')
    parser.add_argument('--n_trials', type=int, default=100,
                      help='Optuna试验次数')
    parser.add_argument('--timeout', type=int, default=7200,
                      help='Optuna超时时间(秒)')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 解析训练刀具列表
    train_cutters = args.train_cutters.split(',')
    
    # 为输出路径添加时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化数据处理器
    data_processor = DataProcessor(
        data_path=args.features_path,
        target_column=args.target_column,
        train_cutters=train_cutters,
        test_cutter=args.test_cutter
    )
    
    # 使用Optuna进行超参数调优
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner()
    )
    
    # 打印配置信息
    logger.info(f"开始CNN模型超参数调优")
    logger.info(f"训练刀具: {', '.join(train_cutters)}, 测试刀具: {args.test_cutter}")
    logger.info(f"计划试验次数: {args.n_trials}, 超时时间: {args.timeout}秒")
    logger.info(f"结果将保存至: {output_dir}")
    
    # 开始优化
    try:
        study.optimize(
            lambda trial: objective(trial, data_processor, output_dir),
            n_trials=args.n_trials,
            timeout=args.timeout,
            n_jobs=1,  # PyTorch模型并行训练可能会有问题，设为1更安全
            show_progress_bar=True
        )
        
        # 打印最佳结果
        best_trial = study.best_trial
        logger.info(f"优化完成! 最佳试验 #{best_trial.number}")
        logger.info(f"验证损失: {best_trial.value:.4f}")
        logger.info(f"测试MSE: {best_trial.user_attrs['MSE']:.4f}")
        logger.info(f"测试RMSE: {best_trial.user_attrs['RMSE']:.4f}")
        logger.info(f"测试MAE: {best_trial.user_attrs['MAE']:.4f}")
        logger.info(f"测试R²: {best_trial.user_attrs['R2']:.4f}")
        logger.info(f"最佳超参数:")
        for key, value in best_trial.params.items():
            logger.info(f"  {key}: {value}")
        
        # 保存study对象
        study_file = os.path.join(output_dir, 'study.pkl')
        joblib.dump(study, study_file)
        logger.info(f"Study对象已保存至: {study_file}")
        
        # 保存最佳参数
        best_params_file = os.path.join(output_dir, 'best_params.json')
        with open(best_params_file, 'w') as f:
            json.dump(best_trial.params, f, indent=4)
        logger.info(f"最佳参数已保存至: {best_params_file}")
        
        # 保存最佳指标
        best_metrics = {
            'MSE': best_trial.user_attrs['MSE'],
            'RMSE': best_trial.user_attrs['RMSE'],
            'MAE': best_trial.user_attrs['MAE'],
            'R2': best_trial.user_attrs['R2'],
            'valid_loss': best_trial.value
        }
        best_metrics_file = os.path.join(output_dir, 'best_metrics.json')
        with open(best_metrics_file, 'w') as f:
            json.dump(best_metrics, f, indent=4)
        logger.info(f"最佳指标已保存至: {best_metrics_file}")
        
        # 可视化参数重要性和优化历史
        try:
            fig1 = plot_optimization_history(study)
            fig1.write_image(os.path.join(output_dir, 'optimization_history.png'))
            
            fig2 = plot_param_importances(study)
            fig2.write_image(os.path.join(output_dir, 'param_importances.png'))
            
            logger.info(f"可视化结果已保存")
        except Exception as e:
            logger.warning(f"生成可视化失败: {e}")
        
    except KeyboardInterrupt:
        logger.info("优化被用户中断")
    except Exception as e:
        logger.error(f"优化过程中发生错误: {e}")
        raise e

if __name__ == "__main__":
    main() 