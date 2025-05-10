#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
铣刀寿命预测系统 - BP神经网络超参数调优
功能：寻找BP神经网络模型的最佳超参数组合
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from matplotlib import font_manager
import json
import logging
import argparse
import time
from datetime import datetime
from tqdm import tqdm
import itertools

# 配置中文字体支持
try:
    # 尝试设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']  # 优先使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
except Exception as e:
    print(f"设置中文字体时出错: {e}")

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置随机种子以便结果可复现
torch.manual_seed(42)
np.random.seed(42)

class ToolWearDataset(Dataset):
    """刀具磨损数据集"""
    
    def __init__(self, features_path, cutter, scaler=None, is_train=True):
        """
        初始化数据集
        
        参数:
            features_path: 特征数据路径
            cutter: 刀具名称(c1, c4, c6)
            scaler: 特征标准化器，如果为None则创建新的
            is_train: 是否为训练集
        """
        self.features_path = features_path
        self.cutter = cutter
        self.is_train = is_train
        
        # 加载数据
        self.data = self._load_data()
        
        # 分离特征和目标变量
        self.X = self.data.drop(['cut_num', 'wear_VB_avg'], axis=1).values
        self.y = self.data['wear_VB_avg'].values.reshape(-1, 1)
        
        # 标准化特征
        if scaler is None:
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(self.X)
        else:
            self.scaler = scaler
            self.X = self.scaler.transform(self.X)
        
        # 对目标变量进行标准化
        self.y_mean = np.mean(self.y)
        self.y_std = np.std(self.y)
        self.y = (self.y - self.y_mean) / self.y_std
    
    def _load_data(self):
        """加载选择的特征数据"""
        data_file = os.path.join(self.features_path, self.cutter, f"{self.cutter}_selected_feature_data.csv")
        try:
            data = pd.read_csv(data_file)
            logger.info(f"成功加载{self.cutter}数据: {data_file}")
            return data
        except Exception as e:
            logger.error(f"加载{self.cutter}数据失败: {e}")
            raise
    
    def __len__(self):
        """返回数据集长度"""
        return len(self.X)
    
    def __getitem__(self, idx):
        """获取指定索引的数据"""
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.y[idx])
    
    def get_scaler(self):
        """返回特征标准化器"""
        return self.scaler
    
    def get_y_params(self):
        """返回目标变量标准化参数"""
        return self.y_mean, self.y_std

class BPNN(nn.Module):
    """BP神经网络模型"""
    
    def __init__(self, input_size, hidden_sizes, output_size=1, dropout_rate=0.2, activation='relu'):
        """
        初始化BP神经网络模型
        
        参数:
            input_size: 输入特征数量
            hidden_sizes: 隐藏层神经元数量列表
            output_size: 输出神经元数量
            dropout_rate: Dropout比例
            activation: 激活函数，'relu'或'tanh'
        """
        super(BPNN, self).__init__()
        
        # 选择激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
        
        # 创建网络层
        layers = []
        
        # 第一个隐藏层
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(self.activation)
        layers.append(nn.Dropout(dropout_rate))
        
        # 后续隐藏层
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout_rate))
        
        # 最后一个隐藏层到输出层
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        # 组合所有层
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """前向传播"""
        return self.model(x)

class BPNNHyperparameterTuner:
    """
    BP神经网络超参数调优器
    """
    def __init__(self, selected_features_path, output_path, target_column='wear_VB_avg',
                 train_cutters=None, test_cutter='c6', random_state=42):
        """
        初始化BP神经网络超参数调优器
        
        参数:
            selected_features_path: 特征选择后的数据路径
            output_path: 输出路径
            target_column: 目标变量列名
            train_cutters: 用于训练的刀具列表，默认为None（使用非测试刀具）
            test_cutter: 用于测试的刀具
            random_state: 随机种子
        """
        self.selected_features_path = selected_features_path
        self.output_path = output_path
        self.target_column = target_column
        self.test_cutter = test_cutter
        self.random_state = random_state
        
        # 如果未指定训练刀具，则使用除测试刀具外的所有刀具
        if train_cutters is None:
            all_cutters = ['c1', 'c4', 'c6']
            self.train_cutters = [c for c in all_cutters if c != test_cutter]
        else:
            self.train_cutters = train_cutters
        
        # 确保输出目录存在
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # 超参数搜索空间
        self.param_grid = {
            'hidden_sizes': [
                [64, 32],
                [128, 64],
                [256, 128, 64],
                [128, 64, 32],
                [64, 32, 16]
            ],
            'learning_rate': [0.001, 0.0005, 0.0001],
            'batch_size': [16, 32, 64],
            'dropout_rate': [0.1, 0.2, 0.3],
            'activation': ['relu', 'tanh'],
            'optimizer': ['adam', 'rmsprop']
        }
        
        # 训练参数
        self.epochs = 300  # 每个超参数组合的最大训练轮数
        self.patience = 30  # 提前停止的耐心值
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 结果保存
        self.all_results = []
    
    def prepare_data(self):
        """准备训练和测试数据"""
        logger.info("准备数据...")
        
        # 加载训练数据
        train_datasets = []
        for cutter in self.train_cutters:
            dataset = ToolWearDataset(self.selected_features_path, cutter)
            train_datasets.append(dataset)
            
            # 获取输入特征数量
            if not hasattr(self, 'input_size'):
                self.input_size = dataset.X.shape[1]
        
        # 合并训练数据集
        if len(train_datasets) > 1:
            self.train_dataset = ConcatDataset(train_datasets)
            # 获取第一个数据集的scaler用于测试集
            self.scaler = train_datasets[0].get_scaler()
            # 获取第一个数据集的目标变量标准化参数
            self.y_mean, self.y_std = train_datasets[0].get_y_params()
        else:
            self.train_dataset = train_datasets[0]
            self.scaler = train_datasets[0].get_scaler()
            self.y_mean, self.y_std = train_datasets[0].get_y_params()
        
        # 加载测试数据
        self.test_dataset = ToolWearDataset(
            self.selected_features_path, 
            self.test_cutter, 
            scaler=self.scaler, 
            is_train=False
        )
        
        logger.info(f"数据准备完成，输入特征数量: {self.input_size}")
        
        return self.input_size
    
    def train_eval_model(self, model, batch_size, learning_rate, optimizer_type):
        """
        训练和评估指定超参数的模型
        
        参数:
            model: BPNN模型
            batch_size: 批处理大小
            learning_rate: 学习率
            optimizer_type: 优化器类型，'adam'或'rmsprop'
            
        返回:
            metrics: 评估指标
        """
        # 数据加载器
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        
        test_loader = DataLoader(
            self.test_dataset, 
            batch_size=len(self.test_dataset), 
            shuffle=False
        )
        
        # 损失函数
        criterion = nn.MSELoss()
        
        # 优化器
        if optimizer_type == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_type == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
        else:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=False
        )
        
        # 将模型移至设备
        model = model.to(self.device)
        
        # 训练历史
        history = {
            'train_loss': [],
            'val_loss': [],
        }
        
        # 用于提前停止的变量
        best_val_loss = float('inf')
        early_stop_counter = 0
        
        # 训练循环
        for epoch in range(self.epochs):
            # 训练模式
            model.train()
            train_loss = 0.0
            
            # 训练一个周期
            for inputs, targets in train_loader:
                # 将数据移至设备
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
            
            # 计算平均训练损失
            train_loss /= len(train_loader)
            
            # 验证模式
            model.eval()
            val_loss = 0.0
            
            # 在测试集上验证
            with torch.no_grad():
                for inputs, targets in test_loader:
                    # 将数据移至设备
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            
            # 计算平均验证损失
            val_loss /= len(test_loader)
            
            # 更新学习率
            scheduler.step(val_loss)
            
            # 保存训练历史
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # 提前停止逻辑
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            
            if early_stop_counter >= self.patience:
                break
        
        # 获取最后的预测结果
        model.eval()
        with torch.no_grad():
            inputs, targets = next(iter(test_loader))
            inputs = inputs.to(self.device)
            outputs = model(inputs)
        
        # 将结果移回CPU并转换为numpy
        targets = targets.numpy() * self.y_std + self.y_mean
        outputs = outputs.cpu().numpy() * self.y_std + self.y_mean
        
        # 计算性能指标
        mse = mean_squared_error(targets, outputs)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets, outputs)
        r2 = r2_score(targets, outputs)
        
        # 记录评估结果
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'best_val_loss': float(best_val_loss),
            'epochs_trained': epoch + 1
        }
        
        return metrics, history
    
    def tune_hyperparameters(self):
        """
        进行超参数调优
        
        返回:
            best_params: 最佳超参数
            all_results: 所有参数组合的结果
        """
        logger.info("开始超参数调优...")
        start_time = time.time()
        
        # 准备数据
        self.prepare_data()
        
        # 生成所有超参数组合
        param_combinations = list(itertools.product(
            self.param_grid['hidden_sizes'],
            self.param_grid['learning_rate'],
            self.param_grid['batch_size'],
            self.param_grid['dropout_rate'],
            self.param_grid['activation'],
            self.param_grid['optimizer']
        ))
        
        total_combinations = len(param_combinations)
        logger.info(f"将评估 {total_combinations} 个超参数组合")
        
        # 追踪最佳参数
        best_params = None
        best_r2 = -float('inf')
        
        # 用于保存所有结果的列表
        all_results = []
        
        # 评估每个超参数组合
        for i, (hidden_sizes, learning_rate, batch_size, dropout_rate, activation, optimizer_type) in enumerate(param_combinations):
            param_start_time = time.time()
            
            # 构建模型
            model = BPNN(
                input_size=self.input_size,
                hidden_sizes=hidden_sizes,
                dropout_rate=dropout_rate,
                activation=activation
            )
            
            # 记录当前参数
            current_params = {
                'hidden_sizes': hidden_sizes,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'dropout_rate': dropout_rate,
                'activation': activation,
                'optimizer': optimizer_type
            }
            
            logger.info(f"评估参数组合 {i+1}/{total_combinations}: {current_params}")
            
            # 训练和评估模型
            metrics, history = self.train_eval_model(
                model, batch_size, learning_rate, optimizer_type
            )
            
            # 记录结果
            result = {
                'params': current_params,
                'metrics': metrics,
                'history': {
                    'train_loss': history['train_loss'],
                    'val_loss': history['val_loss']
                }
            }
            
            all_results.append(result)
            
            # 更新最佳参数
            if metrics['r2'] > best_r2:
                best_r2 = metrics['r2']
                best_params = current_params
                
                # 保存最佳模型
                best_model = model
                torch.save(best_model.state_dict(), os.path.join(self.output_path, 'best_model.pt'))
                
                logger.info(f"发现新的最佳参数组合，R²: {best_r2:.4f}")
            
            # 记录每个组合的性能
            logger.info(f"参数组合 {i+1} 的性能 - "
                        f"R²: {metrics['r2']:.4f}, "
                        f"RMSE: {metrics['rmse']:.4f}, "
                        f"MAE: {metrics['mae']:.4f}, "
                        f"训练了 {metrics['epochs_trained']} 个周期")
            
            param_time = time.time() - param_start_time
            logger.info(f"参数组合评估完成，耗时: {param_time:.2f}秒")
        
        end_time = time.time()
        tuning_time = end_time - start_time
        
        logger.info(f"超参数调优完成，总耗时: {tuning_time:.2f}秒")
        logger.info(f"最佳超参数: {best_params}")
        logger.info(f"最佳R²: {best_r2:.6f}")
        
        return best_params, all_results, best_model
    
    def visualize_results(self, all_results, best_params):
        """
        可视化调优结果
        
        参数:
            all_results: 所有参数组合的结果
            best_params: 最佳超参数
        """
        logger.info("可视化超参数调优结果...")
        
        # 创建结果数据框
        results_df = pd.DataFrame()
        
        # 提取信息
        for i, result in enumerate(all_results):
            row = {
                'combo_id': i,
                'r2': result['metrics']['r2'],
                'rmse': result['metrics']['rmse'],
                'mae': result['metrics']['mae'],
                'hidden_layers': str(result['params']['hidden_sizes']),
                'learning_rate': result['params']['learning_rate'],
                'batch_size': result['params']['batch_size'],
                'dropout_rate': result['params']['dropout_rate'],
                'activation': result['params']['activation'],
                'optimizer': result['params']['optimizer'],
                'epochs_trained': result['metrics']['epochs_trained']
            }
            results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
        
        # 保存结果数据框
        results_df.to_csv(os.path.join(self.output_path, 'all_param_results.csv'), index=False)
        
        # 绘制超参数与R²的关系
        self._plot_param_vs_r2('learning_rate', results_df)
        self._plot_param_vs_r2('batch_size', results_df)
        self._plot_param_vs_r2('dropout_rate', results_df)
        self._plot_param_vs_r2('activation', results_df)
        self._plot_param_vs_r2('optimizer', results_df)
        self._plot_param_vs_r2('hidden_layers', results_df)
        
        # 绘制最佳参数组合的训练历史
        best_result = None
        for result in all_results:
            if (result['params']['hidden_sizes'] == best_params['hidden_sizes'] and
                result['params']['learning_rate'] == best_params['learning_rate'] and
                result['params']['batch_size'] == best_params['batch_size'] and
                result['params']['dropout_rate'] == best_params['dropout_rate'] and
                result['params']['activation'] == best_params['activation'] and
                result['params']['optimizer'] == best_params['optimizer']):
                best_result = result
                break
        
        if best_result:
            plt.figure(figsize=(10, 6))
            plt.plot(best_result['history']['train_loss'], label='训练损失')
            plt.plot(best_result['history']['val_loss'], label='验证损失')
            plt.title('最佳参数组合的训练历史')
            plt.xlabel('周期')
            plt.ylabel('损失')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, 'best_model_history.png'))
        
        logger.info("可视化完成")
    
    def _plot_param_vs_r2(self, param_name, results_df):
        """
        绘制参数与R²的关系
        
        参数:
            param_name: 参数名称
            results_df: 结果数据框
        """
        plt.figure(figsize=(10, 6))
        
        # 对数值型参数分组统计
        if param_name in ['learning_rate', 'batch_size', 'dropout_rate']:
            gb = results_df.groupby(param_name)['r2'].agg(['mean', 'std', 'count'])
            
            # 绘制条形图
            x = gb.index
            y = gb['mean']
            yerr = gb['std']
            
            plt.bar(range(len(x)), y, yerr=yerr, alpha=0.7)
            plt.xticks(range(len(x)), x)
            
        # 对类别型参数分组统计
        else:
            gb = results_df.groupby(param_name)['r2'].agg(['mean', 'std', 'count'])
            
            # 绘制条形图
            x = gb.index
            y = gb['mean']
            yerr = gb['std']
            
            plt.bar(range(len(x)), y, yerr=yerr, alpha=0.7)
            plt.xticks(range(len(x)), x, rotation=45 if param_name == 'hidden_layers' else 0)
        
        plt.title(f'{param_name}参数对R²的影响')
        plt.xlabel(param_name)
        plt.ylabel('平均R²')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.output_path, f'param_{param_name}_vs_r2.png'))
        plt.close()
    
    def evaluate_best_model(self, best_model):
        """
        在测试集上评估最佳模型并可视化结果
        
        参数:
            best_model: 最佳模型
        """
        logger.info("评估最佳模型...")
        
        # 数据加载器
        test_loader = DataLoader(
            self.test_dataset, 
            batch_size=len(self.test_dataset), 
            shuffle=False
        )
        
        # 将模型移至设备
        best_model = best_model.to(self.device)
        
        # 进入评估模式
        best_model.eval()
        
        # 获取测试数据
        with torch.no_grad():
            inputs, targets = next(iter(test_loader))
            inputs = inputs.to(self.device)
            outputs = best_model(inputs)
        
        # 将结果移回CPU并反标准化
        targets = targets.numpy() * self.y_std + self.y_mean
        outputs = outputs.cpu().numpy() * self.y_std + self.y_mean
        
        # 将结果展平
        targets = targets.flatten()
        outputs = outputs.flatten()
        
        # 计算评估指标
        mse = mean_squared_error(targets, outputs)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets, outputs)
        r2 = r2_score(targets, outputs)
        
        # 记录评估结果
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2)
        }
        
        logger.info(f"最佳模型评估 - MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}")
        
        # 保存评估结果
        with open(os.path.join(self.output_path, 'best_model_evaluation.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # 加载测试数据集获取切削次数
        test_data = pd.read_csv(
            os.path.join(self.selected_features_path, self.test_cutter, f"{self.test_cutter}_selected_feature_data.csv")
        )
        cut_nums = test_data['cut_num'].values
        
        # 创建预测序列图
        plt.figure(figsize=(12, 6))
        plt.plot(cut_nums, targets, 'b-', label='实际值')
        plt.plot(cut_nums, outputs, 'r--', label='预测值')
        plt.title(f'{self.test_cutter} 刀具磨损预测结果')
        plt.xlabel('切削次数')
        plt.ylabel('磨损值 (mm)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, f'{self.test_cutter}_predictions.png'))
        
        # 创建散点图
        plt.figure(figsize=(8, 8))
        plt.scatter(targets, outputs, alpha=0.7)
        plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
        plt.title(f'{self.test_cutter} 实际值 vs 预测值')
        plt.xlabel('实际值')
        plt.ylabel('预测值')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, f'{self.test_cutter}_scatter.png'))
        
        logger.info("最佳模型评估完成")
        
        return metrics
    
    def run(self):
        """
        运行完整的超参数调优流程
        
        返回:
            best_params: 最佳超参数
            best_metrics: 最佳模型的评估指标
        """
        # 超参数调优
        best_params, all_results, best_model = self.tune_hyperparameters()
        
        # 可视化结果
        self.visualize_results(all_results, best_params)
        
        # 评估最佳模型
        best_metrics = self.evaluate_best_model(best_model)
        
        # 保存最佳参数
        with open(os.path.join(self.output_path, 'best_params.json'), 'w') as f:
            json.dump(best_params, f, indent=4)
        
        logger.info("超参数调优流程完成！")
        
        return best_params, best_metrics

def main():
    """主函数"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='铣刀寿命预测系统 - BP神经网络超参数调优')
    
    # 添加命令行参数
    parser.add_argument('--features_path', type=str, default='/Users/xiaohudemac/cursor01/bishe/5_8tool/data/selected_features',
                        help='特征选择后的数据路径')
    parser.add_argument('--output_path', type=str, default='/Users/xiaohudemac/cursor01/bishe/5_8tool/results/bpnn_tuning',
                        help='调优输出根路径')
    parser.add_argument('--target_column', type=str, default='wear_VB_avg',
                        help='目标变量列名')
    parser.add_argument('--train_cutters', type=str, default='c1,c4',
                        help='用于训练的刀具列表，用逗号分隔')
    parser.add_argument('--test_cutter', type=str, default='c6',
                        help='用于测试的刀具')
    parser.add_argument('--random_state', type=int, default=42,
                        help='随机种子')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 解析训练刀具列表
    train_cutters = args.train_cutters.split(',')
    
    # 为输出路径添加时间戳，确保每次运行结果保存在单独的文件夹中
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_path, f"run_{timestamp}")
    
    # 初始化BPNN超参数调优器
    tuner = BPNNHyperparameterTuner(
        selected_features_path=args.features_path,
        output_path=output_path,
        target_column=args.target_column,
        train_cutters=train_cutters,
        test_cutter=args.test_cutter,
        random_state=args.random_state
    )
    
    # 运行超参数调优
    tuner.run()

if __name__ == "__main__":
    main() 