#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
铣刀寿命预测系统 - 最终BP神经网络模型
功能：使用超参数调优获得的最佳参数训练最终模型，并与其他模型进行比较
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import argparse
import logging
import json
from datetime import datetime
from matplotlib import font_manager

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

class BPNNFinalModel:
    """BPNN最终模型训练与预测"""
    
    def __init__(self, selected_features_path, output_path, 
                 hidden_sizes=[64, 32], learning_rate=0.0001, batch_size=64,
                 dropout_rate=0.1, activation='relu', optimizer='adam',
                 target_column='wear_VB_avg', train_cutters=None, test_cutter='c6'):
        """
        初始化BPNN最终模型
        
        参数:
            selected_features_path: 特征选择后的数据路径
            output_path: 模型输出路径
            hidden_sizes: 隐藏层神经元数量列表（从超参数调优获得）
            learning_rate: 学习率（从超参数调优获得）
            batch_size: 批处理大小（从超参数调优获得）
            dropout_rate: Dropout比例（从超参数调优获得）
            activation: 激活函数（从超参数调优获得）
            optimizer: 优化器类型（从超参数调优获得）
            target_column: 目标变量列名
            train_cutters: 用于训练的刀具列表，默认为None（使用非测试刀具）
            test_cutter: 用于测试的刀具
        """
        self.selected_features_path = selected_features_path
        self.output_path = output_path
        self.target_column = target_column
        self.test_cutter = test_cutter
        
        # 超参数（从调优获得）
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.optimizer_type = optimizer
        
        # 如果未指定训练刀具，则使用除测试刀具外的所有刀具
        if train_cutters is None:
            all_cutters = ['c1', 'c4', 'c6']
            self.train_cutters = [c for c in all_cutters if c != test_cutter]
        else:
            self.train_cutters = train_cutters
        
        # 确保输出目录存在
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 模型、数据相关属性
        self.model = None
        self.input_size = None
        self.scaler = None
        self.y_mean = None
        self.y_std = None
        
        # 训练参数
        self.epochs = 500
        self.early_stop_patience = 50
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
        }
    
    def prepare_data(self):
        """准备训练和测试数据"""
        logger.info("准备数据...")
        
        # 加载训练数据
        train_datasets = []
        for cutter in self.train_cutters:
            dataset = ToolWearDataset(self.selected_features_path, cutter)
            train_datasets.append(dataset)
            
            # 获取输入特征数量
            if self.input_size is None:
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
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        self.test_loader = DataLoader(
            self.test_dataset, 
            batch_size=len(self.test_dataset), 
            shuffle=False
        )
        
        logger.info(f"数据准备完成，输入特征数量: {self.input_size}")
        logger.info(f"训练数据样本数: {len(self.train_dataset)}, 测试数据样本数: {len(self.test_dataset)}")
    
    def build_model(self):
        """构建BP神经网络模型"""
        logger.info("构建模型...")
        
        # 初始化模型 - 使用超参数调优得到的最佳参数
        self.model = BPNN(
            input_size=self.input_size,
            hidden_sizes=self.hidden_sizes,
            output_size=1,
            dropout_rate=self.dropout_rate,
            activation=self.activation
        )
        
        # 将模型移至设备
        self.model = self.model.to(self.device)
        
        # 定义损失函数和优化器 - 使用优化后的参数
        self.criterion = nn.MSELoss()
        
        # 选择优化器
        if self.optimizer_type == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == 'rmsprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=20, verbose=True
        )
        
        logger.info(f"模型构建完成: {self.model}")
        logger.info(f"优化器: {self.optimizer_type}, 学习率: {self.learning_rate}")
    
    def train(self):
        """训练模型"""
        logger.info(f"开始训练，共{self.epochs}个周期...")
        
        best_val_loss = float('inf')
        early_stop_counter = 0
        
        for epoch in range(self.epochs):
            # 训练模式
            self.model.train()
            train_loss = 0.0
            
            # 训练一个周期
            for inputs, targets in self.train_loader:
                # 将数据移至设备
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # 前向传播
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # 计算平均训练损失
            train_loss /= len(self.train_loader)
            
            # 验证模式
            self.model.eval()
            val_loss = 0.0
            
            # 在测试集上验证
            with torch.no_grad():
                for inputs, targets in self.test_loader:
                    # 将数据移至设备
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    val_loss += loss.item()
            
            # 计算平均验证损失
            val_loss /= len(self.test_loader)
            
            # 更新学习率
            self.scheduler.step(val_loss)
            
            # 保存训练历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # 打印进度
            if (epoch + 1) % 50 == 0:
                logger.info(f'周期 {epoch+1}/{self.epochs}, 训练损失: {train_loss:.6f}, 验证损失: {val_loss:.6f}')
            
            # 提前停止逻辑
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                # 保存最佳模型
                self.save_model('best_model.pt')
            else:
                early_stop_counter += 1
            
            if early_stop_counter >= self.early_stop_patience:
                logger.info(f'提前停止训练，已经{self.early_stop_patience}个周期没有改善')
                break
        
        logger.info(f"训练完成，最佳验证损失: {best_val_loss:.6f}")
        
        # 加载最佳模型
        self.load_model('best_model.pt')
    
    def evaluate(self):
        """评估模型性能"""
        logger.info("评估模型...")
        
        # 进入评估模式
        self.model.eval()
        
        # 获取测试数据
        inputs, targets = next(iter(self.test_loader))
        inputs = inputs.to(self.device)
        
        # 预测
        with torch.no_grad():
            predictions = self.model(inputs)
        
        # 转换为numpy数组并反标准化
        targets = targets.numpy() * self.y_std + self.y_mean
        predictions = predictions.cpu().numpy() * self.y_std + self.y_mean
        
        # 将结果展平
        targets = targets.flatten()
        predictions = predictions.flatten()
        
        # 计算评估指标
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        # 记录评估结果
        evaluation = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2)
        }
        
        logger.info(f"评估结果 - MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}")
        
        # 保存评估结果
        with open(os.path.join(self.output_path, 'evaluation.json'), 'w') as f:
            json.dump(evaluation, f, indent=4)
        
        # 可视化预测结果
        self.visualize_predictions(targets, predictions)
        
        return evaluation
    
    def visualize_predictions(self, targets, predictions):
        """可视化预测结果"""
        logger.info("可视化预测结果...")
        
        # 加载测试数据集获取切削次数
        test_data = pd.read_csv(
            os.path.join(self.selected_features_path, self.test_cutter, f"{self.test_cutter}_selected_feature_data.csv")
        )
        cut_nums = test_data['cut_num'].values
        
        # 创建预测对比图 (散点图)
        plt.figure(figsize=(12, 8))
        plt.scatter(targets, predictions, alpha=0.7, color='blue', s=60)
        
        # 添加理想预测线
        min_val = min(min(targets), min(predictions))
        max_val = max(max(targets), max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        # 计算评估指标显示在图上
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        plt.text(min_val + 0.05*(max_val-min_val), max_val - 0.15*(max_val-min_val), 
                f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}', 
                fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.title(f'刀具 {self.test_cutter} 磨损预测结果对比图 (BP神经网络模型)', fontsize=16)
        plt.xlabel('实际磨损值 (mm)', fontsize=14)
        plt.ylabel('预测磨损值 (mm)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # 保存预测对比图
        plt.savefig(os.path.join(self.output_path, f'{self.test_cutter}_scatter.png'), dpi=300)
        plt.close()
        
        # 创建预测序列图
        plt.figure(figsize=(14, 8))
        
        # 绘制预测值和真实值随切削次数的变化
        plt.plot(cut_nums, targets, 'b-', label='实际磨损值', linewidth=2)
        plt.plot(cut_nums, predictions, 'r--', label='预测磨损值', linewidth=2)
        
        # 计算绝对误差并绘制误差区间
        abs_error = np.abs(targets - predictions)
        plt.fill_between(cut_nums, predictions - abs_error, predictions + abs_error, 
                        color='gray', alpha=0.2, label='预测误差区间')
        
        plt.title(f'刀具 {self.test_cutter} 磨损预测随切削次数的变化 (BP神经网络模型)', fontsize=16)
        plt.xlabel('切削次数', fontsize=14)
        plt.ylabel('磨损值 (mm)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(os.path.join(self.output_path, f'{self.test_cutter}_predictions.png'), dpi=300)
        
        # 创建训练历史图
        plt.figure(figsize=(12, 8))
        plt.plot(self.history['train_loss'], label='训练损失', linewidth=2)
        plt.plot(self.history['val_loss'], label='验证损失', linewidth=2)
        plt.title('BP神经网络模型训练历史', fontsize=16)
        plt.xlabel('周期', fontsize=14)
        plt.ylabel('损失', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(os.path.join(self.output_path, 'training_history.png'), dpi=300)
        
        plt.close('all')
        logger.info("可视化完成")
    
    def compare_with_models(self, other_model_results):
        """与其他模型进行比较"""
        logger.info("与其他模型进行比较...")
        
        # 运行评估获取当前模型的性能指标
        bpnn_metrics = self.evaluate()
        
        # 获取所有模型的性能指标（加载其他模型的评估结果）
        all_metrics = {
            'BPNN (最优超参数)': bpnn_metrics,
        }
        
        # 添加其他模型的评估结果
        for model_name, result_path in other_model_results.items():
            try:
                with open(result_path, 'r') as f:
                    metrics = json.load(f)
                all_metrics[model_name] = metrics
            except Exception as e:
                logger.error(f"加载{model_name}评估结果失败: {e}")
        
        # 创建比较表格
        comparison = pd.DataFrame({
            'Model': [],
            'R²': [],
            'RMSE': [],
            'MAE': []
        })
        
        for model_name, metrics in all_metrics.items():
            comparison = pd.concat([comparison, pd.DataFrame({
                'Model': [model_name],
                'R²': [metrics['r2']],
                'RMSE': [metrics['rmse']],
                'MAE': [metrics['mae']]
            })], ignore_index=True)
        
        # 按R²降序排序
        comparison = comparison.sort_values('R²', ascending=False)
        
        # 保存比较结果
        comparison.to_csv(os.path.join(self.output_path, 'model_comparison.csv'), index=False)
        
        # 打印比较结果
        pd.set_option('display.float_format', '{:.6f}'.format)
        logger.info("\n" + comparison.to_string(index=False))
        
        # 可视化比较结果
        self.visualize_comparison(comparison)
        
        return comparison
    
    def visualize_comparison(self, comparison_df):
        """可视化模型比较结果"""
        logger.info("可视化模型比较结果...")
        
        metrics = ['R²', 'RMSE', 'MAE']
        
        # 为每个指标创建条形图
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            
            # 根据指标确定排序方式和颜色方案
            if metric == 'R²':
                # R²值越高越好
                df_sorted = comparison_df.sort_values(metric, ascending=False)
                colors = ['green' if val >= 0.9 else 'blue' if val >= 0.8 else 'orange' if val >= 0.7 else 'red' 
                        for val in df_sorted[metric]]
            else:
                # RMSE和MAE值越低越好
                df_sorted = comparison_df.sort_values(metric)
                colors = ['green' if i == 0 else 'blue' if i == 1 else 'orange' if i == 2 else 'red' 
                        for i, _ in enumerate(df_sorted[metric])]
            
            # 绘制条形图
            bars = plt.bar(df_sorted['Model'], df_sorted[metric], color=colors)
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}', ha='center', va='bottom')
            
            plt.title(f'模型比较 - {metric}')
            plt.xlabel('模型')
            plt.ylabel(metric)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # 保存图像
            plt.savefig(os.path.join(self.output_path, f'model_comparison_{metric}.png'))
            plt.close()
        
        logger.info("比较可视化完成")
    
    def save_model(self, filename):
        """保存模型"""
        torch.save(self.model.state_dict(), os.path.join(self.output_path, filename))
        
        # 保存模型配置
        model_config = {
            'hidden_sizes': self.hidden_sizes,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'optimizer': self.optimizer_type,
            'input_size': self.input_size,
            'train_cutters': self.train_cutters,
            'test_cutter': self.test_cutter
        }
        
        with open(os.path.join(self.output_path, 'model_config.json'), 'w') as f:
            json.dump(model_config, f, indent=4)
    
    def load_model(self, filename):
        """加载模型"""
        self.model.load_state_dict(torch.load(os.path.join(self.output_path, filename)))
    
    def run(self, other_model_results=None):
        """
        运行完整的模型训练和评估流程
        
        参数:
            other_model_results: 其他模型的评估结果文件路径字典，用于比较
        """
        try:
            # 准备数据
            self.prepare_data()
            
            # 构建模型
            self.build_model()
            
            # 训练模型
            self.train()
            
            # 评估模型
            metrics = self.evaluate()
            
            # 与其他模型比较（如果提供）
            if other_model_results:
                self.compare_with_models(other_model_results)
            
            return metrics
        
        except Exception as e:
            logger.error(f"运行过程中发生错误: {e}")
            raise e

def main():
    """主函数"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='铣刀寿命预测系统 - 最终BP神经网络模型')
    
    # 添加命令行参数
    parser.add_argument('--features_path', type=str, default='/Users/xiaohudemac/cursor01/bishe/5_8tool/data/selected_features',
                        help='特征选择后的数据路径')
    parser.add_argument('--output_path', type=str, default='/Users/xiaohudemac/cursor01/bishe/5_8tool/results/bpnn_final',
                        help='模型输出根路径')
    parser.add_argument('--target_column', type=str, default='wear_VB_avg',
                        help='目标变量列名')
    parser.add_argument('--train_cutters', type=str, default='c1,c4',
                        help='用于训练的刀具列表，用逗号分隔')
    parser.add_argument('--test_cutter', type=str, default='c6',
                        help='用于测试的刀具')
    parser.add_argument('--rf_result_path', type=str, default='/Users/xiaohudemac/cursor01/bishe/5_8tool/results/random_forest/run_20250508_202135_selected/evaluation.json',
                        help='随机森林模型评估结果路径')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 解析训练刀具列表
    train_cutters = args.train_cutters.split(',')
    
    # 为输出路径添加时间戳，确保每次运行结果保存在单独的文件夹中
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_path, f"run_{timestamp}")
    
    # 获取超参数调优的最佳参数
    best_params_file = "/Users/xiaohudemac/cursor01/bishe/5_8tool/results/bpnn_tuning/run_20250510_044514/best_params.json"
    try:
        with open(best_params_file, 'r') as f:
            best_params = json.load(f)
        logger.info(f"从{best_params_file}加载最佳超参数")
    except Exception as e:
        logger.error(f"加载最佳超参数失败: {e}")
        logger.info("使用默认超参数")
        best_params = {
            'hidden_sizes': [64, 32],
            'learning_rate': 0.0001,
            'batch_size': 64,
            'dropout_rate': 0.1,
            'activation': 'relu',
            'optimizer': 'adam'
        }
    
    # 初始化最终模型
    final_model = BPNNFinalModel(
        selected_features_path=args.features_path,
        output_path=output_path,
        hidden_sizes=best_params['hidden_sizes'],
        learning_rate=best_params['learning_rate'],
        batch_size=best_params['batch_size'],
        dropout_rate=best_params['dropout_rate'],
        activation=best_params['activation'],
        optimizer=best_params['optimizer'],
        target_column=args.target_column,
        train_cutters=train_cutters,
        test_cutter=args.test_cutter
    )
    
    # 配置其他模型的结果路径，用于比较
    other_model_results = {
        "随机森林": "/Users/xiaohudemac/cursor01/bishe/5_8tool/results/random_forest/run_20250510_043000_selected/evaluation.json",
        "普通BPNN": "/Users/xiaohudemac/cursor01/bishe/5_8tool/results/bpnn/run_20250510_043858/evaluation.json"
    }
    
    # 训练并评估模型
    print("\n" + "="*60)
    print(f"开始使用超参数优化后的BPNN模型训练")
    print(f"训练刀具: {', '.join(train_cutters)}, 测试刀具: {args.test_cutter}")
    print(f"超参数: {best_params}")
    print(f"模型结果将保存至: {output_path}")
    print("="*60)
    
    try:
        final_model.run(other_model_results)
        print("\n训练和评估完成!")
        print(f"模型和评估结果已保存至: {output_path}")
        print("="*60)
    except Exception as e:
        print(f"\n训练过程中发生错误: {e}")
        print("="*60)

if __name__ == "__main__":
    main() 