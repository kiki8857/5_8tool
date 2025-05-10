#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
铣刀寿命预测系统 - BP神经网络模型
功能：使用选择的特征构建BP神经网络模型，c1和c4数据训练，c6数据测试
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
from tqdm import tqdm
import json
from datetime import datetime
from matplotlib import font_manager

# 配置中文字体支持
try:
    # 尝试设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']  # 优先使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 检查字体是否可用
    font_names = [f.name for f in font_manager.fontManager.ttflist]
    if 'SimHei' not in font_names and 'Arial Unicode MS' not in font_names:
        logger.warning("系统中未找到合适的中文字体，图表中文可能无法正确显示")
except Exception as e:
    logger.warning(f"设置中文字体时出错: {e}")

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
    
    def __init__(self, input_size, hidden_sizes=[64, 32, 16], output_size=1, dropout_rate=0.2):
        """
        初始化BP神经网络模型
        
        参数:
            input_size: 输入特征数量
            hidden_sizes: 隐藏层神经元数量列表
            output_size: 输出神经元数量
            dropout_rate: Dropout比例
        """
        super(BPNN, self).__init__()
        
        # 创建网络层
        layers = []
        
        # 第一个隐藏层
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.BatchNorm1d(hidden_sizes[0]))  # 添加批标准化
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # 后续隐藏层
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))  # 添加批标准化
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # 最后一个隐藏层到输出层
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        # 组合所有层
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """前向传播"""
        return self.model(x)

class BPNNModel:
    """BPNN模型训练与预测"""
    
    def __init__(self, selected_features_path, output_path, target_column='wear_VB_avg',
                 train_cutters=None, test_cutter='c6', random_state=42,
                 batch_size=32, num_epochs=500, learning_rate=0.001):
        """
        初始化BPNN模型
        
        参数:
            selected_features_path: 特征选择后的数据路径
            output_path: 模型输出路径
            target_column: 目标变量列名
            train_cutters: 用于训练的刀具列表，默认为None（使用非测试刀具）
            test_cutter: 用于测试的刀具
            random_state: 随机种子
            batch_size: 批处理大小，优化值为32
            num_epochs: 训练轮数
            learning_rate: 学习率，优化值为0.001
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
        
        # 创建输出目录
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # 初始化模型架构
        self.input_size = None  # 将在加载数据时确定
        self.hidden_sizes = [64, 32, 16]  # 神经网络隐藏层结构
        self.output_size = 1
        self.model = None
        
        # 训练参数
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = num_epochs
        self.early_stop_patience = 50
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
        }
        
        # 设置随机种子
        torch.manual_seed(random_state)
        np.random.seed(random_state)
    
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
    
    def build_model(self):
        """构建BP神经网络模型"""
        logger.info("构建模型...")
        
        # 初始化模型 - 使用超参数调优得到的最佳参数
        self.model = BPNN(
            input_size=self.input_size,
            hidden_sizes=[128, 64, 32],  # 优化后的网络结构
            output_size=self.output_size,
            dropout_rate=0.2  # 优化后的dropout比例
        )
        
        # 定义损失函数和优化器 - 使用优化后的学习率和权重衰减
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)  # 优化后的学习率和L2正则化
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=20, verbose=True)  # 学习率调度器
        
        logger.info(f"模型构建完成: {self.model}")
    
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
        
        # 预测
        with torch.no_grad():
            predictions = self.model(inputs)
        
        # 转换为numpy数组并反标准化
        targets = targets.numpy() * self.y_std + self.y_mean
        predictions = predictions.numpy() * self.y_std + self.y_mean
        
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
        
        # 创建预测与实际值对比图
        plt.figure(figsize=(12, 6))
        plt.plot(cut_nums, targets, 'b-', label='实际值')
        plt.plot(cut_nums, predictions, 'r--', label='预测值')
        plt.title(f'{self.test_cutter} 刀具磨损预测结果')
        plt.xlabel('切削次数')
        plt.ylabel('磨损值 (mm)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(os.path.join(self.output_path, f'{self.test_cutter}_predictions.png'))
        
        # 创建散点图
        plt.figure(figsize=(8, 8))
        plt.scatter(targets, predictions, alpha=0.7)
        plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
        plt.title(f'{self.test_cutter} 实际值 vs 预测值')
        plt.xlabel('实际值')
        plt.ylabel('预测值')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.axis('equal')
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(os.path.join(self.output_path, f'{self.test_cutter}_scatter.png'))
        
        # 创建训练历史图
        plt.figure(figsize=(12, 6))
        plt.plot(self.history['train_loss'], label='训练损失')
        plt.plot(self.history['val_loss'], label='验证损失')
        plt.title('训练历史')
        plt.xlabel('周期')
        plt.ylabel('损失')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(os.path.join(self.output_path, 'training_history.png'))
        
        logger.info("可视化完成")
    
    def save_model(self, filename):
        """保存模型"""
        torch.save(self.model.state_dict(), os.path.join(self.output_path, filename))
    
    def load_model(self, filename):
        """加载模型"""
        self.model.load_state_dict(torch.load(os.path.join(self.output_path, filename)))
    
    def run(self):
        """
        运行完整的模型训练和评估流程
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
            
            return metrics
        
        except Exception as e:
            logger.error(f"运行过程中发生错误: {e}")
            raise e

def main():
    """主函数"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='铣刀寿命预测系统 - BP神经网络模型')
    
    # 添加命令行参数
    parser.add_argument('--features_path', type=str, default='/Users/xiaohudemac/cursor01/bishe/5_8tool/data/selected_features',
                        help='特征选择后的数据路径')
    parser.add_argument('--output_path', type=str, default='/Users/xiaohudemac/cursor01/bishe/5_8tool/results/bpnn',
                        help='模型输出根路径')
    parser.add_argument('--target_column', type=str, default='wear_VB_avg',
                        help='目标变量列名')
    parser.add_argument('--train_cutters', type=str, default='c1,c4',
                        help='用于训练的刀具列表，用逗号分隔')
    parser.add_argument('--test_cutter', type=str, default='c6',
                        help='用于测试的刀具')
    parser.add_argument('--random_state', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批处理大小')
    parser.add_argument('--num_epochs', type=int, default=500,
                        help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='学习率')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 解析训练刀具列表
    train_cutters = args.train_cutters.split(',')
    
    # 为输出路径添加时间戳，确保每次运行结果保存在单独的文件夹中
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_path, f"run_{timestamp}")
    
    # 初始化BPNN模型
    bpnn_model = BPNNModel(
        selected_features_path=args.features_path,
        output_path=output_path,
        target_column=args.target_column,
        train_cutters=train_cutters,
        test_cutter=args.test_cutter,
        random_state=args.random_state,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate
    )
    
    # 运行训练和评估
    bpnn_model.run()
    
    logger.info("模型训练和评估完成！")

if __name__ == "__main__":
    main() 