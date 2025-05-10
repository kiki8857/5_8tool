#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
铣刀寿命预测系统 - LSTM神经网络模型
功能：使用LSTM网络构建刀具磨损预测模型，c1和c4数据训练，c6数据测试
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import argparse
import logging
from tqdm import tqdm
import json

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置随机种子以便结果可复现
torch.manual_seed(42)
np.random.seed(42)

class TimeSeriesDataset(Dataset):
    """时间序列数据集"""
    
    def __init__(self, features_path, cutter, seq_length=5, scaler=None, is_train=True):
        """
        初始化数据集
        
        参数:
            features_path: 特征数据路径
            cutter: 刀具名称(c1, c4, c6)
            seq_length: 序列长度（使用前几个切削次数的数据预测下一个）
            scaler: 特征标准化器，如果为None则创建新的
            is_train: 是否为训练集
        """
        self.features_path = features_path
        self.cutter = cutter
        self.seq_length = seq_length
        self.is_train = is_train
        
        # 加载数据
        self.data = self._load_data()
        
        # 按切削次数排序
        self.data = self.data.sort_values('cut_num')
        
        # 分离特征和目标变量
        self.X_raw = self.data.drop(['cut_num', 'wear_VB_avg'], axis=1).values
        self.y_raw = self.data['wear_VB_avg'].values
        self.cut_nums = self.data['cut_num'].values
        
        # 标准化特征
        if scaler is None:
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            self.X = self.scaler.fit_transform(self.X_raw)
        else:
            self.scaler = scaler
            self.X = self.scaler.transform(self.X_raw)
        
        # 对目标变量进行标准化（使用MinMaxScaler更适合LSTM）
        if is_train:
            self.y_scaler = MinMaxScaler(feature_range=(0, 1))
            self.y = self.y_scaler.fit_transform(self.y_raw.reshape(-1, 1)).flatten()
        else:
            self.y_scaler = scaler[1]  # 使用传入的y_scaler
            self.y = self.y_scaler.transform(self.y_raw.reshape(-1, 1)).flatten()
        
        # 创建序列数据
        self.X_seqs, self.y_seqs = self._create_sequences()
    
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
    
    def _create_sequences(self):
        """创建序列数据"""
        X_seqs = []
        y_seqs = []
        
        # 创建输入序列和目标值序列
        for i in range(len(self.X) - self.seq_length):
            X_seqs.append(self.X[i:i+self.seq_length])
            y_seqs.append(self.y[i+self.seq_length])
        
        return np.array(X_seqs), np.array(y_seqs)
    
    def __len__(self):
        """返回数据集长度"""
        return len(self.X_seqs)
    
    def __getitem__(self, idx):
        """获取指定索引的数据"""
        return torch.FloatTensor(self.X_seqs[idx]), torch.FloatTensor([self.y_seqs[idx]])
    
    def get_scalers(self):
        """返回特征和目标变量标准化器"""
        if self.is_train:
            return self.scaler, self.y_scaler
        return None, None

class LSTMModel(nn.Module):
    """LSTM模型"""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        """
        初始化LSTM模型
        
        参数:
            input_size: 输入特征数量
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
            output_size: 输出变量数量
            dropout: Dropout比率
        """
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
        
        # Dropout层，防止过拟合
        self.dropout = nn.Dropout(dropout)
        
        # 全连接层
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """前向传播"""
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

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, features_path, output_path, seq_length=5, train_cutters=['c1', 'c4'], test_cutter='c6'):
        """
        初始化模型训练器
        
        参数:
            features_path: 特征数据路径
            output_path: 输出路径
            seq_length: 序列长度
            train_cutters: 训练集刀具列表
            test_cutter: 测试集刀具
        """
        self.features_path = features_path
        self.output_path = output_path
        self.seq_length = seq_length
        self.train_cutters = train_cutters
        self.test_cutter = test_cutter
        
        # 创建输出目录
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # 初始化模型架构
        self.input_size = None  # 将在加载数据时确定
        self.hidden_size = 64
        self.num_layers = 2
        self.output_size = 1
        self.model = None
        
        # 训练参数
        self.batch_size = 8
        self.learning_rate = 0.001
        self.epochs = 1000
        self.early_stop_patience = 100
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
        }
    
    def prepare_data(self):
        """准备训练和测试数据"""
        logger.info("准备数据...")
        
        # 加载所有训练数据集的所有数据，合并后创建序列
        all_X_train = []
        all_y_train = []
        all_cuts_train = []
        
        for cutter in self.train_cutters:
            # 加载数据
            data_file = os.path.join(self.features_path, cutter, f"{cutter}_selected_feature_data.csv")
            data = pd.read_csv(data_file)
            
            # 按切削次数排序
            data = data.sort_values('cut_num')
            
            # 分离特征和目标变量
            X = data.drop(['cut_num', 'wear_VB_avg'], axis=1).values
            y = data['wear_VB_avg'].values
            cuts = data['cut_num'].values
            
            all_X_train.append(X)
            all_y_train.append(y)
            all_cuts_train.append(cuts)
            
            # 获取输入特征数量
            if self.input_size is None:
                self.input_size = X.shape[1]
        
        # 合并所有训练数据
        X_train_combined = np.vstack(all_X_train)
        y_train_combined = np.concatenate(all_y_train)
        
        # 创建特征标准化器
        self.X_scaler = MinMaxScaler(feature_range=(0, 1))
        X_train_scaled = self.X_scaler.fit_transform(X_train_combined)
        
        # 创建目标变量标准化器
        self.y_scaler = MinMaxScaler(feature_range=(0, 1))
        y_train_scaled = self.y_scaler.fit_transform(y_train_combined.reshape(-1, 1)).flatten()
        
        # 创建用于训练的时间序列数据集
        self.train_datasets = []
        
        for i, cutter in enumerate(self.train_cutters):
            # 获取当前刀具的数据
            X = all_X_train[i]
            y = all_y_train[i]
            
            # 标准化
            X_scaled = self.X_scaler.transform(X)
            y_scaled = self.y_scaler.transform(y.reshape(-1, 1)).flatten()
            
            # 创建序列
            X_seqs = []
            y_seqs = []
            
            for j in range(len(X_scaled) - self.seq_length):
                X_seqs.append(X_scaled[j:j+self.seq_length])
                y_seqs.append(y_scaled[j+self.seq_length])
            
            if len(X_seqs) > 0:
                # 创建数据集
                X_tensor = torch.FloatTensor(X_seqs)
                y_tensor = torch.FloatTensor(y_seqs).unsqueeze(1)
                
                self.train_datasets.append((X_tensor, y_tensor))
        
        # 合并所有训练数据集的序列
        X_train_all = []
        y_train_all = []
        
        for X_tensor, y_tensor in self.train_datasets:
            X_train_all.append(X_tensor)
            y_train_all.append(y_tensor)
        
        if len(X_train_all) > 0:
            self.X_train = torch.cat(X_train_all, dim=0)
            self.y_train = torch.cat(y_train_all, dim=0)
            
            logger.info(f"训练数据形状: {self.X_train.shape}, {self.y_train.shape}")
        else:
            logger.error("没有足够的训练数据创建序列")
            return
        
        # 加载测试数据
        test_data_file = os.path.join(self.features_path, self.test_cutter, f"{self.test_cutter}_selected_feature_data.csv")
        test_data = pd.read_csv(test_data_file)
        
        # 按切削次数排序
        test_data = test_data.sort_values('cut_num')
        
        # 分离特征和目标变量
        self.X_test_raw = test_data.drop(['cut_num', 'wear_VB_avg'], axis=1).values
        self.y_test_raw = test_data['wear_VB_avg'].values
        self.test_cut_nums = test_data['cut_num'].values
        
        # 标准化测试数据
        self.X_test_scaled = self.X_scaler.transform(self.X_test_raw)
        self.y_test_scaled = self.y_scaler.transform(self.y_test_raw.reshape(-1, 1)).flatten()
        
        # 创建测试数据序列
        X_test_seqs = []
        y_test_seqs = []
        self.test_cut_seqs = []
        
        for i in range(len(self.X_test_scaled) - self.seq_length):
            X_test_seqs.append(self.X_test_scaled[i:i+self.seq_length])
            y_test_seqs.append(self.y_test_scaled[i+self.seq_length])
            self.test_cut_seqs.append(self.test_cut_nums[i+self.seq_length])
        
        self.X_test = torch.FloatTensor(X_test_seqs)
        self.y_test = torch.FloatTensor(y_test_seqs).unsqueeze(1)
        
        logger.info(f"测试数据形状: {self.X_test.shape}, {self.y_test.shape}")
        logger.info(f"数据准备完成，输入特征数量: {self.input_size}")
    
    def build_model(self):
        """构建LSTM模型"""
        logger.info("构建模型...")
        
        # 初始化模型
        self.model = LSTMModel(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=self.output_size,
            dropout=0.3
        )
        
        # 定义损失函数和优化器
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=50, verbose=True
        )
        
        logger.info(f"模型构建完成: {self.model}")
    
    def train(self):
        """训练模型"""
        logger.info(f"开始训练，共{self.epochs}个周期...")
        
        best_val_loss = float('inf')
        early_stop_counter = 0
        
        for epoch in range(self.epochs):
            # 训练模式
            self.model.train()
            
            # 前向传播和反向传播
            self.optimizer.zero_grad()
            outputs = self.model(self.X_train)
            loss = self.criterion(outputs, self.y_train)
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            train_loss = loss.item()
            
            # 验证模式
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(self.X_test)
                val_loss = self.criterion(val_outputs, self.y_test).item()
            
            # 更新学习率
            self.scheduler.step(val_loss)
            
            # 保存训练历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # 打印进度
            if (epoch + 1) % 100 == 0:
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
        
        # 预测
        with torch.no_grad():
            predictions_scaled = self.model(self.X_test)
        
        # 转换为numpy数组并反标准化
        predictions_scaled = predictions_scaled.numpy()
        targets_scaled = self.y_test.numpy()
        
        predictions = self.y_scaler.inverse_transform(predictions_scaled)
        targets = self.y_scaler.inverse_transform(targets_scaled)
        
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
        
        # 确保数据是一维数组
        targets = targets.flatten()
        predictions = predictions.flatten()
        
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
        
        plt.title(f'刀具 {self.test_cutter} 磨损预测结果对比图 (LSTM模型)', fontsize=16)
        plt.xlabel('实际磨损值 (mm)', fontsize=14)
        plt.ylabel('预测磨损值 (mm)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # 保存预测对比图
        plt.savefig(os.path.join(self.output_path, f'{self.test_cutter}_scatter.png'), dpi=300)
        plt.close()
        
        # 创建预测序列图
        plt.figure(figsize=(14, 8))
        
        # 确保切削次数也是一维数组
        test_cut_seqs = np.array(self.test_cut_seqs).flatten()
        
        # 绘制预测值和真实值随切削次数的变化
        plt.plot(test_cut_seqs, targets, 'b-', label='实际磨损值', linewidth=2)
        plt.plot(test_cut_seqs, predictions, 'r--', label='预测磨损值', linewidth=2)
        
        # 计算绝对误差并绘制误差区间
        abs_error = np.abs(targets - predictions)
        plt.fill_between(test_cut_seqs, predictions - abs_error, predictions + abs_error, 
                        color='gray', alpha=0.2, label='预测误差区间')
        
        plt.title(f'刀具 {self.test_cutter} 磨损预测随切削次数的变化 (LSTM模型)', fontsize=16)
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
        plt.title('LSTM模型训练历史', fontsize=16)
        plt.xlabel('周期', fontsize=14)
        plt.ylabel('损失', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(os.path.join(self.output_path, 'training_history.png'), dpi=300)
        
        plt.close('all')
        logger.info("可视化完成")
    
    def save_model(self, filename):
        """保存模型"""
        torch.save(self.model.state_dict(), os.path.join(self.output_path, filename))
    
    def load_model(self, filename):
        """加载模型"""
        self.model.load_state_dict(torch.load(os.path.join(self.output_path, filename)))
    
    def run(self):
        """运行完整的训练和评估流程"""
        self.prepare_data()
        self.build_model()
        self.train()
        return self.evaluate()

def main():
    """主函数"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='铣刀寿命预测系统 - LSTM模型')
    
    # 添加命令行参数
    parser.add_argument('--features_path', type=str, default='/Users/xiaohudemac/cursor01/bishe/5_8tool/data/selected_features',
                        help='特征数据路径')
    parser.add_argument('--output_path', type=str, default='/Users/xiaohudemac/cursor01/bishe/5_8tool/results/lstm',
                        help='模型输出路径')
    parser.add_argument('--seq_length', type=int, default=5,
                        help='序列长度')
    parser.add_argument('--train_cutters', type=str, default='c1,c4',
                        help='训练集刀具列表，用逗号分隔')
    parser.add_argument('--test_cutter', type=str, default='c6',
                        help='测试集刀具')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 解析训练刀具列表
    train_cutters = args.train_cutters.split(',')
    
    # 初始化模型训练器
    trainer = ModelTrainer(
        features_path=args.features_path,
        output_path=args.output_path,
        seq_length=args.seq_length,
        train_cutters=train_cutters,
        test_cutter=args.test_cutter
    )
    
    # 运行训练和评估
    trainer.run()
    
    logger.info("模型训练和评估完成！")

if __name__ == "__main__":
    main() 