#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
铣刀寿命预测系统 - 集成模型
功能：结合LSTM、随机森林、XGBoost等多种模型的预测结果，提高铣刀磨损预测精度
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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tqdm import tqdm
import argparse
import logging
import json
import joblib

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置随机种子以便结果可复现
np.random.seed(42)
torch.manual_seed(42)

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

class EnsembleModel:
    """集成模型"""
    
    def __init__(self, features_path, output_path, seq_length=5, train_cutters=['c1', 'c4'], test_cutter='c6'):
        """
        初始化集成模型
        
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
        
        # 初始化模型参数
        self.input_size = None  # 将在加载数据时确定
        self.hidden_size = 64
        self.num_layers = 2
        
        # 初始化模型
        self.lstm_model = None
        self.rf_model = None
        self.xgb_model = None
        self.stacking_model = None
        
        # 训练参数
        self.learning_rate = 0.001
        self.epochs = 1000
        self.early_stop_patience = 100
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
        }
    
    def load_data(self):
        """加载并预处理数据"""
        logger.info("加载数据...")
        
        # 加载训练数据
        all_train_data = []
        for cutter in self.train_cutters:
            data_file = os.path.join(self.features_path, cutter, f"{cutter}_selected_feature_data.csv")
            try:
                data = pd.read_csv(data_file)
                data['cutter'] = cutter  # 添加刀具标识
                all_train_data.append(data)
                logger.info(f"成功加载{cutter}数据: {data_file}")
            except Exception as e:
                logger.error(f"加载{cutter}数据失败: {e}")
                continue
        
        # 合并所有训练数据
        if len(all_train_data) > 0:
            self.train_data = pd.concat(all_train_data, ignore_index=True)
            logger.info(f"合并训练数据，共{len(self.train_data)}条记录")
        else:
            logger.error("没有有效的训练数据")
            return False
        
        # 加载测试数据
        test_data_file = os.path.join(self.features_path, self.test_cutter, f"{self.test_cutter}_selected_feature_data.csv")
        try:
            self.test_data = pd.read_csv(test_data_file)
            self.test_data['cutter'] = self.test_cutter  # 添加刀具标识
            logger.info(f"成功加载测试数据: {test_data_file}")
        except Exception as e:
            logger.error(f"加载测试数据失败: {e}")
            return False
        
        # 找出训练集和测试集共同的特征
        non_feature_cols = ['cut_num', 'cutter', 'wear_VB1', 'wear_VB2', 'wear_VB3', 'wear_VB_avg']
        train_features = [col for col in self.train_data.columns if col not in non_feature_cols]
        test_features = [col for col in self.test_data.columns if col not in non_feature_cols]
        
        # 取交集
        common_features = list(set(train_features).intersection(set(test_features)))
        logger.info(f"找到{len(common_features)}个共同特征")
        
        if len(common_features) == 0:
            logger.error("没有共同特征，无法训练模型")
            return False
        
        # 提取特征和目标变量
        self.train_data = self.train_data.sort_values(['cutter', 'cut_num'])
        self.test_data = self.test_data.sort_values('cut_num')
        
        # 只使用共同特征
        self.X_train = self.train_data[common_features].values
        self.y_train = self.train_data['wear_VB_avg'].values
        self.X_test = self.test_data[common_features].values
        self.y_test = self.test_data['wear_VB_avg'].values
        self.test_cut_nums = self.test_data['cut_num'].values
        
        # 特征标准化
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # 确定输入特征数量
        self.input_size = self.X_train.shape[1]
        
        # 准备LSTM模型的序列数据
        self.prepare_sequence_data()
        
        logger.info(f"数据加载完成，输入特征数量: {self.input_size}")
        return True
    
    def prepare_sequence_data(self):
        """准备LSTM模型的序列数据"""
        logger.info("准备序列数据...")
        
        # 对每个刀具分别准备序列数据
        all_X_seqs = []
        all_y_seqs = []
        
        for cutter in self.train_cutters:
            cutter_data = self.train_data[self.train_data['cutter'] == cutter].sort_values('cut_num')
            if len(cutter_data) <= self.seq_length:
                logger.warning(f"刀具{cutter}的数据不足以创建序列")
                continue
            
            # 使用已经准备好的标准化特征数据
            cutter_indices = self.train_data[self.train_data['cutter'] == cutter].index
            X = self.X_train_scaled[cutter_indices]
            y = self.y_train[cutter_indices]
            
            # 创建序列
            for i in range(len(X) - self.seq_length):
                all_X_seqs.append(X[i:i+self.seq_length])
                all_y_seqs.append(y[i+self.seq_length])
        
        self.X_train_seqs = np.array(all_X_seqs)
        self.y_train_seqs = np.array(all_y_seqs)
        
        # 准备测试序列数据
        test_X_seqs = []
        test_y_seqs = []
        self.test_cut_seqs = []
        
        for i in range(len(self.X_test_scaled) - self.seq_length):
            test_X_seqs.append(self.X_test_scaled[i:i+self.seq_length])
            test_y_seqs.append(self.y_test[i+self.seq_length])
            self.test_cut_seqs.append(self.test_cut_nums[i+self.seq_length])
        
        self.X_test_seqs = np.array(test_X_seqs)
        self.y_test_seqs = np.array(test_y_seqs)
        
        # 转换为PyTorch张量
        self.X_train_tensor = torch.FloatTensor(self.X_train_seqs)
        self.y_train_tensor = torch.FloatTensor(self.y_train_seqs).view(-1, 1)
        self.X_test_tensor = torch.FloatTensor(self.X_test_seqs)
        self.y_test_tensor = torch.FloatTensor(self.y_test_seqs).view(-1, 1)
        
        logger.info(f"训练序列数据形状: {self.X_train_seqs.shape}, {self.y_train_seqs.shape}")
        logger.info(f"测试序列数据形状: {self.X_test_seqs.shape}, {self.y_test_seqs.shape}")
    
    def train_lstm(self):
        """训练LSTM模型"""
        logger.info("训练LSTM模型...")
        
        # 初始化LSTM模型
        self.lstm_model = LSTMModel(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=0.3
        )
        
        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.lstm_model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=50, verbose=True
        )
        
        # 训练
        best_val_loss = float('inf')
        early_stop_counter = 0
        
        for epoch in range(self.epochs):
            # 训练模式
            self.lstm_model.train()
            
            # 前向传播和反向传播
            optimizer.zero_grad()
            outputs = self.lstm_model(self.X_train_tensor)
            loss = criterion(outputs, self.y_train_tensor)
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.lstm_model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss = loss.item()
            
            # 验证模式
            self.lstm_model.eval()
            with torch.no_grad():
                val_outputs = self.lstm_model(self.X_test_tensor)
                val_loss = criterion(val_outputs, self.y_test_tensor).item()
            
            # 更新学习率
            scheduler.step(val_loss)
            
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
                torch.save(self.lstm_model.state_dict(), os.path.join(self.output_path, 'lstm_model.pt'))
            else:
                early_stop_counter += 1
            
            if early_stop_counter >= self.early_stop_patience:
                logger.info(f'提前停止训练，已经{self.early_stop_patience}个周期没有改善')
                break
        
        logger.info(f"LSTM模型训练完成，最佳验证损失: {best_val_loss:.6f}")
        
        # 加载最佳模型
        self.lstm_model.load_state_dict(torch.load(os.path.join(self.output_path, 'lstm_model.pt')))
    
    def train_traditional_models(self):
        """训练传统机器学习模型（随机森林、XGBoost等）"""
        logger.info("训练传统机器学习模型...")
        
        # 随机森林回归
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.rf_model.fit(self.X_train_scaled, self.y_train)
        
        # 保存模型
        joblib.dump(self.rf_model, os.path.join(self.output_path, 'rf_model.pkl'))
        
        rf_train_pred = self.rf_model.predict(self.X_train_scaled)
        rf_test_pred = self.rf_model.predict(self.X_test_scaled)
        rf_train_mse = mean_squared_error(self.y_train, rf_train_pred)
        rf_test_mse = mean_squared_error(self.y_test, rf_test_pred)
        rf_test_r2 = r2_score(self.y_test, rf_test_pred)
        
        logger.info(f"随机森林 - 训练MSE: {rf_train_mse:.6f}, 测试MSE: {rf_test_mse:.6f}, 测试R²: {rf_test_r2:.6f}")
        
        # XGBoost回归
        self.xgb_model = XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        self.xgb_model.fit(self.X_train_scaled, self.y_train)
        
        # 保存模型
        joblib.dump(self.xgb_model, os.path.join(self.output_path, 'xgb_model.pkl'))
        
        xgb_train_pred = self.xgb_model.predict(self.X_train_scaled)
        xgb_test_pred = self.xgb_model.predict(self.X_test_scaled)
        xgb_train_mse = mean_squared_error(self.y_train, xgb_train_pred)
        xgb_test_mse = mean_squared_error(self.y_test, xgb_test_pred)
        xgb_test_r2 = r2_score(self.y_test, xgb_test_pred)
        
        logger.info(f"XGBoost - 训练MSE: {xgb_train_mse:.6f}, 测试MSE: {xgb_test_mse:.6f}, 测试R²: {xgb_test_r2:.6f}")
        
        # Gradient Boosting回归
        self.gb_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.gb_model.fit(self.X_train_scaled, self.y_train)
        
        # 保存模型
        joblib.dump(self.gb_model, os.path.join(self.output_path, 'gb_model.pkl'))
        
        gb_train_pred = self.gb_model.predict(self.X_train_scaled)
        gb_test_pred = self.gb_model.predict(self.X_test_scaled)
        gb_train_mse = mean_squared_error(self.y_train, gb_train_pred)
        gb_test_mse = mean_squared_error(self.y_test, gb_test_pred)
        gb_test_r2 = r2_score(self.y_test, gb_test_pred)
        
        logger.info(f"Gradient Boosting - 训练MSE: {gb_train_mse:.6f}, 测试MSE: {gb_test_mse:.6f}, 测试R²: {gb_test_r2:.6f}")
    
    def train_stacking_model(self):
        """训练Stacking集成模型"""
        logger.info("训练Stacking集成模型...")
        
        # 使用LSTM、随机森林和XGBoost的预测结果作为新特征
        self.lstm_model.eval()
        with torch.no_grad():
            lstm_train_pred = self.lstm_model(self.X_train_tensor).numpy().flatten()
            lstm_test_pred = self.lstm_model(self.X_test_tensor).numpy().flatten()
        
        # 注意：LSTM的序列长度可能导致预测结果数量与原始数据不一致
        # 我们只使用有对应LSTM预测的样本
        train_indices = range(self.seq_length, len(self.X_train))
        lstm_train_indices = range(len(lstm_train_pred))
        
        # 确保索引不超过数组长度
        valid_indices = [i for i, j in zip(train_indices, lstm_train_indices) if i < len(self.X_train)]
        
        # 如果没有有效的样本，则不训练Stacking模型
        if len(valid_indices) == 0:
            logger.warning("没有足够的样本来训练Stacking模型")
            return
        
        # 创建用于Stacking的特征集
        rf_train_pred = self.rf_model.predict(self.X_train_scaled)
        xgb_train_pred = self.xgb_model.predict(self.X_train_scaled)
        gb_train_pred = self.gb_model.predict(self.X_train_scaled)
        
        # 准备Stacking训练数据
        X_stack_train = np.column_stack([
            rf_train_pred[valid_indices],
            xgb_train_pred[valid_indices],
            gb_train_pred[valid_indices],
            self.X_train_scaled[valid_indices]
        ])
        y_stack_train = self.y_train[valid_indices]
        
        # 训练Stacking模型
        self.stacking_model = Ridge(alpha=1.0)
        self.stacking_model.fit(X_stack_train, y_stack_train)
        
        # 保存模型
        joblib.dump(self.stacking_model, os.path.join(self.output_path, 'stacking_model.pkl'))
        
        logger.info("Stacking模型训练完成")
    
    def predict(self):
        """使用集成模型进行预测"""
        logger.info("使用集成模型进行预测...")
        
        # LSTM预测
        self.lstm_model.eval()
        with torch.no_grad():
            lstm_pred = self.lstm_model(self.X_test_tensor).numpy().flatten()
        
        # 传统模型预测
        rf_pred = self.rf_model.predict(self.X_test_scaled[self.seq_length:])
        xgb_pred = self.xgb_model.predict(self.X_test_scaled[self.seq_length:])
        gb_pred = self.gb_model.predict(self.X_test_scaled[self.seq_length:])
        
        # Stacking模型预测
        X_stack_test = np.column_stack([
            rf_pred,
            xgb_pred,
            gb_pred,
            self.X_test_scaled[self.seq_length:]
        ])
        stack_pred = self.stacking_model.predict(X_stack_test)
        
        # 各模型在测试集上的性能
        y_test_seq = self.y_test[self.seq_length:]
        
        # 评估各个模型的性能
        models = {
            'RF': rf_pred,
            'XGBoost': xgb_pred,
            'GradientBoosting': gb_pred,
            'LSTM': lstm_pred,
            'Stacking': stack_pred
        }
        
        results = {}
        best_model = None
        best_r2 = -float('inf')
        
        for name, pred in models.items():
            mse = mean_squared_error(y_test_seq, pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test_seq, pred)
            r2 = r2_score(y_test_seq, pred)
            
            results[name] = {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2)
            }
            
            logger.info(f"{name} - MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}")
            
            if r2 > best_r2:
                best_r2 = r2
                best_model = name
        
        results['best_model'] = best_model
        
        # 保存评估结果
        with open(os.path.join(self.output_path, 'evaluation.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        # 可视化预测结果
        self.visualize_predictions(y_test_seq, models, self.test_cut_seqs)
        
        logger.info(f"最佳模型是: {best_model}")
        return results
    
    def visualize_predictions(self, y_true, model_predictions, cut_nums):
        """可视化各个模型的预测结果"""
        logger.info("可视化预测结果...")
        
        # 创建预测与实际值对比图
        plt.figure(figsize=(14, 8))
        plt.plot(cut_nums, y_true, 'k-', linewidth=2, label='实际值')
        
        colors = ['b', 'g', 'r', 'c', 'm']
        for i, (name, pred) in enumerate(model_predictions.items()):
            plt.plot(cut_nums, pred, colors[i % len(colors)] + '--', label=f'{name}预测值')
        
        plt.title(f'{self.test_cutter} 刀具磨损预测结果比较')
        plt.xlabel('切削次数')
        plt.ylabel('磨损值 (mm)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(os.path.join(self.output_path, f'{self.test_cutter}_predictions_comparison.png'))
        
        # 为每个模型创建散点图
        for name, pred in model_predictions.items():
            plt.figure(figsize=(8, 8))
            plt.scatter(y_true, pred, alpha=0.7)
            plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
            plt.title(f'{self.test_cutter} 实际值 vs {name}预测值')
            plt.xlabel('实际值')
            plt.ylabel('预测值')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.axis('equal')
            plt.tight_layout()
            
            # 保存图像
            plt.savefig(os.path.join(self.output_path, f'{self.test_cutter}_{name}_scatter.png'))
        
        # 创建训练历史图
        if len(self.history['train_loss']) > 0:
            plt.figure(figsize=(12, 6))
            plt.plot(self.history['train_loss'], label='训练损失')
            plt.plot(self.history['val_loss'], label='验证损失')
            plt.title('LSTM训练历史')
            plt.xlabel('周期')
            plt.ylabel('损失')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            
            # 保存图像
            plt.savefig(os.path.join(self.output_path, 'lstm_training_history.png'))
        
        logger.info("可视化完成")
    
    def run(self):
        """运行完整的训练和评估流程"""
        # 加载数据
        if not self.load_data():
            return None
        
        # 训练各个模型
        self.train_lstm()
        self.train_traditional_models()
        self.train_stacking_model()
        
        # 预测并评估
        results = self.predict()
        
        logger.info("集成模型训练和评估完成！")
        return results

def main():
    """主函数"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='铣刀寿命预测系统 - 集成模型')
    
    # 添加命令行参数
    parser.add_argument('--features_path', type=str, default='/Users/xiaohudemac/cursor01/bishe/5_8tool/data/selected_features',
                        help='特征数据路径')
    parser.add_argument('--output_path', type=str, default='/Users/xiaohudemac/cursor01/bishe/5_8tool/results/ensemble',
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
    
    # 初始化集成模型
    ensemble = EnsembleModel(
        features_path=args.features_path,
        output_path=args.output_path,
        seq_length=args.seq_length,
        train_cutters=train_cutters,
        test_cutter=args.test_cutter
    )
    
    # 运行训练和评估
    ensemble.run()

if __name__ == "__main__":
    main() 