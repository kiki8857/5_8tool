# -*- coding: utf-8 -*-

"""
铣刀寿命预测系统 - LSTM超参数调优
功能：网格搜索寻找LSTM模型的最佳超参数组合
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
import json
from datetime import datetime
import time
import itertools
from tqdm import tqdm
from matplotlib import font_manager
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

# 设置CUDA的并行计算能力
torch.backends.cudnn.benchmark = True

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

class LSTMHyperparameterTuner:
    """LSTM超参数调优器"""
    
    def __init__(self, features_path, output_path, train_cutters=['c1', 'c4'], test_cutter='c6', num_workers=4):
        """
        初始化LSTM超参数调优器
        
        参数:
            features_path: 特征数据路径
            output_path: 输出路径
            train_cutters: 训练集刀具列表
            test_cutter: 测试集刀具
            num_workers: 数据加载器使用的工作进程数
        """
        self.features_path = features_path
        self.output_path = output_path
        self.train_cutters = train_cutters
        self.test_cutter = test_cutter
        self.num_workers = num_workers
        
        # 确保输出目录存在
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 超参数搜索空间
        self.param_grid = {
            'seq_length': [5],  # 只使用序列长度5进行测试
            'hidden_size': [32, 64, 128],
            'num_layers': [1, 2],
            'dropout': [0.2, 0.3],
            'learning_rate': [0.001, 0.0001],
            'batch_size': [16, 32],
            'weight_decay': [0, 1e-5]
        }
        
        # 训练参数
        self.epochs = 300  # 每个超参数组合的最大训练轮数
        self.patience = 50  # 提前停止的耐心值
    
    def prepare_data(self, seq_length):
        """
        准备训练和测试数据
        
        参数:
            seq_length: 序列长度
        """
        logger.info(f"准备数据，序列长度: {seq_length}...")
        
        # 加载所有训练数据集的所有数据，合并后创建序列
        all_X_train = []
        all_y_train = []
        
        for cutter in self.train_cutters:
            # 加载数据
            data_file = os.path.join(self.features_path, cutter, f"{cutter}_selected_feature_data.csv")
            data = pd.read_csv(data_file)
            
            # 按切削次数排序
            data = data.sort_values('cut_num')
            
            # 分离特征和目标变量
            X = data.drop(['cut_num', 'wear_VB_avg'], axis=1).values
            y = data['wear_VB_avg'].values
            
            all_X_train.append(X)
            all_y_train.append(y)
            
            # 获取输入特征数量
            if not hasattr(self, 'input_size'):
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
            
            for j in range(len(X_scaled) - seq_length):
                X_seqs.append(X_scaled[j:j+seq_length])
                y_seqs.append(y_scaled[j+seq_length])
            
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
            return False
        
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
        
        for i in range(len(self.X_test_scaled) - seq_length):
            X_test_seqs.append(self.X_test_scaled[i:i+seq_length])
            y_test_seqs.append(self.y_test_scaled[i+seq_length])
            self.test_cut_seqs.append(self.test_cut_nums[i+seq_length])
        
        self.X_test = torch.FloatTensor(X_test_seqs)
        self.y_test = torch.FloatTensor(y_test_seqs).unsqueeze(1)
        
        logger.info(f"测试数据形状: {self.X_test.shape}, {self.y_test.shape}")
        logger.info(f"数据准备完成，输入特征数量: {self.input_size}")
        
        return True
    
    def train_eval_model(self, hidden_size, num_layers, dropout, batch_size, learning_rate, weight_decay):
        """
        训练和评估指定超参数的模型
        
        参数:
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
            dropout: Dropout率
            batch_size: 批处理大小
            learning_rate: 学习率
            weight_decay: 权重衰减
            
        返回:
            metrics: 评估指标
        """
        # 初始化模型
        model = LSTMModel(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=1,
            dropout=dropout
        ).to(self.device)
        
        # 数据加载器 - 增加num_workers和pin_memory提高数据加载速度
        train_loader = DataLoader(
            torch.utils.data.TensorDataset(self.X_train, self.y_train),
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=20, verbose=False
        )
        
        # 训练历史
        history = {
            'train_loss': [],
            'val_loss': [],
        }
        
        # 用于提前停止的变量
        best_val_loss = float('inf')
        early_stop_counter = 0
        
        # 将测试数据预先移至GPU
        X_test_gpu = self.X_test.to(self.device)
        y_test_gpu = self.y_test.to(self.device)
        
        # 训练循环
        for epoch in range(self.epochs):
            # 训练模式
            model.train()
            train_loss = 0.0
            
            # 训练一个周期
            for inputs, targets in train_loader:
                # 将数据移至设备
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # 反向传播和优化
                optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            # 计算平均训练损失
            train_loss /= len(train_loader)
            
            # 验证模式
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test_gpu)
                val_loss = criterion(val_outputs, y_test_gpu).item()
            
            # 更新学习率
            scheduler.step(val_loss)
            
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
        
        # 加载最佳模型
        model.load_state_dict(best_model_state)
        
        # 最终评估
        model.eval()
        with torch.no_grad():
            predictions_scaled = model(X_test_gpu).cpu().numpy()
            y_test = self.y_test.numpy()
        
        # 反标准化
        predictions = self.y_scaler.inverse_transform(predictions_scaled)
        targets = self.y_scaler.inverse_transform(y_test)
        
        # 计算评估指标
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        # 记录评估结果
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'best_val_loss': float(best_val_loss),
            'epochs_trained': epoch + 1
        }
        
        return metrics, history, model, best_model_state
    
    def _evaluate_param_combination(self, params):
        """
        评估单个参数组合
        
        参数:
            params: 参数组合(hidden_size, num_layers, dropout, batch_size, learning_rate, weight_decay)
            
        返回:
            result: 评估结果
        """
        hidden_size, num_layers, dropout, batch_size, learning_rate, weight_decay = params
        
        # 记录当前参数
        current_params = {
            'seq_length': self.current_seq_length,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay
        }
        
        param_start_time = time.time()
        logger.info(f"评估参数组合: {current_params}")
        
        try:
            # 训练和评估模型
            metrics, history, model, model_state = self.train_eval_model(
                hidden_size, num_layers, dropout, batch_size, learning_rate, weight_decay
            )
            
            # 记录结果
            result = {
                'params': current_params,
                'metrics': metrics,
                'history': {
                    'train_loss': history['train_loss'],
                    'val_loss': history['val_loss']
                },
                'model_state': model_state
            }
            
            param_time = time.time() - param_start_time
            logger.info(f"参数组合评估完成 - R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, 耗时: {param_time:.2f}秒")
            
            return result
        except Exception as e:
            logger.error(f"评估参数组合时出错: {e}")
            return None
    
    def tune_hyperparameters(self):
        """
        进行超参数调优
        
        返回:
            best_params: 最佳超参数
            all_results: 所有参数组合的结果
        """
        logger.info("开始超参数调优...")
        start_time = time.time()
        
        best_params = None
        best_r2 = -float('inf')
        best_model_state = None
        all_results = []
        
        # 为不同的序列长度准备数据
        seq_lengths = self.param_grid['seq_length']
        
        for seq_length in seq_lengths:
            # 准备数据
            data_prepared = self.prepare_data(seq_length)
            if not data_prepared:
                logger.warning(f"序列长度 {seq_length} 的数据准备失败，跳过")
                continue
            
            # 保存当前序列长度
            self.current_seq_length = seq_length
            
            # 生成参数组合（除了seq_length）
            param_combinations = list(itertools.product(
                self.param_grid['hidden_size'],
                self.param_grid['num_layers'],
                self.param_grid['dropout'],
                self.param_grid['batch_size'],
                self.param_grid['learning_rate'],
                self.param_grid['weight_decay']
            ))
            
            total_combinations = len(param_combinations)
            logger.info(f"序列长度 {seq_length}，将评估 {total_combinations} 个超参数组合")
            
            # 使用tqdm显示进度条
            for i, params in enumerate(tqdm(param_combinations, desc="超参数搜索进度")):
                result = self._evaluate_param_combination(params)
                
                if result:
                    all_results.append(result)
                    
                    # 更新最佳参数
                    if result['metrics']['r2'] > best_r2:
                        best_r2 = result['metrics']['r2']
                        best_params = result['params']
                        best_model_state = result['model_state']
                        
                        # 保存最佳模型
                        torch.save(best_model_state, os.path.join(self.output_path, 'best_model.pt'))
                        
                        logger.info(f"发现新的最佳参数组合，R²: {best_r2:.4f}")
        
        end_time = time.time()
        tuning_time = end_time - start_time
        
        logger.info(f"超参数调优完成，总耗时: {tuning_time:.2f}秒")
        logger.info(f"最佳超参数: {best_params}")
        logger.info(f"最佳R²: {best_r2:.6f}")
        
        # 保存最佳参数
        with open(os.path.join(self.output_path, 'best_params.json'), 'w') as f:
            json.dump(best_params, f, indent=4)
        
        # 保存所有结果
        with open(os.path.join(self.output_path, 'all_results.json'), 'w') as f:
            # 将numpy数组转换为列表以便JSON序列化
            serializable_results = []
            for result in all_results:
                serializable_result = {
                    'params': result['params'],
                    'metrics': result['metrics'],
                    'history': {
                        'train_loss': [float(x) for x in result['history']['train_loss']],
                        'val_loss': [float(x) for x in result['history']['val_loss']]
                    }
                }
                serializable_results.append(serializable_result)
            
            json.dump(serializable_results, f, indent=4)
        
        return best_params, all_results, best_model_state
    
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
                'seq_length': result['params']['seq_length'],
                'hidden_size': result['params']['hidden_size'],
                'num_layers': result['params']['num_layers'],
                'dropout': result['params']['dropout'],
                'batch_size': result['params']['batch_size'],
                'learning_rate': result['params']['learning_rate'],
                'weight_decay': result['params']['weight_decay'],
                'epochs_trained': result['metrics']['epochs_trained']
            }
            results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
        
        # 保存结果数据框
        results_df.to_csv(os.path.join(self.output_path, 'all_param_results.csv'), index=False)
        
        # 绘制超参数与R²的关系
        params_to_plot = ['seq_length', 'hidden_size', 'num_layers', 'dropout', 
                         'batch_size', 'learning_rate', 'weight_decay']
        
        for param in params_to_plot:
            self._plot_param_vs_r2(param, results_df)
        
        # 绘制最佳参数组合的训练历史
        best_result = None
        for result in all_results:
            if all(result['params'][key] == best_params[key] for key in best_params.keys()):
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
        
        # 分组统计
        gb = results_df.groupby(param_name)['r2'].agg(['mean', 'std', 'count'])
        
        # 绘制条形图
        x = gb.index
        y = gb['mean']
        yerr = gb['std']
        
        plt.bar(range(len(x)), y, yerr=yerr, alpha=0.7)
        plt.xticks(range(len(x)), x)
        
        plt.title(f'{param_name}参数对R²的影响')
        plt.xlabel(param_name)
        plt.ylabel('平均R²')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.output_path, f'param_{param_name}_vs_r2.png'))
        plt.close()
    
    def evaluate_best_model(self, best_params):
        """
        在测试集上评估最佳模型并可视化结果
        
        参数:
            best_params: 最佳超参数
        """
        logger.info("评估最佳模型...")
        
        # 准备数据
        self.prepare_data(best_params['seq_length'])
        
        # 初始化最佳模型
        best_model = LSTMModel(
            input_size=self.input_size,
            hidden_size=best_params['hidden_size'],
            num_layers=best_params['num_layers'],
            output_size=1,
            dropout=best_params['dropout']
        ).to(self.device)
        
        # 加载模型权重
        best_model.load_state_dict(torch.load(os.path.join(self.output_path, 'best_model.pt')))
        
        # 进入评估模式
        best_model.eval()
        
        # 预测
        with torch.no_grad():
            X_test = self.X_test.to(self.device)
            predictions_scaled = best_model(X_test).cpu().numpy()
        
        # 反标准化
        targets = self.y_scaler.inverse_transform(self.y_test.numpy())
        predictions = self.y_scaler.inverse_transform(predictions_scaled)
        
        # 将结果展平
        targets = targets.flatten()
        predictions = predictions.flatten()
        
        # 计算评估指标
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
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
        
        # 创建预测序列图
        plt.figure(figsize=(12, 6))
        plt.plot(self.test_cut_seqs, targets, 'b-', label='实际值')
        plt.plot(self.test_cut_seqs, predictions, 'r--', label='预测值')
        plt.title(f'{self.test_cutter} 刀具磨损预测结果 (LSTM)')
        plt.xlabel('切削次数')
        plt.ylabel('磨损值 (mm)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, f'{self.test_cutter}_predictions.png'))
        
        # 创建散点图
        plt.figure(figsize=(8, 8))
        plt.scatter(targets, predictions, alpha=0.7)
        plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
        plt.title(f'{self.test_cutter} 实际值 vs 预测值 (LSTM)')
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
        best_params, all_results, _ = self.tune_hyperparameters()
        
        # 可视化结果
        self.visualize_results(all_results, best_params)
        
        # 评估最佳模型
        best_metrics = self.evaluate_best_model(best_params)
        
        logger.info("超参数调优流程完成！")
        
        return best_params, best_metrics

def main():
    """主函数"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='铣刀寿命预测系统 - LSTM超参数调优')
    
    # 添加命令行参数
    parser.add_argument('--features_path', type=str, default='/Users/xiaohudemac/cursor01/bishe/5_8tool/data/selected_features',
                       help='特征数据路径')
    parser.add_argument('--output_path', type=str, default='/Users/xiaohudemac/cursor01/bishe/5_8tool/results/lstm_tuning',
                       help='输出根路径')
    parser.add_argument('--train_cutters', type=str, default='c1,c4',
                       help='训练集刀具列表，用逗号分隔')
    parser.add_argument('--test_cutter', type=str, default='c6',
                       help='测试集刀具')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载器使用的工作进程数')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 解析训练刀具列表
    train_cutters = args.train_cutters.split(',')
    
    # 为输出路径添加时间戳，确保每次运行结果保存在单独的文件夹中
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_path, f"run_{timestamp}")
    
    # 初始化LSTM超参数调优器
    tuner = LSTMHyperparameterTuner(
        features_path=args.features_path,
        output_path=output_path,
        train_cutters=train_cutters,
        test_cutter=args.test_cutter,
        num_workers=args.num_workers
    )
    
    # 运行超参数调优
    tuner.run()

if __name__ == "__main__":
    main()