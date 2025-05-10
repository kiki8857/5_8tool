#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
铣刀寿命预测系统 - 卷积神经网络模型
功能：使用CNN算法预测刀具磨损（PyTorch实现）
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json
import logging
import argparse
from datetime import datetime
import joblib

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


# 添加注意力机制模块
class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.scale = torch.sqrt(torch.FloatTensor([input_dim])).to(device)
        
    def forward(self, x):
        # x: [batch_size, seq_len, hidden_dim]
        batch_size, seq_len, hidden_dim = x.size()
        
        # 计算查询、键、值
        q = self.query(x)  # [batch_size, seq_len, hidden_dim]
        k = self.key(x)    # [batch_size, seq_len, hidden_dim]
        v = self.value(x)  # [batch_size, seq_len, hidden_dim]
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(1, 2)) / self.scale  # [batch_size, seq_len, seq_len]
        
        # 应用softmax
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [batch_size, seq_len, seq_len]
        
        # 加权和
        output = torch.matmul(attn_weights, v)  # [batch_size, seq_len, hidden_dim]
        
        return output, attn_weights

# 定义CNN模型
class CNNNet(nn.Module):
    def __init__(self, params, input_channels, seq_length, num_features):
        """
        初始化CNN网络
        
        参数:
            params: 包含网络结构超参数的字典
            input_channels: 输入通道数，通常为1
            seq_length: 序列长度，即窗口大小
            num_features: 特征数量
        """
        super(CNNNet, self).__init__()
        
        # 从params中提取超参数
        conv_layers = params.get('conv_layers', 3) # 默认值参考调优结果
        first_conv_out = params.get('first_conv_out', 32)
        kernel_size = params.get('kernel_size', 7)
        dropout_rate1 = params.get('dropout_rate1', 0.1739)
        dropout_rate2 = params.get('dropout_rate2', 0.3909)
        fc_units1 = params.get('fc_units1', 256)
        fc_units2 = params.get('fc_units2', 128)
        use_attention = params.get('use_attention', True)
        attention_weight = params.get('attention_weight', 0.3928)
        
        # 计算每层的输出通道数
        channels_list = [first_conv_out]
        for i in range(1, conv_layers):
            if i <= 2: # 示例逻辑，可以根据调优结果调整
                channels_list.append(channels_list[-1] * 2 if channels_list[-1] * 2 <= 512 else 512) 
            else:
                channels_list.append(channels_list[-1] // 2 if channels_list[-1] // 2 >= 32 else 32)

        
        # 构建卷积层
        self.conv_blocks = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        
        # 第一个卷积层
        self.conv_blocks.append(nn.Sequential(
            nn.Conv1d(input_channels, channels_list[0], kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(channels_list[0]),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(kernel_size=2, stride=2)
        ))
        self.shortcuts.append(nn.Conv1d(input_channels, channels_list[0], kernel_size=1))
        
        # 剩余卷积层
        current_in_channels = channels_list[0]
        for i in range(1, conv_layers):
            out_channels = channels_list[i]
            
            # 确保池化不会导致维度过小
            pool_stride = 2 if (seq_length * num_features) // (2**(i)) > 1 else 1
            
            if i < conv_layers - 1:
                self.conv_blocks.append(nn.Sequential(
                    nn.Conv1d(current_in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
                    nn.BatchNorm1d(out_channels),
                    nn.LeakyReLU(0.1),
                    nn.MaxPool1d(kernel_size=2, stride=pool_stride)
                ))
            else:
                # 最后一层不使用池化
                self.conv_blocks.append(nn.Sequential(
                    nn.Conv1d(current_in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
                    nn.BatchNorm1d(out_channels),
                    nn.LeakyReLU(0.1)
                ))
            
            self.shortcuts.append(nn.Conv1d(current_in_channels, out_channels, kernel_size=1))
            current_in_channels = out_channels
            
        # 计算展平后的特征维度
        # 模拟卷积和池化操作来确定最终的特征长度
        sim_len = seq_length * num_features
        for i in range(conv_layers -1): # 最后一层卷积后没有池化
            sim_len = sim_len // 2 # MaxPool1d kernel_size=2, stride=2
            if sim_len == 0: # 防止维度变为0
                sim_len = 1
        
        flattened_size = channels_list[-1] * sim_len
        
        # 注意力机制
        self.use_attention = use_attention
        self.attention_weight = attention_weight
        if use_attention:
            self.attention = SelfAttention(channels_list[-1])
        
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
        """
        前向传播
        
        参数:
            x: 输入数据，形状为 (batch_size, 1, seq_length, num_features)
            
        返回:
            输出，形状为 (batch_size, 1)
        """
        # 调整输入形状为卷积1D所需形状: (batch_size, channels, seq_length * features)
        batch_size, input_channels, seq_length, num_features = x.shape
        x = x.view(batch_size, input_channels, seq_length * num_features)
        
        # 应用卷积层和残差连接
        for i, (conv_block, shortcut_conv) in enumerate(zip(self.conv_blocks, self.shortcuts)):
            identity = x
            x = conv_block(x)
            
            shortcut_out = shortcut_conv(identity)
            # 对shortcut应用与主路径相同的池化
            if i < len(self.conv_blocks) - 1 : # 如果不是最后一个卷积块
                # 模拟主路径中的池化
                pool_stride = 2 if identity.size(2) // 2 > 0 else 1
                shortcut_out = nn.functional.max_pool1d(shortcut_out, kernel_size=2, stride=pool_stride)

            # 确保维度匹配
            if shortcut_out.size() != x.size():
                 # 如果因为池化导致维度不匹配（例如输入长度为奇数），则调整 x
                target_len = min(shortcut_out.size(2), x.size(2))
                x = x[:, :, :target_len]
                shortcut_out = shortcut_out[:, :, :target_len]

            x = x + shortcut_out
        
        # 应用注意力机制
        if self.use_attention:
            # 获取当前特征图的形状
            batch_size, channels, feature_len = x.shape
            
            # 简化：直接将特征图视为一个序列，每个位置有channels个特征
            x_reshaped = x.permute(0, 2, 1)  # [batch_size, feature_len, channels]
            
            # 应用自注意力
            x_att, _ = self.attention(x_reshaped)
            
            # 将注意力结果变回原始形状
            x_att = x_att.permute(0, 2, 1)  # [batch_size, channels, feature_len]
            
            # 融合注意力输出与原始输出
            x = x + self.attention_weight * x_att
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.fc(x)
        
        return x

class CNNModelTrainer:
    def __init__(self, selected_features_path, output_path, target_column='wear_VB_avg',
                 train_cutters=None, test_cutter='c6', random_state=RANDOM_SEED,
                 model_params=None, train_params=None): # 添加model_params和train_params
        """
        初始化CNN模型
        
        参数:
            selected_features_path: 特征选择后的数据路径
            output_path: 模型输出路径
            target_column: 目标变量列名
            train_cutters: 用于训练的刀具列表，默认为None（使用非测试刀具）
            test_cutter: 用于测试的刀具
            random_state: 随机种子
            model_params: CNNNet模型的超参数字典
            train_params: 训练过程的超参数字典
        """
        self.selected_features_path = selected_features_path
        self.output_path = output_path
        self.target_column = target_column
        self.test_cutter = test_cutter
        self.random_state = random_state
        self.device = device
        
        # 模型和训练参数 - 使用调优结果或默认值
        self.model_params = model_params if model_params else {
            'conv_layers': 3, 'first_conv_out': 32, 'kernel_size': 7,
            'dropout_rate1': 0.1739, 'dropout_rate2': 0.3909,
            'fc_units1': 256, 'fc_units2': 128,
            'use_attention': True, 'attention_weight': 0.3928
        }
        self.train_params = train_params if train_params else {
            'window_size': 19, 'stride': 1, 'batch_size': 32,
            'optimizer': 'Adam', 'learning_rate': 0.000157,
            'weight_decay': 0.000113, 'scheduler': 'OneCycleLR',
            'max_lr_factor': 5.96, 'epochs': 200 # 增加epochs
        }

        # 设置随机种子
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        # 如果未指定训练刀具，则使用除测试刀具外的所有刀具
        if train_cutters is None:
            all_cutters = ['c1', 'c4', 'c6']
            self.train_cutters = [c for c in all_cutters if c != test_cutter]
        else:
            self.train_cutters = train_cutters
        
        # 确保输出目录存在
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # 模型
        self.model = None
        self.scaler = StandardScaler()
    
    def load_data(self):
        """
        加载训练和测试数据
        
        返回:
            X_train_raw: 训练特征原始数据
            y_train_raw: 训练目标原始数据
            X_test_raw: 测试特征原始数据
            y_test_raw: 测试目标原始数据
            test_cut_nums: 测试数据的切削次数
        """
        logger.info("准备数据...")
        
        # 加载训练数据
        train_data = []
        for cutter in self.train_cutters:
            data_file = os.path.join(self.selected_features_path, cutter, f"{cutter}_selected_feature_data.csv")
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
        test_data_file = os.path.join(self.selected_features_path, self.test_cutter, 
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
        
        logger.info(f"数据准备完成，输入特征数量: {X_train_raw.shape[1]}")
        logger.info(f"训练数据形状: {X_train_raw.shape}, 测试数据形状: {X_test_raw.shape}")
        logger.info(f"使用的特征: {', '.join(feature_cols)}")
        
        return X_train_raw, y_train_raw, X_test_raw, y_test_raw, test_cut_nums
    
    def create_sequences(self, X, y, window_size, stride=1):
        """
        创建时序序列数据
        
        参数:
            X: 特征数据，形状为 (samples, features)
            y: 目标数据，形状为 (samples,)
            window_size: 窗口大小
            stride: 步长
            
        返回:
            X_seq: 序列特征数据，形状为 (sequences, 1, window_size, features)
            y_seq: 序列目标数据，形状为 (sequences,)
        """
        sequences_X = []
        sequences_y = []
        
        for i in range(0, len(X) - window_size + 1, stride):
            # 添加一个通道维度
            sequences_X.append(X[i:i+window_size].reshape(1, window_size, -1))
            sequences_y.append(y[i+window_size-1])  # 使用窗口最后一个时间点的目标值
        
        return np.array(sequences_X), np.array(sequences_y)
    
    def preprocess_data(self, X_train_raw, y_train_raw, X_test_raw, y_test_raw):
        """
        预处理数据：标准化特征并创建时序序列
        
        参数:
            X_train_raw: 训练特征原始数据
            y_train_raw: 训练目标原始数据
            X_test_raw: 测试特征原始数据
            y_test_raw: 测试目标原始数据
            
        返回:
            train_loader: 训练数据加载器
            X_test_tensor: 测试特征张量
            y_test_tensor: 测试目标张量
        """
        logger.info("预处理数据...")
        
        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train_raw)
        X_test_scaled = self.scaler.transform(X_test_raw)
        
        # 创建时序序列
        window_size = self.train_params['window_size']
        stride = self.train_params['stride']

        X_train_seq, y_train_seq = self.create_sequences(
            X_train_scaled, y_train_raw, window_size, stride
        )
        X_test_seq, y_test_seq = self.create_sequences(
            X_test_scaled, y_test_raw, window_size, stride
        )
        
        # 打印序列数据形状
        logger.info(f"序列训练数据形状: {X_train_seq.shape}, 序列测试数据形状: {X_test_seq.shape}")
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train_seq).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train_seq).reshape(-1, 1).to(self.device)
        
        X_test_tensor = torch.FloatTensor(X_test_seq).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test_seq).reshape(-1, 1).to(self.device)
        
        # 创建数据集和数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.train_params['batch_size'], shuffle=True)
        
        return train_loader, X_test_tensor, y_test_tensor
    
    def build_model(self, num_features):
        """
        构建CNN模型
        
        参数:
            num_features: 特征数量
            
        返回:
            model: 构建的CNN模型
        """
        logger.info("构建CNN模型...")
        
        model = CNNNet(self.model_params, input_channels=1, 
                       seq_length=self.train_params['window_size'], 
                       num_features=num_features)
        model = model.to(self.device)
        
        # 打印模型结构
        logger.info(f"模型结构: {model}")
        
        return model
    
    def train(self, train_loader):
        """
        训练CNN模型
        
        参数:
            train_loader: 训练数据加载器
            
        返回:
            history: 训练历史记录
        """
        logger.info("开始训练模型...")
        
        # 损失函数
        criterion = nn.MSELoss()

        # 优化器
        optimizer_name = self.train_params.get('optimizer', 'Adam')
        lr = self.train_params.get('learning_rate', 0.000157)
        weight_decay = self.train_params.get('weight_decay', 0.000113)

        if optimizer_name.lower() == 'adamw':
            optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else: # Default to Adam
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # 学习率调度器
        scheduler_name = self.train_params.get('scheduler', 'OneCycleLR')
        epochs = self.train_params.get('epochs', 200)
        max_lr_factor = self.train_params.get('max_lr_factor', 5.96)

        if scheduler_name.lower() == 'onecyclelr':
            steps_per_epoch = len(train_loader)
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, 
                max_lr=lr * max_lr_factor, 
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.3,
                div_factor=25.0, # Default from Optuna trial, can be tuned
                final_div_factor=10000.0 # Default from Optuna trial, can be tuned
            )
        elif scheduler_name.lower() == 'cosineannealinglr':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/100)
        else: # Default or fallback
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, min_lr=1e-6)

        # 训练历史记录
        history = {
            'train_loss': [],
            'valid_loss': [],
            'lr': []
        }
        
        # 提前停止参数
        best_loss = float('inf')
        patience = 40  # 从之前的代码中获取的耐心值
        patience_counter = 0
        best_model_state = None
        
        # 划分训练集和验证集 (如果需要验证集)
        # 在这里，我们直接使用完整的train_loader进行训练，并用测试集评估，因为目标是使用最佳参数训练最终模型
        # 如果需要训练时验证，需要从train_loader中再划分出验证集
        
        # 训练模型
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            epoch_train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                # 前向传播
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) # 梯度裁剪
                optimizer.step()

                if scheduler_name.lower() == 'onecyclelr':
                    scheduler.step() # OneCycleLR在每个batch后更新
                
                epoch_train_loss += loss.item() * X_batch.size(0)
            
            epoch_train_loss /= len(train_loader.dataset)
            history['train_loss'].append(epoch_train_loss)
            
            # 验证阶段 (可选，如果需要基于验证集提前停止)
            # 这里我们简化为只在训练结束时评估测试集
            # 如果需要运行时验证，请取消注释并调整下面的代码
            # self.model.eval()
            # epoch_valid_loss = 0.0
            # with torch.no_grad():
            #     for X_batch_val, y_batch_val in valid_loader: # 假设有valid_loader
            #         outputs_val = self.model(X_batch_val)
            #         loss_val = criterion(outputs_val, y_batch_val)
            #         epoch_valid_loss += loss_val.item() * X_batch_val.size(0)
            # epoch_valid_loss /= len(valid_loader.dataset)
            # history['valid_loss'].append(epoch_valid_loss)
            
            current_lr = optimizer.param_groups[0]['lr']
            history['lr'].append(current_lr)
            
            # 打印训练进度
            if (epoch + 1) % 10 == 0 or epoch == 0:
                # log_msg = f'Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_train_loss:.6f}, LR: {current_lr:.6f}'
                # if 'valid_loss' in history and history['valid_loss']:
                #     log_msg += f', Valid Loss: {history["valid_loss"][-1]:.6f}'
                # logger.info(log_msg)
                logger.info(f'Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_train_loss:.6f}, LR: {current_lr:.6f}')


            # 学习率调度器更新 (对于非OneCycleLR)
            if scheduler_name.lower() != 'onecyclelr':
                if scheduler_name.lower() == 'cosineannealinglr':
                    scheduler.step()
                # elif scheduler_name.lower() == 'reducelronplateau' and history['valid_loss']:
                #     scheduler.step(history['valid_loss'][-1]) # 需要验证损失
            
            # 提前停止判断 (如果使用验证集)
            # if history['valid_loss'] and history['valid_loss'][-1] < best_loss:
            #     best_loss = history['valid_loss'][-1]
            #     patience_counter = 0
            #     best_model_state = self.model.state_dict().copy()
            # elif history['valid_loss']:
            #     patience_counter += 1
            #     if patience_counter >= patience:
            #         logger.info(f'提前停止训练，已经{patience}个轮次没有改善')
            #         break
        
        # # 加载最佳模型 (如果使用验证集提前停止)
        # if best_model_state is not None:
        #     self.model.load_state_dict(best_model_state)
        
        # 保存最终模型 (即使没有提前停止，也保存最后一个epoch的模型)
        model_path = os.path.join(self.output_path, 'final_cnn_model.pt')
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"最终模型已保存至: {model_path}")
        
        # 保存训练历史
        with open(os.path.join(self.output_path, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=4)
        
        logger.info("模型训练完成")
        
        return history
    
    def evaluate_model(self, X_test_tensor, y_test_tensor, test_cut_nums):
        """
        评估模型
        
        参数:
            X_test_tensor: 测试特征张量
            y_test_tensor: 测试目标张量
            test_cut_nums: 测试数据的切削次数
            
        返回:
            evaluation_metrics: 评估指标字典
        """
        logger.info("评估模型...")
        
        # 切换到评估模式
        self.model.eval()
        
        # 预测
        with torch.no_grad():
            y_pred_tensor = self.model(X_test_tensor)
        
        # 转换为NumPy数组
        y_test = y_test_tensor.cpu().numpy().flatten()
        y_pred = y_pred_tensor.cpu().numpy().flatten()
        
        # 计算评估指标
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # 创建评估指标字典
        evaluation_metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2)
        }
        
        # 将评估指标保存为JSON文件
        with open(os.path.join(self.output_path, 'evaluation.json'), 'w') as f:
            json.dump(evaluation_metrics, f, indent=4)
        
        # 打印评估指标
        print("\n" + "="*50)
        print("CNN模型评估指标 (使用最佳超参数)")
        print("="*50)
        print(f"测试刀具: {self.test_cutter}")
        print(f"训练刀具: {', '.join(self.train_cutters)}")
        print(f"均方误差(MSE): {mse:.6f}")
        print(f"均方根误差(RMSE): {rmse:.6f}")
        print(f"平均绝对误差(MAE): {mae:.6f}")
        print(f"决定系数(R²): {r2:.6f}")
        print("="*50)
        
        # 可视化预测结果
        self.visualize_predictions(y_test, y_pred, test_cut_nums)
        self.visualize_training_history()
        
        return evaluation_metrics
    
    def visualize_predictions(self, y_test, y_pred, test_cut_nums):
        """
        可视化预测结果
        
        参数:
            y_test: 测试集真实值
            y_pred: 预测值
            test_cut_nums: 测试集的切削次数
        """
        # 创建预测对比图
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.7)
        
        # 添加理想预测线
        min_val = min(min(y_test), min(y_pred))
        max_val = max(max(y_test), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.title(f'刀具 {self.test_cutter} 的磨损预测 (CNN - 最佳参数)')
        plt.xlabel('实际磨损值')
        plt.ylabel('预测磨损值')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # 保存预测对比图
        plt.savefig(os.path.join(self.output_path, f'{self.test_cutter}_scatter_best_params.png'))
        plt.close()
        
        # 获取测试序列对应的原始切削次数
        window_size = self.train_params['window_size']
        stride = self.train_params['stride']
        sequence_cut_nums = test_cut_nums[window_size-1:][::stride] # 修正对齐
        
        # 确保长度一致
        min_len = min(len(sequence_cut_nums), len(y_test), len(y_pred))
        sequence_cut_nums = sequence_cut_nums[:min_len]
        y_test_plot = y_test[:min_len]
        y_pred_plot = y_pred[:min_len]
        
        # 创建预测序列图
        plt.figure(figsize=(12, 6))
        
        # 绘制预测值和真实值随切削次数的变化
        plt.plot(sequence_cut_nums, y_test_plot, 'b-', label='实际磨损值')
        plt.plot(sequence_cut_nums, y_pred_plot, 'r--', label='预测磨损值')
        
        plt.title(f'刀具 {self.test_cutter} 的磨损预测随切削次数的变化 (CNN - 最佳参数)')
        plt.xlabel('切削次数')
        plt.ylabel('磨损值')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # 保存预测序列图
        plt.savefig(os.path.join(self.output_path, f'{self.test_cutter}_predictions_best_params.png'))
        plt.close()
    
    def visualize_training_history(self):
        """
        可视化训练历史
        """
        try:
            # 读取训练历史
            with open(os.path.join(self.output_path, 'training_history.json'), 'r') as f:
                history = json.load(f)
            
            # 绘制损失曲线
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(history['train_loss'], label='训练损失')
            if 'valid_loss' in history and history['valid_loss']: # 检查是否有验证损失
                 plt.plot(history['valid_loss'], label='验证损失')
            plt.title('模型损失 (最佳参数)')
            plt.xlabel('Epoch')
            plt.ylabel('损失')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.subplot(1, 2, 2)
            plt.plot(history['lr'], label='学习率')
            plt.title('学习率变化 (最佳参数)')
            plt.xlabel('Epoch / Step (OneCycleLR)') # 根据调度器调整标签
            plt.ylabel('学习率')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, 'training_history_best_params.png'))
            plt.close()
            
        except Exception as e:
            logger.error(f"可视化训练历史时发生错误: {e}")
    
    def run(self):
        """
        运行完整的模型训练和评估流程
        """
        try:
            # 加载原始数据
            X_train_raw, y_train_raw, X_test_raw, y_test_raw, test_cut_nums = self.load_data()
            
            # 预处理数据
            train_loader, X_test_tensor, y_test_tensor = self.preprocess_data(
                X_train_raw, y_train_raw, X_test_raw, y_test_raw
            )
            
            # 构建模型
            self.model = self.build_model(num_features=X_train_raw.shape[1])
            
            # 训练模型
            self.train(train_loader)
            
            # 评估模型
            metrics = self.evaluate_model(X_test_tensor, y_test_tensor, test_cut_nums)
            
            logger.info("CNN模型训练和评估完成！(使用最佳超参数)")
            
            return metrics
        
        except Exception as e:
            logger.error(f"运行过程中发生错误: {e}")
            raise e

def main():
    """主函数"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='铣刀寿命预测系统 - 卷积神经网络模型 (使用最佳超参数)')
    
    # 添加命令行参数
    parser.add_argument('--features_path', type=str, default='/Users/xiaohudemac/cursor01/bishe/5_8tool/data/selected_features',
                      help='特征选择后的数据路径')
    parser.add_argument('--output_path', type=str, default='/Users/xiaohudemac/cursor01/bishe/5_8tool/results/cnn_best_params', # 新的输出路径
                      help='模型输出根路径')
    parser.add_argument('--target_column', type=str, default='wear_VB_avg',
                      help='目标变量列名')
    parser.add_argument('--train_cutters', type=str, default='c1,c4',
                      help='用于训练的刀具列表，用逗号分隔')
    parser.add_argument('--test_cutter', type=str, default='c6',
                      help='用于测试的刀具')
    parser.add_argument('--random_state', type=int, default=RANDOM_SEED,
                      help='随机种子')
    # 移除原有可调参数，因为我们将从JSON文件加载
    parser.add_argument('--best_params_path', type=str, 
                        default='/Users/xiaohudemac/cursor01/bishe/5_8tool/results/cnn_tuning/run_20250509_030535/best_params.json',
                        help='最佳超参数JSON文件路径')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 解析训练刀具列表
    train_cutters = args.train_cutters.split(',')
    
    # 为输出路径添加时间戳，确保每次运行结果保存在单独的文件夹中
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_path, f"run_{timestamp}_best_params")
    
    # 加载最佳超参数
    try:
        with open(args.best_params_path, 'r') as f:
            best_params_loaded = json.load(f)
        logger.info(f"成功加载最佳超参数从: {args.best_params_path}")
    except Exception as e:
        logger.error(f"加载最佳超参数失败: {e}. 将使用脚本内定义的默认值。")
        best_params_loaded = {} # 使用空字典，触发CNNModelTrainer中的默认值

    # 分离模型结构参数和训练参数
    model_structure_keys = ['conv_layers', 'first_conv_out', 'kernel_size', 
                            'dropout_rate1', 'dropout_rate2', 'fc_units1', 'fc_units2',
                            'use_attention', 'attention_weight']
    training_process_keys = ['window_size', 'stride', 'batch_size', 'optimizer', 
                             'learning_rate', 'weight_decay', 'scheduler', 'max_lr_factor']
    
    model_params_from_json = {k: best_params_loaded[k] for k in model_structure_keys if k in best_params_loaded}
    train_params_from_json = {k: best_params_loaded[k] for k in training_process_keys if k in best_params_loaded}
    train_params_from_json['epochs'] = 200 # 固定epochs或从json加载

    # 初始化CNN模型
    cnn_model_trainer = CNNModelTrainer(
        selected_features_path=args.features_path,
        output_path=output_path,
        target_column=args.target_column,
        train_cutters=train_cutters,
        test_cutter=args.test_cutter,
        random_state=args.random_state,
        model_params=model_params_from_json,
        train_params=train_params_from_json
    )
    
    # 打印配置信息
    print("\n" + "="*60)
    print(f"开始使用刀具 {', '.join(train_cutters)} 的数据训练CNN模型 (最佳参数)")
    print(f"测试集使用刀具 {args.test_cutter} 的数据")
    print(f"使用特征数据路径: {args.features_path}")
    print(f"加载的最佳模型参数: {cnn_model_trainer.model_params}")
    print(f"加载的最佳训练参数: {cnn_model_trainer.train_params}")
    print(f"模型结果将保存至: {output_path}")
    print("="*60)
    
    try:
        cnn_model_trainer.run()
        print("\n训练和评估完成!")
        print(f"模型和评估结果已保存至: {output_path}")
        print("="*60)
    except Exception as e:
        print(f"\n训练过程中发生错误: {e}")
        print("="*60)

if __name__ == "__main__":
    main() 