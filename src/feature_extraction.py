#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
铣刀寿命预测系统 - 特征提取模块
功能：从预处理后的数据中提取时域、频域和时频域特征，以及小波包变换特征
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, signal, fft
import pywt
from tqdm import tqdm
import argparse
import glob
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self, processed_data_path, output_path, c_list=None):
        """
        初始化特征提取器
        
        参数:
            processed_data_path: 预处理后的数据路径
            output_path: 特征输出路径
            c_list: 要处理的刀具列表，默认为None表示处理所有刀具
        """
        self.processed_data_path = processed_data_path
        self.output_path = output_path
        self.c_list = c_list if c_list else ['c1', 'c4', 'c6']
        
        # 确保输出目录存在
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
        # 信号通道列表
        self.signal_channels = [
            'Force_X', 'Force_Y', 'Force_Z',
            'Vibration_X', 'Vibration_Y', 'Vibration_Z',
            'AE_RMS'
        ]
    
    def load_normalized_data(self, cutter, cut_num):
        """
        加载标准化后的切削数据
        
        参数:
            cutter: 刀具名称 (c1, c4, c6等)
            cut_num: 切削次数
        
        返回:
            标准化后的切削数据DataFrame
        """
        # 格式化切削次数为3位数字（例如：1 -> 001）
        formatted_cut_num = f"{cut_num:03d}"
        normalized_file = os.path.join(
            self.processed_data_path, 
            cutter, 
            f"c_{cutter[-1]}_{formatted_cut_num}_normalized.csv"
        )
        
        try:
            data = pd.read_csv(normalized_file)
            return data
        except Exception as e:
            logger.error(f"加载标准化数据失败 {normalized_file}: {e}")
            return None
    
    def load_wear_data(self, cutter):
        """
        加载磨损数据
        
        参数:
            cutter: 刀具名称 (c1, c4, c6等)
        
        返回:
            磨损数据DataFrame
        """
        wear_file = os.path.join(self.processed_data_path, cutter, f"{cutter}_wear_processed.csv")
        try:
            wear_data = pd.read_csv(wear_file)
            logger.info(f"成功加载磨损数据: {wear_file}")
            return wear_data
        except Exception as e:
            logger.error(f"加载磨损数据失败: {e}")
            return None
    
    def extract_time_domain_features(self, data):
        """
        提取时域特征
        
        参数:
            data: 输入数据DataFrame
        
        返回:
            时域特征字典
        """
        features = {}
        
        for channel in self.signal_channels:
            signal_data = data[channel].values
            
            # 统计特征
            features[f"{channel}_mean"] = np.mean(signal_data)  # 均值
            features[f"{channel}_std"] = np.std(signal_data)    # 标准差
            features[f"{channel}_var"] = np.var(signal_data)    # 方差
            features[f"{channel}_rms"] = np.sqrt(np.mean(np.square(signal_data)))  # 均方根值
            features[f"{channel}_max"] = np.max(signal_data)    # 最大值
            features[f"{channel}_min"] = np.min(signal_data)    # 最小值
            features[f"{channel}_peak"] = np.max(np.abs(signal_data))  # 峰值
            features[f"{channel}_peak2peak"] = features[f"{channel}_max"] - features[f"{channel}_min"]  # 峰峰值
            features[f"{channel}_crest"] = features[f"{channel}_peak"] / features[f"{channel}_rms"] if features[f"{channel}_rms"] != 0 else 0  # 峰值因子
            
            # 峭度和偏度
            features[f"{channel}_kurtosis"] = stats.kurtosis(signal_data)  # 峭度
            features[f"{channel}_skewness"] = stats.skew(signal_data)      # 偏度
            
            # 能量特征
            features[f"{channel}_energy"] = np.sum(np.square(signal_data))  # 能量
            features[f"{channel}_abs_mean"] = np.mean(np.abs(signal_data))  # 绝对均值
            
            # 形状因子
            features[f"{channel}_shape_factor"] = features[f"{channel}_rms"] / features[f"{channel}_abs_mean"] if features[f"{channel}_abs_mean"] != 0 else 0
            
            # 脉冲因子
            features[f"{channel}_impulse_factor"] = features[f"{channel}_peak"] / features[f"{channel}_abs_mean"] if features[f"{channel}_abs_mean"] != 0 else 0
            
            # 裕度因子
            features[f"{channel}_clearance_factor"] = features[f"{channel}_peak"] / np.mean(np.sqrt(np.abs(signal_data))) if np.mean(np.sqrt(np.abs(signal_data))) != 0 else 0
        
        return features
    
    def extract_frequency_domain_features(self, data, fs=50000):
        """
        提取频域特征
        
        参数:
            data: 输入数据DataFrame
            fs: 采样频率，默认为50kHz
        
        返回:
            频域特征字典
        """
        features = {}
        
        for channel in self.signal_channels:
            signal_data = data[channel].values
            
            # 计算FFT
            n = len(signal_data)
            fft_data = fft.fft(signal_data)
            fft_magnitude = np.abs(fft_data[:n//2]) / n  # 单侧频谱幅值
            freq = np.linspace(0, fs/2, n//2)            # 频率轴
            
            # 频谱特征
            features[f"{channel}_freq_mean"] = np.mean(fft_magnitude)    # 频谱均值
            features[f"{channel}_freq_std"] = np.std(fft_magnitude)      # 频谱标准差
            features[f"{channel}_freq_max"] = np.max(fft_magnitude)      # 频谱最大值
            
            # 频谱能量
            features[f"{channel}_freq_energy"] = np.sum(np.square(fft_magnitude))  # 频谱能量
            
            # 峰值频率及其幅值
            peak_idx = np.argmax(fft_magnitude)
            features[f"{channel}_peak_freq"] = freq[peak_idx]                # 峰值频率
            features[f"{channel}_peak_magnitude"] = fft_magnitude[peak_idx]  # 峰值幅值
            
            # 频谱中心质量
            if np.sum(fft_magnitude) != 0:
                features[f"{channel}_freq_centroid"] = np.sum(freq * fft_magnitude) / np.sum(fft_magnitude)
            else:
                features[f"{channel}_freq_centroid"] = 0
            
            # 频谱偏度和峭度
            features[f"{channel}_freq_skewness"] = stats.skew(fft_magnitude) if len(fft_magnitude) > 0 else 0
            features[f"{channel}_freq_kurtosis"] = stats.kurtosis(fft_magnitude) if len(fft_magnitude) > 0 else 0
            
            # 频段能量比
            total_energy = np.sum(np.square(fft_magnitude))
            if total_energy != 0:
                # 将频谱分为5个频段，计算各频段能量占比
                bands = 5
                band_size = len(freq) // bands
                for i in range(bands):
                    start_idx = i * band_size
                    end_idx = (i + 1) * band_size if i < bands - 1 else len(freq)
                    band_energy = np.sum(np.square(fft_magnitude[start_idx:end_idx]))
                    features[f"{channel}_freq_band_{i+1}_ratio"] = band_energy / total_energy
            else:
                for i in range(5):
                    features[f"{channel}_freq_band_{i+1}_ratio"] = 0
        
        return features
    
    def extract_time_frequency_features(self, data, scales=32):
        """
        提取时频域特征（连续小波变换）
        
        参数:
            data: 输入数据DataFrame
            scales: 尺度数量，默认为32
        
        返回:
            时频域特征字典
        """
        features = {}
        
        # 定义小波
        wavelet = 'morl'  # Morlet小波
        
        for channel in self.signal_channels:
            signal_data = data[channel].values
            
            # 计算连续小波变换
            scales_array = np.arange(1, scales + 1)
            coefficients, frequencies = pywt.cwt(signal_data, scales_array, wavelet)
            
            # 小波能量
            wavelet_energy = np.sum(np.square(np.abs(coefficients)), axis=1)
            
            # 提取特征
            features[f"{channel}_wavelet_mean"] = np.mean(wavelet_energy)     # 小波能量均值
            features[f"{channel}_wavelet_std"] = np.std(wavelet_energy)       # 小波能量标准差
            features[f"{channel}_wavelet_max"] = np.max(wavelet_energy)       # 最大小波能量
            features[f"{channel}_wavelet_min"] = np.min(wavelet_energy)       # 最小小波能量
            
            # 小波熵 (归一化能量后的Shannon熵)
            normalized_energy = wavelet_energy / np.sum(wavelet_energy) if np.sum(wavelet_energy) != 0 else np.zeros_like(wavelet_energy)
            entropy = -np.sum(normalized_energy * np.log2(normalized_energy + 1e-10))
            features[f"{channel}_wavelet_entropy"] = entropy
            
            # 各尺度能量比例
            total_energy = np.sum(wavelet_energy)
            if total_energy != 0:
                # 选取几个关键尺度
                key_scales = [1, 4, 8, 16, 32] if scales >= 32 else list(range(1, scales + 1, max(1, scales // 5)))
                for scale in key_scales:
                    if scale <= scales:
                        scale_idx = scale - 1
                        features[f"{channel}_wavelet_scale_{scale}_ratio"] = wavelet_energy[scale_idx] / total_energy
            else:
                key_scales = [1, 4, 8, 16, 32] if scales >= 32 else list(range(1, scales + 1, max(1, scales // 5)))
                for scale in key_scales:
                    if scale <= scales:
                        features[f"{channel}_wavelet_scale_{scale}_ratio"] = 0
        
        return features
    
    def extract_wavelet_packet_features(self, data, wavelet='db4', level=3):
        """
        提取小波包变换特征
        
        参数:
            data: 输入数据DataFrame
            wavelet: 小波基函数，默认为'db4'
            level: 分解层数，默认为3
        
        返回:
            小波包特征字典
        """
        features = {}
        
        for channel in self.signal_channels:
            signal_data = data[channel].values
            
            # 小波包分解
            wp = pywt.WaveletPacket(data=signal_data, wavelet=wavelet, mode='symmetric', maxlevel=level)
            
            # 获取指定层级的所有节点
            nodes = [node.path for node in wp.get_level(level, 'natural')]
            
            # 对每个节点提取特征
            for node in nodes:
                # 获取节点系数
                coeff = wp[node].data
                
                # 提取特征
                features[f"{channel}_wp_{node}_mean"] = np.mean(coeff)       # 均值
                features[f"{channel}_wp_{node}_std"] = np.std(coeff)         # 标准差
                features[f"{channel}_wp_{node}_energy"] = np.sum(coeff**2)   # 能量
                features[f"{channel}_wp_{node}_entropy"] = np.sum(-coeff**2 * np.log(coeff**2 + 1e-10))  # 熵
        
        return features
    
    def extract_all_features(self, data):
        """
        提取所有特征
        
        参数:
            data: 输入数据DataFrame
        
        返回:
            所有特征合并后的字典
        """
        # 提取各类特征
        time_features = self.extract_time_domain_features(data)
        freq_features = self.extract_frequency_domain_features(data)
        time_freq_features = self.extract_time_frequency_features(data)
        wp_features = self.extract_wavelet_packet_features(data)
        
        # 合并所有特征
        all_features = {}
        all_features.update(time_features)
        all_features.update(freq_features)
        all_features.update(time_freq_features)
        all_features.update(wp_features)
        
        return all_features
    
    def normalize_features(self, features_df):
        """
        归一化特征数据
        
        参数:
            features_df: 特征DataFrame
        
        返回:
            归一化后的特征DataFrame
        """
        # 复制原始DataFrame
        normalized_df = features_df.copy()
        
        # 确定需要归一化的列
        non_feature_cols = ['cut_num', 'wear_VB1', 'wear_VB2', 'wear_VB3', 'wear_VB_avg']
        feature_cols = [col for col in features_df.columns if col not in non_feature_cols]
        
        # 应用Min-Max归一化
        for col in feature_cols:
            min_val = features_df[col].min()
            max_val = features_df[col].max()
            
            # 避免除以零
            if max_val - min_val > 0:
                normalized_df[col] = (features_df[col] - min_val) / (max_val - min_val)
            else:
                # 如果所有值相同，设为0.5
                normalized_df[col] = 0.5
        
        return normalized_df
    
    def process_cutter_data(self, cutter):
        """
        处理单个刀具的所有数据
        
        参数:
            cutter: 刀具名称
        """
        logger.info(f"开始提取刀具 {cutter} 的特征...")
        
        # 加载磨损数据
        wear_data = self.load_wear_data(cutter)
        if wear_data is None:
            return
        
        # 确保输出目录存在
        cutter_output_dir = os.path.join(self.output_path, cutter)
        if not os.path.exists(cutter_output_dir):
            os.makedirs(cutter_output_dir)
        
        # 获取所有切削次数
        cut_numbers = wear_data['cut'].tolist()
        
        # 存储所有切削的特征
        all_features = []
        
        # 处理每次切削数据
        for cut_num in tqdm(cut_numbers, desc=f"提取{cutter}的特征"):
            # 加载标准化后的数据
            cutting_data = self.load_normalized_data(cutter, cut_num)
            if cutting_data is None:
                continue
            
            # 提取特征
            features = self.extract_all_features(cutting_data)
            
            # 添加切削次数和磨损值
            features['cut_num'] = cut_num
            
            # 查找对应的磨损值
            wear_row = wear_data[wear_data['cut'] == cut_num]
            if not wear_row.empty:
                features['wear_VB1'] = wear_row['flute_1'].values[0]
                features['wear_VB2'] = wear_row['flute_2'].values[0]
                features['wear_VB3'] = wear_row['flute_3'].values[0]
                features['wear_VB_avg'] = (features['wear_VB1'] + features['wear_VB2'] + features['wear_VB3']) / 3
            
            # 添加到列表
            all_features.append(features)
        
        # 转换为DataFrame
        if all_features:
            features_df = pd.DataFrame(all_features)
            
            # 保存原始特征
            output_file = os.path.join(cutter_output_dir, f"{cutter}_features.csv")
            features_df.to_csv(output_file, index=False)
            logger.info(f"原始特征已保存至: {output_file}")
            
            # 特征归一化
            normalized_features_df = self.normalize_features(features_df)
            
            # 保存归一化后的特征
            normalized_output_file = os.path.join(cutter_output_dir, f"{cutter}_normalized_features.csv")
            normalized_features_df.to_csv(normalized_output_file, index=False)
            logger.info(f"归一化特征已保存至: {normalized_output_file}")
            
            # 可视化原始特征和归一化特征的对比
            self.visualize_normalization_comparison(features_df, normalized_features_df, cutter)
        else:
            logger.warning(f"未能提取{cutter}的有效特征")
        
        logger.info(f"刀具 {cutter} 的特征提取完成")
    
    def visualize_normalization_comparison(self, original_df, normalized_df, cutter, n_features=5):
        """
        可视化原始特征和归一化特征的对比
        
        参数:
            original_df: 原始特征DataFrame
            normalized_df: 归一化特征DataFrame
            cutter: 刀具名称
            n_features: 要显示的特征数量
        """
        # 选择相关性最高的n个特征进行可视化
        wear_col = 'wear_VB_avg'
        non_feature_cols = ['cut_num', 'wear_VB1', 'wear_VB2', 'wear_VB3', 'wear_VB_avg']
        feature_cols = [col for col in original_df.columns if col not in non_feature_cols]
        
        # 计算相关系数
        correlations = {}
        for col in feature_cols:
            corr = original_df[col].corr(original_df[wear_col])
            correlations[col] = abs(corr)  # 使用绝对值
        
        # 获取相关性最高的n个特征
        top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:n_features]
        
        # 可视化对比
        fig, axes = plt.subplots(n_features, 2, figsize=(16, 4 * n_features))
        
        for i, (feature, corr) in enumerate(top_features):
            # 原始特征
            ax1 = axes[i, 0]
            ax1.scatter(original_df['cut_num'], original_df[feature], label='原始特征', alpha=0.7)
            ax1.set_title(f"{feature} - 原始特征 (相关系数: {corr:.3f})")
            ax1.set_xlabel('切削次数')
            ax1.set_ylabel('特征值')
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # 在同一图中添加磨损值
            ax1_twin = ax1.twinx()
            ax1_twin.plot(original_df['cut_num'], original_df[wear_col], 'r-', label='磨损值')
            ax1_twin.set_ylabel('磨损值 (mm)', color='r')
            ax1_twin.tick_params(axis='y', labelcolor='r')
            
            # 归一化特征
            ax2 = axes[i, 1]
            ax2.scatter(normalized_df['cut_num'], normalized_df[feature], label='归一化特征', color='g', alpha=0.7)
            ax2.set_title(f"{feature} - 归一化特征")
            ax2.set_xlabel('切削次数')
            ax2.set_ylabel('归一化特征值')
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            # 在同一图中添加磨损值
            ax2_twin = ax2.twinx()
            ax2_twin.plot(normalized_df['cut_num'], normalized_df[wear_col], 'r-', label='磨损值')
            ax2_twin.set_ylabel('磨损值 (mm)', color='r')
            ax2_twin.tick_params(axis='y', labelcolor='r')
            
            # 添加图例
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax1_twin.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_twin.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, f"{cutter}_normalization_comparison.png"))
        plt.close()
        logger.info(f"归一化对比图已保存至: {os.path.join(self.output_path, f'{cutter}_normalization_comparison.png')}")
    
    def process_all_data(self):
        """
        处理所有刀具数据
        """
        for cutter in self.c_list:
            self.process_cutter_data(cutter)
        
        logger.info("所有特征提取完成")
    
    def visualize_features(self, cutter, n_features=10):
        """
        可视化特征与磨损的关系
        
        参数:
            cutter: 刀具名称
            n_features: 显示的特征数量
        """
        # 加载特征数据
        features_file = os.path.join(self.output_path, cutter, f"{cutter}_features.csv")
        if not os.path.exists(features_file):
            logger.error(f"找不到特征文件: {features_file}")
            return
        
        features_df = pd.read_csv(features_file)
        
        # 计算每个特征与磨损值的相关性
        wear_col = 'wear_VB_avg'  # 使用平均磨损值
        if wear_col not in features_df.columns:
            logger.error(f"特征数据中不包含磨损值列: {wear_col}")
            return
        
        # 计算相关系数
        correlations = {}
        for col in features_df.columns:
            if col not in ['cut_num', 'wear_VB1', 'wear_VB2', 'wear_VB3', 'wear_VB_avg']:
                corr = features_df[col].corr(features_df[wear_col])
                correlations[col] = abs(corr)  # 使用绝对值
        
        # 获取相关性最高的n个特征
        top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:n_features]
        
        # 可视化
        fig, axes = plt.subplots(n_features, 1, figsize=(12, 4 * n_features))
        for i, (feature, corr) in enumerate(top_features):
            ax = axes[i] if n_features > 1 else axes
            ax.scatter(features_df['cut_num'], features_df[feature], label=feature, alpha=0.7)
            ax.set_title(f"{feature} (相关系数: {corr:.3f})")
            ax.set_xlabel('切削次数')
            ax.set_ylabel('特征值')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # 添加磨损值到次坐标轴
            ax2 = ax.twinx()
            ax2.plot(features_df['cut_num'], features_df[wear_col], 'r-', label=wear_col)
            ax2.set_ylabel('磨损值 (mm)', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            
            # 添加图例
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, f"{cutter}_top_features.png"))
        plt.close()
        logger.info(f"特征可视化图已保存至: {os.path.join(self.output_path, f'{cutter}_top_features.png')}")


def main():
    """主函数"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='铣刀寿命预测系统 - 特征提取模块')
    
    # 添加命令行参数
    parser.add_argument('--processed_path', type=str, default='/Users/xiaohudemac/cursor01/bishe/5_8tool/data/processed',
                        help='预处理后的数据路径')
    parser.add_argument('--output_path', type=str, default='/Users/xiaohudemac/cursor01/bishe/5_8tool/data/features',
                        help='特征输出路径')
    parser.add_argument('--cutters', type=str, default='c1,c4,c6',
                        help='要处理的刀具列表，用逗号分隔')
    parser.add_argument('--visualize', action='store_true',
                        help='是否生成特征可视化')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 解析刀具列表
    cutters = args.cutters.split(',')
    
    # 初始化特征提取器
    extractor = FeatureExtractor(
        processed_data_path=args.processed_path,
        output_path=args.output_path,
        c_list=cutters
    )
    
    # 提取特征
    extractor.process_all_data()
    
    # 如果需要可视化
    if args.visualize:
        for cutter in cutters:
            extractor.visualize_features(cutter)
    
    logger.info("特征提取完成！")


if __name__ == "__main__":
    main() 