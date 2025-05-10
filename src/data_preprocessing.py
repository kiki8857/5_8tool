#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
铣刀寿命预测系统 - 数据预处理模块
功能：数据读取、清洗、降噪、平滑和标准化
"""

import os
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import glob
import argparse
from tqdm import tqdm

class DataPreprocessor:
    def __init__(self, raw_data_path, output_path, c_list=None):
        """
        初始化数据预处理器
        
        参数:
            raw_data_path: 原始数据路径
            output_path: 输出数据路径
            c_list: 要处理的刀具列表，默认为None表示处理所有刀具
        """
        self.raw_data_path = raw_data_path
        self.output_path = output_path
        self.c_list = c_list if c_list else ['c1', 'c4', 'c6']
        
        # 创建输出目录
        if not os.path.exists(output_path):
            os.makedirs(output_path)
    
    def load_wear_data(self, cutter):
        """
        加载磨损数据
        
        参数:
            cutter: 刀具名称 (c1, c4, c6等)
        
        返回:
            磨损数据DataFrame
        """
        wear_file = os.path.join(self.raw_data_path, cutter, f"{cutter}_wear.csv")
        try:
            wear_data = pd.read_csv(wear_file)
            print(f"成功加载磨损数据: {wear_file}")
            return wear_data
        except Exception as e:
            print(f"加载磨损数据失败: {e}")
            return None
    
    def load_single_cutting_data(self, cutter, cut_num):
        """
        加载单次切削数据
        
        参数:
            cutter: 刀具名称 (c1, c4, c6等)
            cut_num: 切削次数
        
        返回:
            切削数据DataFrame或None（如果加载失败）
        """
        # 格式化切削次数为3位数字（例如：1 -> 001）
        formatted_cut_num = f"{cut_num:03d}"
        cutting_file = os.path.join(self.raw_data_path, cutter, cutter, f"c_{cutter[-1]}_{formatted_cut_num}.csv")
        try:
            # 定义列名
            column_names = ['Force_X', 'Force_Y', 'Force_Z', 
                           'Vibration_X', 'Vibration_Y', 'Vibration_Z', 
                           'AE_RMS']
            cutting_data = pd.read_csv(cutting_file, header=None, names=column_names)
            return cutting_data
        except Exception as e:
            print(f"加载切削数据失败 {cutting_file}: {e}")
            return None
    
    def remove_outliers(self, df, threshold=3):
        """
        使用Z-score方法移除异常值
        
        参数:
            df: 输入数据DataFrame
            threshold: Z-score阈值，默认为3
        
        返回:
            清洗后的DataFrame
        """
        # 创建所有值均为True的掩码
        mask = pd.Series([True] * len(df), index=df.index)
        
        for column in df.columns:
            mean_val = df[column].mean()
            std_val = df[column].std()
            if std_val > 0:  # 防止除以零
                # 计算Z-scores
                z_scores = np.abs((df[column] - mean_val) / std_val)
                # 更新掩码，只保留Z-score小于阈值的行
                mask = mask & (z_scores < threshold)
        
        # 应用掩码
        df_clean = df[mask]
        
        # 如果过滤后的数据太少（少于原始数据的50%），则返回原始数据
        if len(df_clean) < len(df) * 0.5:
            print(f"警告: 过滤后的数据太少 ({len(df_clean)}/{len(df)}), 返回原始数据")
            return df
        
        return df_clean
    
    def apply_smoothing(self, df, window_size=5, method='moving_avg'):
        """
        应用平滑处理
        
        参数:
            df: 输入数据DataFrame
            window_size: 窗口大小
            method: 平滑方法 ('moving_avg', 'savgol', 'exp')
        
        返回:
            平滑后的DataFrame
        """
        df_smooth = df.copy()
        
        for column in df.columns:
            if method == 'moving_avg':
                # 移动平均平滑
                df_smooth[column] = df[column].rolling(window=window_size, center=True).mean()
                # 填充NaN值（窗口边缘）
                df_smooth[column] = df_smooth[column].fillna(df[column])
            
            elif method == 'savgol':
                # Savitzky-Golay滤波
                if len(df) > window_size:
                    polyorder = min(window_size - 1, 3)  # 多项式阶数
                    df_smooth[column] = signal.savgol_filter(df[column], window_size, polyorder)
                else:
                    df_smooth[column] = df[column]  # 如果数据点太少则不处理
            
            elif method == 'exp':
                # 指数平滑
                alpha = 2 / (window_size + 1)  # 平滑系数
                df_smooth[column] = df[column].ewm(alpha=alpha, adjust=False).mean()
        
        return df_smooth
    
    def apply_noise_reduction(self, df, method='butter', cutoff=0.1):
        """
        应用噪声降低滤波
        
        参数:
            df: 输入数据DataFrame
            method: 滤波方法 ('butter', 'median')
            cutoff: 截止频率（butter滤波器用）
        
        返回:
            降噪后的DataFrame
        """
        df_filtered = df.copy()
        
        for column in df.columns:
            if method == 'butter':
                # Butterworth低通滤波
                b, a = signal.butter(4, cutoff, 'low')
                df_filtered[column] = signal.filtfilt(b, a, df[column])
            
            elif method == 'median':
                # 中值滤波
                df_filtered[column] = signal.medfilt(df[column], 5)
        
        return df_filtered
    
    def normalize_data(self, df):
        """
        标准化数据
        
        参数:
            df: 输入数据DataFrame
        
        返回:
            标准化后的DataFrame
        """
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        return df_scaled
    
    def process_cutter_data(self, cutter, sample_rate=None):
        """
        处理单个刀具的所有数据
        
        参数:
            cutter: 刀具名称
            sample_rate: 采样率（如果不为None，则进行降采样）
        
        返回:
            处理后的数据存储在输出目录中
        """
        print(f"开始处理刀具 {cutter} 的数据...")
        
        # 加载磨损数据
        wear_data = self.load_wear_data(cutter)
        if wear_data is None:
            return
        
        # 确保输出目录存在
        cutter_output_dir = os.path.join(self.output_path, cutter)
        if not os.path.exists(cutter_output_dir):
            os.makedirs(cutter_output_dir)
            
        # 保存处理后的磨损数据
        wear_output_file = os.path.join(cutter_output_dir, f"{cutter}_wear_processed.csv")
        wear_data.to_csv(wear_output_file, index=False)
        print(f"磨损数据已保存至: {wear_output_file}")
        
        # 获取所有切削次数
        cut_numbers = wear_data['cut'].tolist()
        
        # 处理每次切削数据
        for cut_num in tqdm(cut_numbers, desc=f"处理{cutter}的切削数据"):
            cutting_data = self.load_single_cutting_data(cutter, cut_num)
            if cutting_data is None:
                continue
            
            # 数据预处理步骤
            # 1. 截取数据范围 (50000:100000)
            data_range_start = 50000
            data_range_end = 100000
            
            # 检查数据长度，确保范围有效
            if len(cutting_data) > data_range_start:
                end_idx = min(data_range_end, len(cutting_data))
                cutting_data = cutting_data.iloc[data_range_start:end_idx].reset_index(drop=True)
                print(f"截取数据点范围: {data_range_start}:{end_idx}, 数据点数量: {len(cutting_data)}")
            else:
                print(f"警告: 数据点数量({len(cutting_data)})小于起始索引({data_range_start}), 使用全部数据")
            
            # 2. 异常值移除 
            cutting_data = self.remove_outliers(cutting_data, threshold=3)
            
            # 3. 噪声降低
            cutting_data = self.apply_noise_reduction(cutting_data, cutoff=0.25)
            
            # 4. 平滑处理
            cutting_data = self.apply_smoothing(cutting_data, window_size=3)
            
            # 5. 标准化
            cutting_data_normalized = self.normalize_data(cutting_data)
            
            # 格式化切削次数为3位数字（例如：1 -> 001）
            formatted_cut_num = f"{cut_num:03d}"
            
            # 保存处理后的切削数据
            output_file = os.path.join(cutter_output_dir, f"c_{cutter[-1]}_{formatted_cut_num}_processed.csv")
            cutting_data.to_csv(output_file, index=False)
            
            # 保存标准化后的切削数据
            norm_output_file = os.path.join(cutter_output_dir, f"c_{cutter[-1]}_{formatted_cut_num}_normalized.csv")
            cutting_data_normalized.to_csv(norm_output_file, index=False)
        
        print(f"刀具 {cutter} 的数据处理完成")
    
    def process_all_data(self, sample_rate=None):
        """
        处理所有刀具数据
        
        参数:
            sample_rate: 降采样率，如果为None则不降采样
        """
        for cutter in self.c_list:
            self.process_cutter_data(cutter, sample_rate)
        
        print("所有数据处理完成")
    
    def visualize_data(self, cutter, cut_num, before_after=True):
        """
        可视化处理前后的数据对比
        
        参数:
            cutter: 刀具名称
            cut_num: 切削次数
            before_after: 是否显示处理前后的对比
        """
        # 加载原始数据
        original_data = self.load_single_cutting_data(cutter, cut_num)
        if original_data is None:
            return
        
        # 格式化切削次数为3位数字（例如：1 -> 001）
        formatted_cut_num = f"{cut_num:03d}"
        
        # 如果需要对比，加载处理后的数据
        if before_after:
            processed_file = os.path.join(self.output_path, cutter, f"c_{cutter[-1]}_{formatted_cut_num}_processed.csv")
            if os.path.exists(processed_file):
                processed_data = pd.read_csv(processed_file)
                
                # 创建对比图
                fig, axes = plt.subplots(7, 2, figsize=(15, 20))
                
                for i, col in enumerate(original_data.columns):
                    # 原始数据
                    axes[i, 0].plot(original_data[col].values[:1000])
                    axes[i, 0].set_title(f'原始 {col}')
                    
                    # 处理后的数据
                    axes[i, 1].plot(processed_data[col].values[:1000])
                    axes[i, 1].set_title(f'处理后 {col}')
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_path, f"{cutter}_cut_{formatted_cut_num}_comparison.png"))
                plt.close()
                print(f"对比图已保存至: {os.path.join(self.output_path, f'{cutter}_cut_{formatted_cut_num}_comparison.png')}")
            else:
                print(f"找不到处理后的文件: {processed_file}")
        else:
            # 只显示原始数据
            fig, axes = plt.subplots(7, 1, figsize=(15, 20))
            
            for i, col in enumerate(original_data.columns):
                axes[i].plot(original_data[col].values[:1000])
                axes[i].set_title(col)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, f"{cutter}_cut_{formatted_cut_num}_original.png"))
            plt.close()
            print(f"原始数据图已保存至: {os.path.join(self.output_path, f'{cutter}_cut_{formatted_cut_num}_original.png')}")

    def generate_comparison_plot(self, cutter, cut_num):
        """
        生成处理前后的信号对比图（时域、频域和功率谱密度）
        
        参数:
            cutter: 刀具名称
            cut_num: 切削次数
        """
        # 格式化切削次数为3位数字（例如：1 -> 001）
        formatted_cut_num = f"{cut_num:03d}"
        
        # 加载原始数据
        original_data = self.load_single_cutting_data(cutter, cut_num)
        if original_data is None:
            print(f"无法加载刀具 {cutter} 的原始数据，切削次数 {cut_num}")
            return
        
        # 截取原始数据的相同范围 (50000:100000)
        data_range_start = 50000
        data_range_end = 100000
        if len(original_data) > data_range_start:
            end_idx = min(data_range_end, len(original_data))
            original_data = original_data.iloc[data_range_start:end_idx].reset_index(drop=True)
        
        # 加载处理后的数据
        processed_file = os.path.join(self.output_path, cutter, f"c_{cutter[-1]}_{formatted_cut_num}_processed.csv")
        normalized_file = os.path.join(self.output_path, cutter, f"c_{cutter[-1]}_{formatted_cut_num}_normalized.csv")
        
        if not os.path.exists(processed_file) or not os.path.exists(normalized_file):
            print(f"找不到处理后的文件，尝试处理数据...")
            cutting_data = original_data.copy()
            
            # 数据预处理步骤
            # 1. 异常值移除 (使用3德尔塔准则)
            cutting_data_clean = self.remove_outliers(cutting_data, threshold=3)
            
            # 2. 噪声降低
            cutting_data_filtered = self.apply_noise_reduction(cutting_data_clean, cutoff=0.25)
            
            # 3. 平滑处理
            cutting_data_smoothed = self.apply_smoothing(cutting_data_filtered, window_size=3)
            
            # 4. 标准化
            cutting_data_normalized = self.normalize_data(cutting_data_smoothed)
            
            processed_data = cutting_data_smoothed
            normalized_data = cutting_data_normalized
        else:
            processed_data = pd.read_csv(processed_file)
            normalized_data = pd.read_csv(normalized_file)
        
        # 选择信号通道进行可视化
        selected_channels = ['Force_Z', 'Vibration_X', 'AE_RMS']
        
        # 创建时域、频域和功率谱密度的对比图
        plt.figure(figsize=(18, 15))
        
        # 计算样本频率 (假设为1kHz，可根据实际情况调整)
        fs = 1000  # Hz
        
        for i, channel in enumerate(selected_channels):
            # ======= 时域分析 =======
            # 原始数据
            plt.subplot(3, 3, i*3+1)
            plt.plot(original_data[channel].values, 'b-')
            plt.title(f'时域 - 原始信号 - {channel}')
            plt.xlabel('采样点')
            plt.ylabel('幅值')
            plt.grid(True)
            
            # ======= 频域分析 =======
            plt.subplot(3, 3, i*3+2)
            
            # 原始数据的FFT
            original_fft = np.fft.rfft(original_data[channel].values)
            original_freq = np.fft.rfftfreq(len(original_data[channel].values), d=1/fs)
            
            # 处理后数据的FFT
            processed_fft = np.fft.rfft(processed_data[channel].values)
            processed_freq = np.fft.rfftfreq(len(processed_data[channel].values), d=1/fs)
            
            # 归一化幅值
            original_amp = np.abs(original_fft) / len(original_data)
            processed_amp = np.abs(processed_fft) / len(processed_data)
            
            # 绘制频谱
            plt.plot(original_freq, original_amp, 'b-', label='原始信号')
            plt.plot(processed_freq, processed_amp, 'g-', label='处理后信号')
            plt.title(f'频域 - {channel}')
            plt.xlabel('频率 (Hz)')
            plt.ylabel('幅值')
            plt.legend()
            plt.grid(True)
            
            # 限制x轴范围到有意义的频率
            plt.xlim(0, fs/2)
            
            # ======= 功率谱密度 =======
            plt.subplot(3, 3, i*3+3)
            
            # 计算原始数据的PSD
            f_original, psd_original = signal.welch(original_data[channel].values, fs, nperseg=1024)
            
            # 计算处理后数据的PSD
            f_processed, psd_processed = signal.welch(processed_data[channel].values, fs, nperseg=1024)
            
            # 计算标准化后数据的PSD
            f_normalized, psd_normalized = signal.welch(normalized_data[channel].values, fs, nperseg=1024)
            
            # 绘制PSD
            plt.semilogy(f_original, psd_original, 'b-', label='原始信号')
            plt.semilogy(f_processed, psd_processed, 'g-', label='处理后信号')
            plt.semilogy(f_normalized, psd_normalized, 'r-', label='标准化后信号')
            plt.title(f'功率谱密度 - {channel}')
            plt.xlabel('频率 (Hz)')
            plt.ylabel('功率/频率 (dB/Hz)')
            plt.legend()
            plt.grid(True)
        
        plt.suptitle(f'刀具 {cutter} - 切削次数 {cut_num} 的信号处理对比 (范围: {data_range_start}-{data_range_end})', fontsize=16)
        plt.tight_layout()
        
        output_fig_path = os.path.join(self.output_path, f"{cutter}_cut_{formatted_cut_num}_signal_comparison_{data_range_start}_{data_range_end}.png")
        plt.savefig(output_fig_path, dpi=300)
        plt.close()
        
        print(f"信号处理对比图已保存至: {output_fig_path}")


def main():
    """主函数"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='铣刀寿命预测系统 - 数据预处理模块')
    
    # 添加命令行参数
    parser.add_argument('--raw_path', type=str, default='/Users/xiaohudemac/cursor01/bishe/5_8tool/data/raw/PHM_2010',
                        help='原始数据路径')
    parser.add_argument('--output_path', type=str, default='/Users/xiaohudemac/cursor01/bishe/5_8tool/data/processed',
                        help='处理后数据的输出路径')
    parser.add_argument('--cutters', type=str, default='c1,c4,c6',
                        help='要处理的刀具列表，用逗号分隔')
    parser.add_argument('--sample_rate', type=int, default=None,
                        help='降采样率（每N个点取1个），设为None禁用降采样')
    parser.add_argument('--visualize', action='store_true',
                        help='是否生成可视化对比图')
    parser.add_argument('--cut_vis', type=int, default=1,
                        help='用于可视化的切削次数')
    parser.add_argument('--comparison_plot', action='store_true',
                        help='是否生成处理前后的信号对比图')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 解析刀具列表
    cutters = args.cutters.split(',')
    
    # 初始化数据预处理器
    preprocessor = DataPreprocessor(
        raw_data_path=args.raw_path,
        output_path=args.output_path,
        c_list=cutters
    )
    
    # 处理所有数据
    preprocessor.process_all_data(sample_rate=args.sample_rate)
    
    # 如果需要可视化
    if args.visualize:
        for cutter in cutters:
            preprocessor.visualize_data(cutter, args.cut_vis)
    
    # 如果需要生成信号处理对比图
    if args.comparison_plot:
        for cutter in cutters:
            preprocessor.generate_comparison_plot(cutter, args.cut_vis)
    else:
        # 为了满足用户要求，默认生成一组对比图
        print("生成一组信号处理对比图...")
        preprocessor.generate_comparison_plot(cutters[0], 1)
    
    print("数据预处理完成！")


if __name__ == "__main__":
    main() 