#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
铣刀寿命预测系统 - 数据可视化模块
功能：为每把刀生成三张可视化图表：时域图、频域图和功率谱密度图
"""

import os
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib import font_manager
import argparse
from tqdm import tqdm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'STXihei', 'STHeiti', 'Heiti SC', 'STKaiti', 'Ar PL UMing CN']
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

class SignalVisualizer:
    def __init__(self, data_path, output_path, fs=1000):
        """
        初始化信号可视化器
        
        参数:
            data_path: 处理后数据路径
            output_path: 可视化图表输出路径
            fs: 采样频率 (Hz)，默认1000Hz
        """
        self.data_path = data_path
        self.output_path = output_path
        self.fs = fs
        
        # 创建输出目录
        if not os.path.exists(output_path):
            os.makedirs(output_path)
    
    def load_data(self, cutter, cut_num, data_type='normalized'):
        """
        加载处理后的数据
        
        参数:
            cutter: 刀具名称 (c1, c4, c6等)
            cut_num: 切削次数
            data_type: 数据类型 ('original', 'processed', 'normalized')
        
        返回:
            加载的数据DataFrame或None（如果加载失败）
        """
        # 确保切削次数为整数并格式化为3位数字（例如：1 -> 001）
        try:
            cut_num_int = int(cut_num)
            formatted_cut_num = f"{cut_num_int:03d}"
        except (ValueError, TypeError):
            print(f"警告: 切削次数 '{cut_num}' 不是有效的整数，尝试使用原始值")
            formatted_cut_num = str(cut_num).replace('.', '_')
        
        if data_type == 'original':
            # 原始数据文件路径
            data_file = os.path.join(self.data_path, 'raw', 'PHM_2010', cutter, cutter, 
                                    f"c_{cutter[-1]}_{formatted_cut_num}.csv")
            try:
                # 定义列名
                column_names = ['Force_X', 'Force_Y', 'Force_Z', 
                               'Vibration_X', 'Vibration_Y', 'Vibration_Z', 
                               'AE_RMS']
                data = pd.read_csv(data_file, header=None, names=column_names)
                
                # 截取50000:100000的范围
                if len(data) > 50000:
                    end_idx = min(100000, len(data))
                    data = data.iloc[50000:end_idx].reset_index(drop=True)
                
                return data
            except Exception as e:
                print(f"加载原始数据失败 {data_file}: {e}")
                return None
        else:
            # 处理后的数据文件路径
            suffix = '_processed.csv' if data_type == 'processed' else '_normalized.csv'
            data_file = os.path.join(self.data_path, 'processed', cutter, 
                                    f"c_{cutter[-1]}_{formatted_cut_num}{suffix}")
            try:
                data = pd.read_csv(data_file)
                return data
            except Exception as e:
                print(f"加载{data_type}数据失败 {data_file}: {e}")
                return None
    
    def load_wear_data(self, cutter):
        """
        加载磨损数据
        
        参数:
            cutter: 刀具名称
            
        返回:
            磨损数据DataFrame
        """
        wear_file = os.path.join(self.data_path, 'processed', cutter, f"{cutter}_wear_processed.csv")
        try:
            wear_data = pd.read_csv(wear_file)
            return wear_data
        except Exception as e:
            print(f"加载磨损数据失败 {wear_file}: {e}")
            # 尝试原始磨损数据
            original_wear_file = os.path.join(self.data_path, 'raw', 'PHM_2010', f"{cutter}_wear.csv")
            try:
                wear_data = pd.read_csv(original_wear_file)
                return wear_data
            except Exception as e:
                print(f"加载原始磨损数据失败 {original_wear_file}: {e}")
                return None
    
    def create_time_domain_plot(self, cutter, cut_num):
        """
        创建时域图
        
        参数:
            cutter: 刀具名称
            cut_num: 切削次数
        """
        # 加载三种类型的数据
        original_data = self.load_data(cutter, cut_num, 'original')
        processed_data = self.load_data(cutter, cut_num, 'processed')
        normalized_data = self.load_data(cutter, cut_num, 'normalized')
        
        if original_data is None or processed_data is None or normalized_data is None:
            print(f"无法为刀具 {cutter} 的切削次数 {cut_num} 创建时域图，数据不完整")
            return
        
        # 选择信号通道
        channels = ['Force_X', 'Force_Y', 'Force_Z', 
                   'Vibration_X', 'Vibration_Y', 'Vibration_Z', 
                   'AE_RMS']
        
        # 创建7x3的时域图（7个通道，3种信号类型）
        plt.figure(figsize=(18, 24))
        plt.suptitle(f'刀具 {cutter} - 切削次数 {cut_num} 的时域信号分析', fontsize=16)
        
        for i, channel in enumerate(channels):
            # 第一列：原始信号
            ax1 = plt.subplot(7, 3, i*3+1)
            ax1.plot(original_data[channel].values, 'b-')
            ax1.set_title(f'原始信号 - {channel}')
            ax1.set_xlabel('采样点')
            ax1.set_ylabel('幅值')
            ax1.grid(True)
            
            # 第二列：处理后信号
            ax2 = plt.subplot(7, 3, i*3+2)
            ax2.plot(processed_data[channel].values, 'g-')
            ax2.set_title(f'处理后信号 - {channel}')
            ax2.set_xlabel('采样点')
            ax2.set_ylabel('幅值')
            ax2.grid(True)
            
            # 第三列：标准化信号
            ax3 = plt.subplot(7, 3, i*3+3)
            ax3.plot(normalized_data[channel].values, 'r-')
            ax3.set_title(f'标准化信号 - {channel}')
            ax3.set_xlabel('采样点')
            ax3.set_ylabel('幅值')
            ax3.grid(True)
        
        # 紧凑布局
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 保存图表
        output_fig_path = os.path.join(self.output_path, f"{cutter}_cut_{cut_num:03d}_time_domain.png")
        plt.savefig(output_fig_path, dpi=300)
        plt.close()
        
        print(f"时域图已保存至: {output_fig_path}")
    
    def create_frequency_domain_plot(self, cutter, cut_num):
        """
        创建频域图
        
        参数:
            cutter: 刀具名称
            cut_num: 切削次数
        """
        # 加载三种类型的数据
        original_data = self.load_data(cutter, cut_num, 'original')
        processed_data = self.load_data(cutter, cut_num, 'processed')
        normalized_data = self.load_data(cutter, cut_num, 'normalized')
        
        if original_data is None or processed_data is None or normalized_data is None:
            print(f"无法为刀具 {cutter} 的切削次数 {cut_num} 创建频域图，数据不完整")
            return
        
        # 选择信号通道
        channels = ['Force_X', 'Force_Y', 'Force_Z', 
                   'Vibration_X', 'Vibration_Y', 'Vibration_Z', 
                   'AE_RMS']
        
        # 创建7x3的频域图（7个通道，3种信号类型）
        plt.figure(figsize=(18, 24))
        plt.suptitle(f'刀具 {cutter} - 切削次数 {cut_num} 的频域信号分析', fontsize=16)
        
        for i, channel in enumerate(channels):
            # 计算FFT
            # 原始数据的FFT
            original_fft = np.fft.rfft(original_data[channel].values)
            original_freq = np.fft.rfftfreq(len(original_data[channel].values), d=1/self.fs)
            
            # 处理后数据的FFT
            processed_fft = np.fft.rfft(processed_data[channel].values)
            processed_freq = np.fft.rfftfreq(len(processed_data[channel].values), d=1/self.fs)
            
            # 标准化数据的FFT
            normalized_fft = np.fft.rfft(normalized_data[channel].values)
            normalized_freq = np.fft.rfftfreq(len(normalized_data[channel].values), d=1/self.fs)
            
            # 归一化幅值
            original_amp = np.abs(original_fft) / len(original_data)
            processed_amp = np.abs(processed_fft) / len(processed_data)
            normalized_amp = np.abs(normalized_fft) / len(normalized_data)
            
            # 第一列：原始信号频谱
            ax1 = plt.subplot(7, 3, i*3+1)
            ax1.plot(original_freq, original_amp, 'b-')
            ax1.set_title(f'原始信号频谱 - {channel}')
            ax1.set_xlabel('频率 (Hz)')
            ax1.set_ylabel('幅值')
            ax1.grid(True)
            ax1.set_xlim(0, self.fs/2)
            
            # 第二列：处理后信号频谱
            ax2 = plt.subplot(7, 3, i*3+2)
            ax2.plot(processed_freq, processed_amp, 'g-')
            ax2.set_title(f'处理后信号频谱 - {channel}')
            ax2.set_xlabel('频率 (Hz)')
            ax2.set_ylabel('幅值')
            ax2.grid(True)
            ax2.set_xlim(0, self.fs/2)
            
            # 第三列：标准化信号频谱
            ax3 = plt.subplot(7, 3, i*3+3)
            ax3.plot(normalized_freq, normalized_amp, 'r-')
            ax3.set_title(f'标准化信号频谱 - {channel}')
            ax3.set_xlabel('频率 (Hz)')
            ax3.set_ylabel('幅值')
            ax3.grid(True)
            ax3.set_xlim(0, self.fs/2)
        
        # 紧凑布局
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 保存图表
        output_fig_path = os.path.join(self.output_path, f"{cutter}_cut_{cut_num:03d}_frequency_domain.png")
        plt.savefig(output_fig_path, dpi=300)
        plt.close()
        
        print(f"频域图已保存至: {output_fig_path}")
    
    def create_power_spectral_density_plot(self, cutter, cut_num):
        """
        创建功率谱密度图
        
        参数:
            cutter: 刀具名称
            cut_num: 切削次数
        """
        # 加载三种类型的数据
        original_data = self.load_data(cutter, cut_num, 'original')
        processed_data = self.load_data(cutter, cut_num, 'processed')
        normalized_data = self.load_data(cutter, cut_num, 'normalized')
        
        if original_data is None or processed_data is None or normalized_data is None:
            print(f"无法为刀具 {cutter} 的切削次数 {cut_num} 创建功率谱密度图，数据不完整")
            return
        
        # 选择信号通道
        channels = ['Force_X', 'Force_Y', 'Force_Z', 
                   'Vibration_X', 'Vibration_Y', 'Vibration_Z', 
                   'AE_RMS']
        
        # 创建7x3的功率谱密度图（7个通道，3种信号类型）
        plt.figure(figsize=(18, 24))
        plt.suptitle(f'刀具 {cutter} - 切削次数 {cut_num} 的功率谱密度分析', fontsize=16)
        
        for i, channel in enumerate(channels):
            # 计算功率谱密度
            # 计算原始数据的PSD
            f_original, psd_original = signal.welch(original_data[channel].values, self.fs, nperseg=1024)
            
            # 计算处理后数据的PSD
            f_processed, psd_processed = signal.welch(processed_data[channel].values, self.fs, nperseg=1024)
            
            # 计算标准化后数据的PSD
            f_normalized, psd_normalized = signal.welch(normalized_data[channel].values, self.fs, nperseg=1024)
            
            # 第一列：原始信号PSD
            ax1 = plt.subplot(7, 3, i*3+1)
            ax1.semilogy(f_original, psd_original, 'b-')
            ax1.set_title(f'原始信号PSD - {channel}')
            ax1.set_xlabel('频率 (Hz)')
            ax1.set_ylabel('功率/频率 (dB/Hz)')
            ax1.grid(True)
            
            # 第二列：处理后信号PSD
            ax2 = plt.subplot(7, 3, i*3+2)
            ax2.semilogy(f_processed, psd_processed, 'g-')
            ax2.set_title(f'处理后信号PSD - {channel}')
            ax2.set_xlabel('频率 (Hz)')
            ax2.set_ylabel('功率/频率 (dB/Hz)')
            ax2.grid(True)
            
            # 第三列：标准化信号PSD
            ax3 = plt.subplot(7, 3, i*3+3)
            ax3.semilogy(f_normalized, psd_normalized, 'r-')
            ax3.set_title(f'标准化信号PSD - {channel}')
            ax3.set_xlabel('频率 (Hz)')
            ax3.set_ylabel('功率/频率 (dB/Hz)')
            ax3.grid(True)
        
        # 紧凑布局
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 保存图表
        output_fig_path = os.path.join(self.output_path, f"{cutter}_cut_{cut_num:03d}_power_spectral_density.png")
        plt.savefig(output_fig_path, dpi=300)
        plt.close()
        
        print(f"功率谱密度图已保存至: {output_fig_path}")
    
    def create_wear_curve_plot(self, cutter='c1'):
        """
        创建磨损曲线图
        
        参数:
            cutter: 刀具名称，默认为'c1'
        """
        # 加载磨损数据
        wear_data = self.load_wear_data(cutter)
        if wear_data is None:
            print(f"无法为刀具 {cutter} 创建磨损曲线图，磨损数据不完整")
            return
        
        # 输出列名，便于调试
        print(f"磨损数据列名: {wear_data.columns.tolist()}")
        
        # 确定磨损列名
        wear_column = None
        if 'flank_wear' in wear_data.columns:
            wear_column = 'flank_wear'
        elif 'flank' in wear_data.columns:
            wear_column = 'flank'
        elif 'wear' in wear_data.columns:
            wear_column = 'wear'
        else:
            # 假设最后一列是磨损数据
            wear_column = wear_data.columns[-1]
            print(f"未找到标准磨损列名，使用最后一列: {wear_column}")
        
        # 确定切削次数列名
        cut_column = None
        if 'cut' in wear_data.columns:
            cut_column = 'cut'
        elif 'cut_no' in wear_data.columns:
            cut_column = 'cut_no'
        else:
            # 假设第一列是切削次数
            cut_column = wear_data.columns[0]
            print(f"未找到标准切削次数列名，使用第一列: {cut_column}")
        
        # 创建磨损曲线图
        plt.figure(figsize=(12, 8))
        plt.plot(wear_data[cut_column], wear_data[wear_column], 'b-', linewidth=2)
        plt.title(f'刀具 {cutter} 的磨损曲线')
        plt.xlabel('切削次数')
        plt.ylabel('刀具磨损量 (μm)')
        plt.grid(True)
        
        # 添加磨损阈值线
        plt.axhline(y=170, color='r', linestyle='--', label='磨损阈值 (170μm)')
        
        # 不再添加数据标签
        
        plt.legend()
        plt.tight_layout()
        
        # 保存图表
        output_fig_path = os.path.join(self.output_path, f"{cutter}_wear_curve.png")
        plt.savefig(output_fig_path, dpi=300)
        plt.close()
        
        print(f"磨损曲线图已保存至: {output_fig_path}")
        
        # 创建原始信号与磨损量关系图
        self.create_signal_wear_relationship_plot(cutter, cut_column, wear_column)
    
    def create_signal_wear_relationship_plot(self, cutter='c1', cut_column='cut', wear_column='flank_wear'):
        """
        创建原始信号特征与磨损量关系图
        
        参数:
            cutter: 刀具名称，默认为'c1'
            cut_column: 切削次数列名
            wear_column: 磨损数据列名
        """
        # 加载磨损数据
        wear_data = self.load_wear_data(cutter)
        if wear_data is None:
            print(f"无法为刀具 {cutter} 创建信号-磨损关系图，磨损数据不完整")
            return
        
        # 选择信号通道
        channels = ['Force_X', 'Force_Y', 'Force_Z', 
                   'Vibration_X', 'Vibration_Y', 'Vibration_Z', 
                   'AE_RMS']
        
        # 存储每个通道的RMS值
        channel_rms_values = {}
        
        # 计算每个切削次数的信号RMS值
        for _, row in wear_data.iterrows():
            cut_num = row[cut_column]
            # 加载原始信号数据
            original_data = self.load_data(cutter, cut_num, 'original')
            if original_data is not None:
                # 计算每个通道的RMS值
                for channel in channels:
                    if channel not in channel_rms_values:
                        channel_rms_values[channel] = []
                    # 计算RMS值
                    rms_value = np.sqrt(np.mean(np.square(original_data[channel].values)))
                    channel_rms_values[channel].append((cut_num, rms_value))
        
        # 创建信号特征与磨损量关系图 (2行4列)
        plt.figure(figsize=(20, 12))
        plt.suptitle(f'刀具 {cutter} 的原始信号特征与磨损量关系', fontsize=16)
        
        # 绘制每个通道的信号RMS值与磨损量的关系
        for i, channel in enumerate(channels):
            ax = plt.subplot(2, 4, i+1)
            
            if channel in channel_rms_values and len(channel_rms_values[channel]) > 0:
                # 提取数据
                cut_nums = [x[0] for x in channel_rms_values[channel]]
                rms_values = [x[1] for x in channel_rms_values[channel]]
                
                # 获取对应的磨损值
                wear_values = []
                for cut_num in cut_nums:
                    wear = wear_data[wear_data[cut_column] == cut_num][wear_column].values
                    if len(wear) > 0:
                        wear_values.append(wear[0])
                    else:
                        wear_values.append(np.nan)
                
                # 绘制RMS值与磨损量的散点图
                scatter = ax.scatter(rms_values, wear_values, c=cut_nums, cmap='viridis', 
                                    alpha=0.8, s=80)
                
                # 添加线性拟合
                if len(rms_values) > 1:
                    # 移除NaN值
                    valid_idx = ~np.isnan(wear_values)
                    valid_rms = np.array(rms_values)[valid_idx]
                    valid_wear = np.array(wear_values)[valid_idx]
                    
                    if len(valid_rms) > 1:
                        # 线性拟合
                        z = np.polyfit(valid_rms, valid_wear, 1)
                        p = np.poly1d(z)
                        
                        # 计算相关系数
                        corr = np.corrcoef(valid_rms, valid_wear)[0, 1]
                        
                        # 绘制拟合线
                        x_range = np.linspace(min(valid_rms), max(valid_rms), 100)
                        ax.plot(x_range, p(x_range), 'r--', 
                                label=f'拟合线: y={z[0]:.2f}x+{z[1]:.2f}\nR={corr:.2f}')
                        ax.legend(fontsize=8)
            
            ax.set_title(f'{channel} RMS值与磨损量关系')
            ax.set_xlabel(f'{channel} RMS值')
            ax.set_ylabel('刀具磨损量 (μm)')
            ax.grid(True)
        
        # 添加颜色条
        cbar_ax = plt.subplot(2, 4, 8)
        plt.colorbar(scatter, cax=cbar_ax)
        cbar_ax.set_title('切削次数')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 保存图表
        output_fig_path = os.path.join(self.output_path, f"{cutter}_signal_wear_relationship.png")
        plt.savefig(output_fig_path, dpi=300)
        plt.close()
        
        print(f"信号-磨损关系图已保存至: {output_fig_path}")
    
    def visualize_cutter_data(self, cutter, cut_nums=None):
        """
        为指定刀具的所有切削次数生成可视化图表
        
        参数:
            cutter: 刀具名称
            cut_nums: 切削次数列表，如果为None则尝试获取所有可用的切削次数
        """
        # 如果未指定切削次数，尝试获取所有可用的切削次数
        if cut_nums is None:
            # 加载磨损数据中的切削次数
            wear_file = os.path.join(self.data_path, 'processed', cutter, f"{cutter}_wear_processed.csv")
            try:
                wear_data = pd.read_csv(wear_file)
                cut_nums = wear_data['cut'].tolist()
            except Exception as e:
                print(f"加载磨损数据失败: {e}")
                print("尝试搜索处理后的数据文件...")
                
                # 搜索处理后的数据文件
                processed_files = os.listdir(os.path.join(self.data_path, 'processed', cutter))
                cut_nums = []
                for file in processed_files:
                    if file.startswith(f"c_{cutter[-1]}_") and file.endswith("_processed.csv"):
                        cut_num = int(file.split('_')[2])
                        cut_nums.append(cut_num)
                
                if not cut_nums:
                    print(f"无法获取刀具 {cutter} 的切削次数")
                    return
        
        # 为每个切削次数生成可视化图表
        for cut_num in tqdm(cut_nums, desc=f"生成刀具 {cutter} 的可视化图表"):
            # 创建时域图
            self.create_time_domain_plot(cutter, cut_num)
            
            # 创建频域图
            self.create_frequency_domain_plot(cutter, cut_num)
            
            # 创建功率谱密度图
            self.create_power_spectral_density_plot(cutter, cut_num)
        
        # 创建磨损曲线图 (仅针对c1刀具)
        if cutter == 'c1':
            self.create_wear_curve_plot(cutter)
    
    def visualize_all_data(self, cutters=None, cut_nums=None):
        """
        为所有刀具生成可视化图表
        
        参数:
            cutters: 刀具列表，如果为None则使用默认的刀具列表
            cut_nums: 切削次数列表，如果为None则尝试获取所有可用的切削次数
        """
        if cutters is None:
            cutters = ['c1', 'c4', 'c6']
        
        for cutter in cutters:
            self.visualize_cutter_data(cutter, cut_nums)


def main():
    """主函数"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='铣刀寿命预测系统 - 数据可视化模块')
    
    # 添加命令行参数
    parser.add_argument('--data_path', type=str, default='/Users/xiaohudemac/cursor01/bishe/5_8tool/data',
                        help='数据路径')
    parser.add_argument('--output_path', type=str, default='/Users/xiaohudemac/cursor01/bishe/5_8tool/visualizations',
                        help='可视化图表输出路径')
    parser.add_argument('--cutters', type=str, default='c1,c4,c6',
                        help='要处理的刀具列表，用逗号分隔')
    parser.add_argument('--cut_nums', type=str, default=None,
                        help='要处理的切削次数列表，用逗号分隔，如果不指定则处理所有可用的切削次数')
    parser.add_argument('--fs', type=int, default=1000,
                        help='采样频率 (Hz)')
    parser.add_argument('--single_cut', type=int, default=None,
                        help='只处理单个切削次数')
    parser.add_argument('--wear_curve', action='store_true',
                        help='只生成c1刀具的磨损曲线图')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 初始化信号可视化器
    visualizer = SignalVisualizer(
        data_path=args.data_path,
        output_path=args.output_path,
        fs=args.fs
    )
    
    # 只生成c1刀具的磨损曲线图
    if args.wear_curve:
        visualizer.create_wear_curve_plot('c1')
        print("磨损曲线图生成完成！")
        return
    
    # 解析刀具列表
    cutters = args.cutters.split(',')
    
    # 解析切削次数列表
    cut_nums = None
    if args.cut_nums is not None:
        cut_nums = [int(x) for x in args.cut_nums.split(',')]
    elif args.single_cut is not None:
        cut_nums = [args.single_cut]
    
    # 生成可视化图表
    visualizer.visualize_all_data(cutters, cut_nums)
    
    print("数据可视化完成！")


if __name__ == "__main__":
    main() 