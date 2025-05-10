#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
铣刀寿命预测系统 - 预测脚本
功能：加载最佳模型进行预测
"""

import os
import numpy as np
import pandas as pd
import joblib
import json
import argparse
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ToolWearPredictor:
    def __init__(self, model_path='../results/best_model', feature_type='selected'):
        """
        初始化工具磨损预测器
        
        参数:
            model_path: 模型目录路径
            feature_type: 特征类型，'selected'或'pca'
        """
        self.model_path = model_path
        self.feature_type = feature_type
        self.model = None
        self.model_summary = None
        self.load_model()
    
    def load_model(self):
        """加载模型和模型摘要"""
        try:
            # 加载模型
            model_file = os.path.join(self.model_path, 'random_forest_model.joblib')
            self.model = joblib.load(model_file)
            logger.info(f"成功加载模型: {model_file}")
            
            # 加载模型摘要
            summary_file = os.path.join(self.model_path, 'model_summary.json')
            if os.path.exists(summary_file):
                with open(summary_file, 'r', encoding='utf-8') as f:
                    self.model_summary = json.load(f)
                logger.info(f"成功加载模型摘要: {summary_file}")
                
                # 显示模型信息
                logger.info(f"模型类型: {self.model_summary['model_name']}")
                logger.info(f"模型性能 - R²: {self.model_summary['metrics']['r2_score']:.4f}, "
                           f"RMSE: {self.model_summary['metrics']['rmse']:.4f}")
                
                if self.model_summary['top_features']:
                    logger.info("重要特征:")
                    for i, feature in enumerate(self.model_summary['top_features'][:5], 1):
                        logger.info(f"{i}. {feature}")
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
    
    def predict(self, X):
        """
        预测磨损值
        
        参数:
            X: 输入特征，可以是文件路径或数据矩阵
            
        返回:
            预测的磨损值
        """
        if isinstance(X, str):
            # 从文件加载特征
            if X.endswith('.csv'):
                data = pd.read_csv(X)
                
                # 根据特征类型确定要使用的特征列
                if self.feature_type == 'selected':
                    # 特征选择数据
                    if 'wear_VB_avg' in data.columns:
                        y_true = data['wear_VB_avg'].values
                    else:
                        y_true = None
                    
                    # 排除磨损相关列和切削次数列
                    exclude_cols = ['cut_num'] + [col for col in data.columns if col.startswith('wear_VB')]
                    feature_cols = [col for col in data.columns if col not in exclude_cols]
                    X = data[feature_cols].values
                    
                elif self.feature_type == 'pca':
                    # PCA降维数据
                    if 'wear_VB_avg' in data.columns:
                        y_true = data['wear_VB_avg'].values
                    else:
                        y_true = None
                    
                    # 使用PC特征
                    pc_cols = [col for col in data.columns if col.startswith('PC')]
                    X = data[pc_cols].values
                
                logger.info(f"从文件加载特征: {X.shape}")
            else:
                raise ValueError(f"不支持的文件格式: {X}")
        elif isinstance(X, np.ndarray) or isinstance(X, pd.DataFrame):
            # 直接使用提供的特征矩阵
            if isinstance(X, pd.DataFrame):
                X = X.values
            
            logger.info(f"使用提供的特征矩阵: {X.shape}")
            y_true = None
        else:
            raise ValueError(f"不支持的输入类型: {type(X)}")
        
        # 进行预测
        if self.model is None:
            raise ValueError("模型未加载")
        
        y_pred = self.model.predict(X)
        logger.info(f"预测完成，预测样本数: {len(y_pred)}")
        
        return y_pred, y_true
    
    def visualize_prediction(self, y_pred, y_true=None, output_path=None):
        """
        可视化预测结果
        
        参数:
            y_pred: 预测的磨损值
            y_true: 真实的磨损值，可选
            output_path: 输出路径，可选
        """
        plt.figure(figsize=(10, 6))
        
        # 绘制预测序列
        plt.plot(range(len(y_pred)), y_pred, 'b-', marker='o', markersize=4, 
                label='预测磨损值', linewidth=2)
        
        # 如果有真实值，也绘制出来
        if y_true is not None:
            plt.plot(range(len(y_true)), y_true, 'r-', marker='s', markersize=4, 
                    label='实际磨损值', linewidth=2)
            
            # 计算评估指标
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            plt.title(f'磨损值预测 (R²: {r2:.4f}, RMSE: {rmse:.4f})')
            
            # 添加评估指标文本
            textstr = f'R²: {r2:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            plt.annotate(textstr, xy=(0.05, 0.95), xycoords='axes fraction', 
                        fontsize=10, verticalalignment='top', bbox=props)
        else:
            plt.title('磨损值预测')
        
        plt.xlabel('切削次数')
        plt.ylabel('磨损值 (mm)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        
        # 保存或显示图像
        if output_path:
            plt.savefig(output_path, dpi=300)
            logger.info(f"预测可视化结果已保存至: {output_path}")
        
        return plt

def main():
    """主函数"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='铣刀寿命预测系统 - 预测脚本')
    
    # 添加命令行参数
    parser.add_argument('--model_path', type=str, default='../results/best_model',
                        help='模型目录路径')
    parser.add_argument('--feature_type', type=str, choices=['selected', 'pca'], default='selected',
                        help='特征类型，selected或pca')
    parser.add_argument('--input_file', type=str, required=True,
                        help='输入特征文件路径（CSV格式）')
    parser.add_argument('--output_path', type=str, default='../results/predictions',
                        help='预测结果输出路径')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 确保输出目录存在
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    # 初始化预测器
    predictor = ToolWearPredictor(model_path=args.model_path, feature_type=args.feature_type)
    
    # 加载数据并预测
    try:
        y_pred, y_true = predictor.predict(args.input_file)
        
        # 获取输入文件名（不含扩展名）
        input_basename = os.path.splitext(os.path.basename(args.input_file))[0]
        
        # 保存预测结果
        results_file = os.path.join(args.output_path, f'{input_basename}_predictions.csv')
        
        # 创建结果DataFrame
        if y_true is not None:
            results_df = pd.DataFrame({
                'actual': y_true,
                'predicted': y_pred,
                'error': y_true - y_pred
            })
        else:
            results_df = pd.DataFrame({
                'predicted': y_pred
            })
        
        results_df.to_csv(results_file, index_label='sample_id')
        logger.info(f"预测结果已保存至: {results_file}")
        
        # 生成可视化
        vis_file = os.path.join(args.output_path, f'{input_basename}_visualization.png')
        predictor.visualize_prediction(y_pred, y_true, output_path=vis_file)
        
        # 如果有真实值，计算并保存评估指标
        if y_true is not None:
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            metrics = {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2)
            }
            
            metrics_file = os.path.join(args.output_path, f'{input_basename}_metrics.json')
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            logger.info(f"评估指标已保存至: {metrics_file}")
            logger.info(f"MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}")
        
    except Exception as e:
        logger.error(f"预测失败: {e}")
        raise

if __name__ == "__main__":
    main() 