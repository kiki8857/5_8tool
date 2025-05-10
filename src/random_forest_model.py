#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
铣刀寿命预测系统 - 随机森林模型
功能：使用随机森林算法预测刀具磨损
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json
import logging
import argparse
import joblib
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RandomForestModel:
    def __init__(self, selected_features_path, output_path, target_column='wear_VB_avg',
                 train_cutters=None, test_cutter='c6', random_state=42, use_pca=False):
        """
        初始化随机森林模型
        
        参数:
            selected_features_path: 特征选择后的数据路径
            output_path: 模型输出路径
            target_column: 目标变量列名
            train_cutters: 用于训练的刀具列表，默认为None（使用非测试刀具）
            test_cutter: 用于测试的刀具
            random_state: 随机种子
            use_pca: 是否使用PCA降维后的数据
        """
        self.selected_features_path = selected_features_path
        self.output_path = output_path
        self.target_column = target_column
        self.test_cutter = test_cutter
        self.random_state = random_state
        self.use_pca = use_pca
        
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
    
    def load_data(self):
        """
        加载训练和测试数据
        
        返回:
            X_train: 训练特征
            y_train: 训练目标
            X_test: 测试特征
            y_test: 测试目标
            test_cut_nums: 测试数据的切削次数
        """
        logger.info("准备数据...")
        
        # 根据是否使用PCA降维后的数据决定文件名模式
        file_suffix = "pca_data.csv" if self.use_pca else "selected_feature_data.csv"
        logger.info(f"使用数据类型: {'PCA降维数据' if self.use_pca else '特征选择数据'}")
        
        # 加载训练数据
        train_data = []
        for cutter in self.train_cutters:
            data_file = os.path.join(self.selected_features_path, cutter, f"{cutter}_{file_suffix}")
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
                                      f"{self.test_cutter}_{file_suffix}")
        try:
            test_df = pd.read_csv(test_data_file)
            logger.info(f"成功加载{self.test_cutter}数据: {test_data_file}")
        except Exception as e:
            logger.error(f"加载{self.test_cutter}数据失败: {e}")
            raise ValueError(f"未能加载测试数据: {e}")
        
        # 提取特征和目标变量
        feature_cols = [col for col in train_df.columns if col != self.target_column and col != 'cut_num']
        X_train = train_df[feature_cols]
        y_train = train_df[self.target_column]
        
        X_test = test_df[feature_cols]
        y_test = test_df[self.target_column]
        test_cut_nums = test_df['cut_num']
        
        logger.info(f"数据准备完成，输入特征数量: {X_train.shape[1]}")
        logger.info(f"训练数据形状: {X_train.shape}, 测试数据形状: {X_test.shape}")
        logger.info(f"使用的特征: {', '.join(feature_cols)}")
        
        return X_train, y_train, X_test, y_test, test_cut_nums
    
    def train(self, X_train, y_train):
        """
        训练随机森林模型
        
        参数:
            X_train: 训练特征
            y_train: 训练目标
            
        返回:
            model: 训练好的模型
        """
        logger.info("构建模型...")
        
        # 构建模型，使用调优后的最佳参数
        self.model = RandomForestRegressor(
            n_estimators=200,  # 树的数量
            max_depth=20,  # 树的最大深度，最佳参数为20
            min_samples_split=10,  # 内部节点分裂所需的最小样本数，最佳参数为10
            min_samples_leaf=1,  # 叶节点所需的最小样本数，最佳参数为1
            max_features='sqrt',  # 寻找最佳分裂点时考虑的特征数，最佳参数为'sqrt'
            bootstrap=True,  # 是否使用bootstrap抽样，最佳参数为True
            random_state=self.random_state
        )
        
        logger.info(f"模型构建完成: {self.model}")
        
        # 训练模型
        logger.info("开始训练模型...")
        self.model.fit(X_train, y_train)
        logger.info("模型训练完成")
        
        # 保存模型
        model_file = os.path.join(self.output_path, 'random_forest_model.joblib')
        joblib.dump(self.model, model_file)
        logger.info(f"模型已保存至: {model_file}")
        
        # 获取特征重要性
        self.feature_importance = self.model.feature_importances_
        
        return self.model
    
    def evaluate_model(self, X_test, y_test, test_cut_nums):
        """
        评估模型
        
        参数:
            X_test: 测试特征
            y_test: 测试目标
            test_cut_nums: 测试数据的切削次数
            
        返回:
            evaluation_metrics: 评估指标字典
        """
        logger.info("评估模型...")
        
        # 预测
        y_pred = self.model.predict(X_test)
        
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
        print("随机森林模型评估指标")
        print("="*50)
        print(f"测试刀具: {self.test_cutter}")
        print(f"训练刀具: {', '.join(self.train_cutters)}")
        print(f"均方误差(MSE): {mse:.6f}")
        print(f"均方根误差(RMSE): {rmse:.6f}")
        print(f"平均绝对误差(MAE): {mae:.6f}")
        print(f"决定系数(R²): {r2:.6f}")
        print("="*50)
        
        # 获取特征重要性
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = self.model.feature_importances_
            features = X_test.columns
            
            # 创建特征重要性数据框
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': feature_importance
            })
            
            # 按重要性排序
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            # 保存特征重要性
            importance_df.to_csv(os.path.join(self.output_path, 'feature_importance.csv'), index=False)
            
            # 打印特征重要性
            print("\n特征重要性:")
            print("-"*50)
            for i, (feature, importance) in enumerate(zip(importance_df['Feature'], importance_df['Importance'])):
                print(f"{i+1}. {feature:<40} {importance:.6f}")
            print("-"*50)
            
            # 绘制特征重要性图
            plt.figure(figsize=(12, 10))
            top_n = 15  # 显示最重要的15个特征
            sorted_idx = importance_df['Importance'].argsort()[::-1][:top_n]
            top_features = importance_df.iloc[sorted_idx]
            
            plt.barh(range(top_n), top_features['Importance'], align='center', color='skyblue')
            plt.yticks(range(top_n), top_features['Feature'], fontsize=12)
            plt.title('随机森林模型特征重要性分析', fontsize=16)
            plt.xlabel('相对重要性', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, 'feature_importance.png'), dpi=300)
            plt.close()
        
        # 可视化预测结果
        self.visualize_predictions(y_test, y_pred, test_cut_nums)
        
        return evaluation_metrics
    
    def visualize_predictions(self, y_test, y_pred, test_cut_nums):
        """
        可视化预测结果
        
        参数:
            y_test: 测试集真实值
            y_pred: 预测值
            test_cut_nums: (pd.Series): 测试集的切削次数
        """
        # 创建预测对比图
        plt.figure(figsize=(12, 8))
        plt.scatter(y_test, y_pred, alpha=0.7, color='blue', s=60)
        
        # 添加理想预测线
        min_val = min(min(y_test), min(y_pred))
        max_val = max(max(y_test), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        # 计算指标显示在图上
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        plt.text(min_val + 0.05*(max_val-min_val), max_val - 0.15*(max_val-min_val), 
                f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}', 
                fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.title(f'刀具 {self.test_cutter} 磨损预测结果对比图 (随机森林模型)', fontsize=16)
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
        plt.plot(test_cut_nums, y_test, 'b-', label='实际磨损值', linewidth=2)
        plt.plot(test_cut_nums, y_pred, 'r--', label='预测磨损值', linewidth=2)
        
        # 计算绝对误差并绘制误差条
        abs_error = np.abs(y_test - y_pred)
        plt.fill_between(test_cut_nums, y_pred - abs_error, y_pred + abs_error, color='gray', alpha=0.2, label='预测误差区间')
        
        plt.title(f'刀具 {self.test_cutter} 磨损预测随切削次数的变化 (随机森林模型)', fontsize=16)
        plt.xlabel('切削次数', fontsize=14)
        plt.ylabel('磨损值 (mm)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # 保存预测序列图
        plt.savefig(os.path.join(self.output_path, f'{self.test_cutter}_predictions.png'), dpi=300)
        plt.close('all')
    
    def run(self):
        """
        运行完整的模型训练和评估流程
        """
        try:
            # 加载数据
            X_train, y_train, X_test, y_test, test_cut_nums = self.load_data()
            
            # 训练模型
            self.train(X_train, y_train)
            
            # 评估模型
            metrics = self.evaluate_model(X_test, y_test, test_cut_nums)
            
            logger.info("随机森林模型训练和评估完成！")
            
            return metrics
        
        except Exception as e:
            logger.error(f"运行过程中发生错误: {e}")
            raise e

def main():
    """主函数"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='铣刀寿命预测系统 - 随机森林模型')
    
    # 添加命令行参数
    parser.add_argument('--features_path', type=str, default='/Users/xiaohudemac/cursor01/bishe/5_8tool/data/selected_features',
                        help='特征选择后的数据路径')
    parser.add_argument('--output_path', type=str, default='/Users/xiaohudemac/cursor01/bishe/5_8tool/results/random_forest',
                        help='模型输出根路径')
    parser.add_argument('--target_column', type=str, default='wear_VB_avg',
                        help='目标变量列名')
    parser.add_argument('--train_cutters', type=str, default='c1,c4',
                        help='用于训练的刀具列表，用逗号分隔')
    parser.add_argument('--test_cutter', type=str, default='c6',
                        help='用于测试的刀具')
    parser.add_argument('--random_state', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--use_pca', action='store_true',
                        help='是否使用PCA降维后的数据')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 解析训练刀具列表
    train_cutters = args.train_cutters.split(',')
    
    # 为输出路径添加时间戳和数据类型标识，确保每次运行结果保存在单独的文件夹中
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_path, f"run_{timestamp}")
    
    # 初始化随机森林模型
    rf_model = RandomForestModel(
        selected_features_path=args.features_path,
        output_path=output_path,
        target_column=args.target_column,
        train_cutters=train_cutters,
        test_cutter=args.test_cutter,
        random_state=args.random_state,
        use_pca=args.use_pca
    )
    
    # 训练并评估模型
    print("\n" + "="*60)
    print(f"开始使用刀具 {', '.join(train_cutters)} 的数据训练随机森林模型")
    print(f"测试集使用刀具 {args.test_cutter} 的数据")
    print(f"使用特征数据路径: {args.features_path}")
    print(f"模型结果将保存至: {output_path}")
    print("="*60)
    
    try:
        rf_model.run()
        print("\n训练和评估完成!")
        print(f"模型和评估结果已保存至: {output_path}")
        print("="*60)
    except Exception as e:
        print(f"\n训练过程中发生错误: {e}")
        print("="*60)

if __name__ == "__main__":
    main() 