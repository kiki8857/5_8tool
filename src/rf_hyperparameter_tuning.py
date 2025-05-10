#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
铣刀寿命预测系统 - 随机森林超参数调优
功能：寻找随机森林模型的最佳超参数组合
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json
import logging
import argparse
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
import time
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RFHyperparameterTuner:
    def __init__(self, selected_features_path, output_path, target_column='wear_VB_avg',
                 train_cutters=None, test_cutter='c6', random_state=42):
        """
        初始化随机森林超参数调优器
        
        参数:
            selected_features_path: 特征选择后的数据路径
            output_path: 输出路径
            target_column: 目标变量列名
            train_cutters: 用于训练的刀具列表，默认为None（使用非测试刀具）
            test_cutter: 用于测试的刀具
            random_state: 随机种子
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
        
        # 确保输出目录存在
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # 参数网格 - 使用更小的搜索范围
        self.param_grid = {
            'n_estimators': [100, 200, 300],      # 减少树的数量选项
            'max_depth': [None, 20, 40],          # 减少最大深度选项
            'min_samples_split': [2, 5, 10],      # 减少内部节点分裂选项
            'min_samples_leaf': [1, 4, 8],        # 减少叶节点样本数选项
            'max_features': ['sqrt', 'log2'],     # 只使用常用的特征选择方式
            'bootstrap': [True]                   # 默认使用bootstrap抽样
        }
    
    def load_data(self):
        """
        加载训练和测试数据
        
        返回:
            X_train: 训练特征
            y_train: 训练目标
            X_test: 测试特征
            y_test: 测试目标
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
        # 排除磨损相关列和切削次数列
        exclude_cols = ['cut_num'] + [col for col in train_df.columns if col.startswith('wear_VB')]
        train_feature_cols = [col for col in train_df.columns if col not in exclude_cols]
        test_feature_cols = [col for col in test_df.columns if col not in exclude_cols]
        
        # 找出训练集和测试集的共同特征列
        common_features = [col for col in train_feature_cols if col in test_feature_cols]
        
        # 记录特征差异
        train_only_features = [col for col in train_feature_cols if col not in common_features]
        test_only_features = [col for col in test_feature_cols if col not in common_features]
        
        logger.info(f"训练集和测试集共有 {len(common_features)} 个共同特征，将用于超参数调优")
        
        if train_only_features:
            logger.warning(f"训练集独有的 {len(train_only_features)} 个特征将被忽略: {train_only_features}")
        
        if test_only_features:
            logger.warning(f"测试集独有的 {len(test_only_features)} 个特征将被忽略: {test_only_features}")
        
        # 提取特征和目标变量，确保使用相同的特征列
        X_train = train_df[common_features].values
        y_train = train_df[self.target_column].values
        
        X_test = test_df[common_features].values
        y_test = test_df[self.target_column].values
        
        logger.info(f"数据准备完成，使用特征数量: {X_train.shape[1]}")
        logger.info(f"使用的特征列: {common_features}")
        
        return X_train, y_train, X_test, y_test
    
    def tune_hyperparameters(self, X_train, y_train):
        """
        调整随机森林超参数
        
        参数:
            X_train: 训练特征
            y_train: 训练目标
            
        返回:
            best_params: 最佳超参数
            cv_results: 所有参数组合的结果
        """
        logger.info("开始超参数调优...")
        start_time = time.time()
        
        # 创建基础随机森林模型
        base_model = RandomForestRegressor(random_state=self.random_state)
        
        # The tune_hyperparameters method with 5-fold cross-validation
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=self.param_grid,
            cv=5,                      # 5折交叉验证
            scoring='r2',              # 使用R²评分
            n_jobs=-1,                 # 使用所有可用CPU
            verbose=2,                 # 增加详细程度以显示更多进度信息
            return_train_score=True    # 同时返回训练集评分
        )
        
        grid_search.fit(X_train, y_train)
        
        # 获取最佳参数
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        end_time = time.time()
        tuning_time = end_time - start_time
        
        logger.info(f"超参数调优完成，耗时: {tuning_time:.2f}秒")
        logger.info(f"最佳超参数: {best_params}")
        logger.info(f"最佳交叉验证R²: {best_score:.6f}")
        
        return best_params, grid_search.cv_results_
    
    def evaluate_best_model(self, X_train, y_train, X_test, y_test, best_params):
        """
        评估使用最佳参数的模型
        
        参数:
            X_train: 训练特征
            y_train: 训练目标
            X_test: 测试特征
            y_test: 测试目标
            best_params: 最佳超参数
            
        返回:
            metrics: 评估指标
        """
        logger.info("评估最佳模型...")
        
        # 使用最佳参数创建模型
        model = RandomForestRegressor(random_state=self.random_state, **best_params)
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 在测试集上进行预测
        y_pred = model.predict(X_test)
        
        # 计算评估指标
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # 记录评估结果
        logger.info(f"测试集评估结果 - MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}")
        
        # 保存评估指标
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2)
        }
        
        return metrics
    
    def visualize_param_scores(self, cv_results, param_name):
        """
        可视化单个参数的得分
        
        参数:
            cv_results: 交叉验证结果
            param_name: 参数名称
        """
        # 提取参数值和对应的得分
        params = cv_results[f'param_{param_name}'].data.astype(str)
        mean_train_scores = cv_results['mean_train_score']
        mean_test_scores = cv_results['mean_test_score']
        
        # 获取唯一的参数值
        unique_params = np.unique(params)
        
        # 计算每个参数值的平均得分
        param_train_scores = []
        param_test_scores = []
        
        for param in unique_params:
            mask = params == param
            param_train_scores.append(np.mean(mean_train_scores[mask]))
            param_test_scores.append(np.mean(mean_test_scores[mask]))
        
        # 绘制图表
        plt.figure(figsize=(10, 6))
        
        x = range(len(unique_params))
        plt.plot(x, param_train_scores, 'o-', label='训练集R²', color='blue')
        plt.plot(x, param_test_scores, 's-', label='验证集R²', color='red')
        
        plt.xlabel(param_name)
        plt.ylabel('R²得分')
        plt.title(f'{param_name}参数对模型性能的影响')
        plt.xticks(x, unique_params)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(os.path.join(self.output_path, f'param_{param_name}_scores.png'))
        plt.close()
    
    def save_results(self, best_params, metrics, cv_results):
        """
        保存调优结果
        
        参数:
            best_params: 最佳超参数
            metrics: 评估指标
            cv_results: 交叉验证结果
        """
        # 保存最佳参数
        best_params_file = os.path.join(self.output_path, 'best_params.json')
        
        with open(best_params_file, 'w') as f:
            json.dump(best_params, f, indent=4)
        
        logger.info(f"最佳超参数已保存至: {best_params_file}")
        
        # 保存评估指标
        metrics_file = os.path.join(self.output_path, 'best_model_metrics.json')
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"评估指标已保存至: {metrics_file}")
        
        # 保存所有参数组合的结果
        results_df = pd.DataFrame()
        
        # 添加参数列
        for param_name in self.param_grid.keys():
            results_df[param_name] = cv_results[f'param_{param_name}'].data.astype(str)
        
        # 添加评分列
        results_df['mean_train_score'] = cv_results['mean_train_score']
        results_df['mean_test_score'] = cv_results['mean_test_score']
        results_df['std_test_score'] = cv_results['std_test_score']
        results_df['rank_test_score'] = cv_results['rank_test_score']
        
        # 按测试集得分排序
        results_df = results_df.sort_values('mean_test_score', ascending=False)
        
        # 保存结果
        results_file = os.path.join(self.output_path, 'all_param_results.csv')
        results_df.to_csv(results_file, index=False)
        
        logger.info(f"所有参数组合的结果已保存至: {results_file}")
    
    def run(self):
        """
        运行超参数调优
        """
        # 加载数据
        X_train, y_train, X_test, y_test = self.load_data()
        
        # 调优超参数
        best_params, cv_results = self.tune_hyperparameters(X_train, y_train)
        
        # 评估最佳模型
        metrics = self.evaluate_best_model(X_train, y_train, X_test, y_test, best_params)
        
        # 可视化参数得分
        for param_name in self.param_grid.keys():
            self.visualize_param_scores(cv_results, param_name)
        
        # 保存结果
        self.save_results(best_params, metrics, cv_results)
        
        logger.info("随机森林超参数调优完成！")
        
        return best_params, metrics

def main():
    """主函数"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='铣刀寿命预测系统 - 随机森林超参数调优')
    
    # 添加命令行参数
    parser.add_argument('--features_path', type=str, default='/Users/xiaohudemac/cursor01/bishe/5_8tool/data/selected_features',
                        help='特征选择后的数据路径')
    parser.add_argument('--output_path', type=str, default='/Users/xiaohudemac/cursor01/bishe/5_8tool/results/rf_tuning',
                        help='调优输出根路径')
    parser.add_argument('--target_column', type=str, default='wear_VB_avg',
                        help='目标变量列名')
    parser.add_argument('--train_cutters', type=str, default='c1,c4',
                        help='用于训练的刀具列表，用逗号分隔')
    parser.add_argument('--test_cutter', type=str, default='c6',
                        help='用于测试的刀具')
    parser.add_argument('--random_state', type=int, default=42,
                        help='随机种子')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 解析训练刀具列表
    train_cutters = args.train_cutters.split(',')
    
    # 为输出路径添加时间戳，确保每次运行结果保存在单独的文件夹中
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_path, f"run_{timestamp}")
    
    # 初始化随机森林超参数调优器
    tuner = RFHyperparameterTuner(
        selected_features_path=args.features_path,
        output_path=output_path,
        target_column=args.target_column,
        train_cutters=train_cutters,
        test_cutter=args.test_cutter,
        random_state=args.random_state
    )
    
    # 运行超参数调优
    tuner.run()

if __name__ == "__main__":
    main() 