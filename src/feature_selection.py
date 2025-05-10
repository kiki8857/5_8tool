#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
铣刀寿命预测系统 - 特征选择模块
功能：使用相关系数选择特征并用PCA降维
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import argparse
import logging
import json
from scipy.stats import pearsonr, spearmanr

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureSelector:
    def __init__(self, features_path, output_path, target_column='wear_VB_avg', n_features=40, n_pca_components=5, 
                 min_correlation=0.85, max_feature_correlation=0.98):
        """
        初始化特征选择器
        
        参数:
            features_path: 特征数据路径
            output_path: 输出路径
            target_column: 目标变量列名
            n_features: 通过相关系数初步选择的特征数量
            n_pca_components: PCA降维后的组件数量
            min_correlation: 至少要求的相关系数阈值（皮尔逊或斯皮尔曼之一达到此值）
            max_feature_correlation: 特征之间的最大允许相关系数，超过则视为冗余
        """
        self.features_path = features_path
        self.output_path = output_path
        self.target_column = target_column
        self.n_features = n_features
        self.n_pca_components = n_pca_components
        self.min_correlation = min_correlation
        self.max_feature_correlation = max_feature_correlation
        
        # 创建输出目录
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # 配置日志
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def select_features(self, cutter, use_normalized=False):
        """
        为单个刀具选择特征
        
        参数:
            cutter: 刀具名称
            use_normalized: 是否使用归一化特征
        
        返回:
            选择的特征数据DataFrame
        """
        self.logger.info(f"正在为刀具 {cutter} 选择特征...")
        
        # 确定特征文件路径
        features_file = os.path.join(self.features_path, cutter, f"{cutter}_{'normalized_features' if use_normalized else 'features'}.csv")
        try:
            # 加载特征数据
            features_data = pd.read_csv(features_file)
            self.logger.info(f"成功加载特征数据: {features_file}")
        except Exception as e:
            self.logger.error(f"加载特征数据失败: {e}")
            return None, None, None
        
        # 确保目标列存在
        if self.target_column not in features_data.columns:
            self.logger.error(f"目标列 {self.target_column} 不存在于特征数据中")
            return None, None, None
        
        # 分离特征和目标
        y = features_data[self.target_column]
        
        # 排除所有wear_VB开头的列和切削次数列
        wear_columns = [col for col in features_data.columns if col.startswith('wear_VB')]
        X = features_data.drop(columns=wear_columns + ['cut_num'])
        
        # 使用Pearson和Spearman相关系数选择特征
        self.logger.info(f"使用Pearson和Spearman相关系数选择前 {self.n_features} 个特征...")
        
        # 计算每个特征与目标的相关系数
        corr = []
        for col in X.columns:
            pearson_corr, _ = pearsonr(X[col], y)
            spearman_corr, _ = spearmanr(X[col], y)
            # 记录绝对相关系数和原始相关系数
            corr.append((col, abs(pearson_corr), pearson_corr, abs(spearman_corr), spearman_corr))
        
        # 根据绝对相关系数排序
        corr.sort(key=lambda x: max(x[1], x[3]), reverse=True)
        
        # 选择前N个特征，且至少满足一个相关系数达到阈值
        selected_features = []
        for col, abs_pearson, pearson, abs_spearman, spearman in corr:
            if max(abs_pearson, abs_spearman) >= self.min_correlation:
                selected_features.append((col, abs_pearson, pearson, abs_spearman, spearman))
                if len(selected_features) >= self.n_features:
                    break
        
        if len(selected_features) == 0:
            self.logger.warning(f"没有特征的相关系数达到 {self.min_correlation}，将使用前 {self.n_features} 个特征")
            selected_features = [(col, abs_pearson, pearson, abs_spearman, spearman) 
                                for col, abs_pearson, pearson, abs_spearman, spearman in corr[:self.n_features]]
        
        # 提取特征名称
        selected_feature_names = [col for col, _, _, _, _ in selected_features]
        
        # 打印选择的特征及其相关系数
        self.logger.info("选择的特征及其相关系数（按最大相关系数绝对值排序）:")
        print("\n特征选择结果:")
        print("-" * 100)
        print(f"{'特征名称':<50} {'皮尔逊相关系数':<15} {'斯皮尔曼相关系数':<15} {'达到阈值':<10}")
        print("-" * 100)
        for i, (col, abs_pearson, pearson, abs_spearman, spearman) in enumerate(selected_features):
            reaches_threshold = "是" if max(abs_pearson, abs_spearman) >= self.min_correlation else "否"
            print(f"{i+1}. {col:<48} {pearson:>+.4f} {spearman:>+15.4f} {reaches_threshold:>10}")
        print("-" * 100)
        
        # 应用特征选择
        X_selected = X[selected_feature_names]
        
        # 检查并移除高度相关的特征
        removed_features = self.remove_highly_correlated_features(X_selected, selected_feature_names)
        
        # 更新选择的特征列表
        final_features = [f for f in selected_feature_names if f not in removed_features]
        
        # 使用最终选择的特征
        X_selected = X[final_features]
        
        # 合并特征和目标
        selected_data = pd.concat([X_selected, y, features_data['cut_num']], axis=1)
        
        # 保存选择的特征数据
        cutter_output_dir = os.path.join(self.output_path, cutter)
        if not os.path.exists(cutter_output_dir):
            os.makedirs(cutter_output_dir)
        
        output_file = os.path.join(cutter_output_dir, f"{cutter}_selected_feature_data.csv")
        selected_data.to_csv(output_file, index=False)
        self.logger.info(f"选择的特征数据已保存至: {output_file}")
        
        # 保存特征选择记录（包括特征名称和相关系数）
        feature_record = pd.DataFrame([(col, abs_pearson, pearson, abs_spearman, spearman) 
                                       for col, abs_pearson, pearson, abs_spearman, spearman in corr],
                                     columns=['feature', 'abs_pearson_corr', 'pearson_corr', 'abs_spearman_corr', 'spearman_corr'])
        
        feature_record_file = os.path.join(cutter_output_dir, f"{cutter}_feature_correlation.csv")
        feature_record.to_csv(feature_record_file, index=False)
        self.logger.info(f"特征相关系数记录已保存至: {feature_record_file}")
        
        # 计算并保存最大相关系数
        max_pearson = max([abs(pearson) for _, _, pearson, _, _ in corr])
        max_spearman = max([abs(spearman) for _, _, _, _, spearman in corr])
        max_corr_info = {
            'max_pearson_abs': max_pearson,
            'max_spearman_abs': max_spearman,
            'max_pearson_feature': corr[0][0],
            'max_pearson_value': corr[0][2],
            'max_spearman_feature': sorted(corr, key=lambda x: x[3], reverse=True)[0][0],
            'max_spearman_value': sorted(corr, key=lambda x: x[3], reverse=True)[0][4]
        }
        
        with open(os.path.join(cutter_output_dir, f"{cutter}_max_correlation.json"), 'w') as f:
            json.dump(max_corr_info, f, indent=4)
        
        self.logger.info(f"最大皮尔逊相关系数: {max_pearson:.4f}, 最大斯皮尔曼相关系数: {max_spearman:.4f}")
        
        return selected_data, final_features, corr
    
    def remove_highly_correlated_features(self, X, features):
        """
        移除高度相关的特征
        
        参数:
            X: 特征数据DataFrame
            features: 特征名称列表
        
        返回:
            被移除的特征列表
        """
        self.logger.info(f"检查并移除高度相关的特征 (阈值 > {self.max_feature_correlation})...")
        
        # 计算特征之间的相关系数矩阵
        corr_matrix = X.corr().abs()
        
        # 创建上三角形矩阵的掩码
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # 找出相关系数高于阈值的特征对
        high_corr_pairs = []
        for col in upper.columns:
            # 找出与当前列高度相关的特征
            high_corr_features = upper.index[upper[col] > self.max_feature_correlation].tolist()
            for feature in high_corr_features:
                high_corr_pairs.append((col, feature, upper.loc[feature, col]))
        
        # 打印高度相关的特征对
        if high_corr_pairs:
            print("\n高度相关的特征对:")
            print("-" * 80)
            print(f"{'特征1':<30} {'特征2':<30} {'相关系数':<10}")
            print("-" * 80)
            for f1, f2, corr_val in high_corr_pairs:
                print(f"{f1:<30} {f2:<30} {corr_val:>10.4f}")
            print("-" * 80)
        
        # 贪婪算法选择要移除的特征
        # 优先移除与多个其他特征高度相关的特征
        to_drop = set()
        
        # 计算每个特征与其他特征高度相关的次数
        feature_counts = {}
        for f1, f2, _ in high_corr_pairs:
            feature_counts[f1] = feature_counts.get(f1, 0) + 1
            feature_counts[f2] = feature_counts.get(f2, 0) + 1
        
        # 按高度相关次数排序
        sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        
        # 贪婪选择要移除的特征
        for feature, _ in sorted_features:
            if feature in to_drop:
                continue
                
            # 找出与当前特征高度相关的特征
            for f1, f2, _ in high_corr_pairs:
                if f1 == feature and f2 not in to_drop:
                    to_drop.add(f2)
                elif f2 == feature and f1 not in to_drop:
                    to_drop.add(f1)
        
        to_drop = list(to_drop)
        
        if to_drop:
            self.logger.info(f"移除 {len(to_drop)} 个高度相关的特征: {', '.join(to_drop)}")
        else:
            self.logger.info("没有发现高度相关的特征")
        
        return to_drop
    
    def apply_pca(self, selected_data, cutter):
        """
        应用PCA降维
        
        参数:
            selected_data: 选择的特征数据DataFrame
            cutter: 刀具名称
        
        返回:
            PCA降维后的数据DataFrame
        """
        self.logger.info(f"对刀具 {cutter} 应用PCA降维...")
        
        # 分离特征和目标
        wear_columns = [col for col in selected_data.columns if col.startswith('wear_VB')]
        cut_num = selected_data['cut_num']
        X = selected_data.drop(columns=wear_columns + ['cut_num'])
        y = selected_data[self.target_column]
        
        # 标准化数据
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 应用PCA
        pca = PCA(n_components=self.n_pca_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # 打印PCA解释的方差比例
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        print("\nPCA降维结果:")
        print("-" * 60)
        print(f"主成分数量: {self.n_pca_components}")
        for i, var in enumerate(explained_variance):
            print(f"PC{i+1}: 解释方差比例 = {var:.4f}, 累积方差比例 = {cumulative_variance[i]:.4f}")
        print("-" * 60)
        
        # 创建PCA结果DataFrame
        pca_columns = [f'PC{i+1}' for i in range(self.n_pca_components)]
        pca_df = pd.DataFrame(X_pca, columns=pca_columns)
        
        # 合并PCA结果与目标变量和切削次数
        pca_data = pd.concat([pca_df, y, cut_num], axis=1)
        
        # 可视化PCA结果
        self.visualize_pca(X_pca, y.values, pca.explained_variance_ratio_, cutter)
        
        # 保存PCA特征加载矩阵
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=pca_columns,
            index=X.columns
        )
        
        cutter_output_dir = os.path.join(self.output_path, cutter)
        loadings_file = os.path.join(cutter_output_dir, f"{cutter}_pca_loadings.csv")
        loadings.to_csv(loadings_file)
        self.logger.info(f"PCA特征加载矩阵已保存至: {loadings_file}")
        
        return pca_data, pca, scaler
    
    def visualize_pca(self, X_pca, y, explained_variance, cutter):
        """
        可视化PCA结果
        
        参数:
            X_pca: PCA降维后的数据
            y: 目标变量
            explained_variance: 各主成分解释的方差比例
            cutter: 刀具名称
        """
        # 创建输出目录
        cutter_output_dir = os.path.join(self.output_path, cutter)
        if not os.path.exists(cutter_output_dir):
            os.makedirs(cutter_output_dir)
        
        # 可视化前两个主成分
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.8)
        plt.colorbar(scatter, label=self.target_column)
        plt.xlabel(f'PC1 ({explained_variance[0]:.2%})')
        plt.ylabel(f'PC2 ({explained_variance[1]:.2%})')
        plt.title(f'刀具 {cutter} 的PCA结果 - 前两个主成分')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 保存图像
        pca_plot_file = os.path.join(cutter_output_dir, f"{cutter}_pca_visualization.png")
        plt.savefig(pca_plot_file, dpi=300)
        plt.close()
        self.logger.info(f"PCA可视化图已保存至: {pca_plot_file}")
        
        # 可视化主成分解释的方差比例
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7)
        plt.step(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), where='mid', color='red')
        plt.ylabel('解释方差比例')
        plt.xlabel('主成分')
        plt.title(f'刀具 {cutter} 的PCA解释方差')
        plt.xticks(range(1, len(explained_variance) + 1))
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 添加累积方差比例的文本
        for i, cum_var in enumerate(np.cumsum(explained_variance)):
            plt.text(i + 1, cum_var + 0.02, f'{cum_var:.2%}', ha='center')
        
        # 保存图像
        variance_plot_file = os.path.join(cutter_output_dir, f"{cutter}_pca_variance.png")
        plt.savefig(variance_plot_file, dpi=300)
        plt.close()
        self.logger.info(f"PCA方差解释图已保存至: {variance_plot_file}")
    
    def select_common_features(self, cutters, use_normalized=False):
        """
        为所有刀具选择相同的特征集合
        
        参数:
            cutters: 刀具列表
            use_normalized: 是否使用归一化特征数据
            
        返回:
            selected_features: 选择的共同特征列表
        """
        self.logger.info(f"为所有刀具选择共同特征集...")
        
        # 1. 收集所有刀具的特征相关系数
        all_corr_data = {}
        all_features = set()
        
        for cutter in cutters:
            _, selected_features, corr = self.select_features(cutter, use_normalized)
            if selected_features is None:
                continue
                
            # 过滤掉所有wear_VB开头的列
            corr = [(col, abs_pearson, pearson, abs_spearman, spearman) 
                   for col, abs_pearson, pearson, abs_spearman, spearman in corr 
                   if not col.startswith('wear_VB')]
            
            # 使用max(abs_pearson, abs_spearman)作为排序标准
            all_corr_data[cutter] = {col: max(abs_pearson, abs_spearman) for col, abs_pearson, _, abs_spearman, _ in corr}
            all_features.update(col for col, _, _, _, _ in corr)
        
        # 2. 计算每个特征在所有刀具上的平均相关系数
        avg_corr = {}
        for feature in all_features:
            total_corr = sum(all_corr_data[cutter].get(feature, 0) for cutter in cutters)
            avg_corr[feature] = total_corr / len(cutters)
        
        # 3. 按平均相关系数排序
        sorted_features = sorted(avg_corr.items(), key=lambda x: x[1], reverse=True)
        
        # 4. 选择满足最小相关性要求的前N个特征
        initial_common_features = []
        for feature, corr_val in sorted_features:
            if corr_val >= self.min_correlation:
                initial_common_features.append(feature)
                if len(initial_common_features) >= self.n_features:
                    break
        
        if len(initial_common_features) < self.n_features:
            self.logger.warning(f"只有 {len(initial_common_features)} 个特征的平均相关系数达到 {self.min_correlation}，将使用前 {self.n_features} 个特征")
            initial_common_features = [feature for feature, _ in sorted_features[:self.n_features]]
        
        # 5. 加载所有刀具的特征数据，用于检查特征间的相关性
        all_cutter_data = {}
        for cutter in cutters:
            features_file = os.path.join(self.features_path, cutter, f"{cutter}_{'normalized_features' if use_normalized else 'features'}.csv")
            try:
                data = pd.read_csv(features_file)
                # 提取初步选择的特征
                selected_columns = [col for col in initial_common_features if col in data.columns]
                all_cutter_data[cutter] = data[selected_columns]
            except Exception as e:
                self.logger.error(f"加载刀具 {cutter} 的特征数据失败: {e}")
        
        # 6. 合并所有刀具的数据，用于检查特征间的相关性
        if all_cutter_data:
            combined_data = pd.concat(all_cutter_data.values(), ignore_index=True)
            
            # 检查并移除高度相关的特征
            removed_features = self.remove_highly_correlated_features(combined_data, initial_common_features)
            
            # 更新共同特征列表
            common_features = [f for f in initial_common_features if f not in removed_features]
        else:
            common_features = initial_common_features
        
        # 7. 打印共同特征
        print("\n所有刀具共同的特征选择结果:")
        print("-" * 80)
        print(f"{'特征名称':<50} {'平均相关系数':<15}")
        print("-" * 80)
        for i, feature in enumerate(common_features):
            print(f"{i+1}. {feature:<48} {avg_corr[feature]:>+.4f}")
        print("-" * 80)
        
        # 8. 为每个刀具应用共同特征并执行PCA
        all_pca_data = {}
        pca_components = None
        pca_scaler = None
        
        for cutter in cutters:
            self.logger.info(f"为刀具 {cutter} 应用共同特征集并执行PCA...")
            
            # 应用共同特征
            selected_data = self.apply_common_features(cutter, common_features, use_normalized)
            
            if selected_data is not None:
                # 执行PCA
                pca_data, pca, scaler = self.apply_pca(selected_data, cutter)
                
                # 保存PCA结果
                cutter_output_dir = os.path.join(self.output_path, cutter)
                pca_output_file = os.path.join(cutter_output_dir, f"{cutter}_pca_data.csv")
                pca_data.to_csv(pca_output_file, index=False)
                self.logger.info(f"PCA降维后的数据已保存至: {pca_output_file}")
                
                # 存储第一个刀具的PCA组件和标准化器，用于后续处理
                if pca_components is None:
                    pca_components = pca.components_
                    pca_scaler = scaler
                
                all_pca_data[cutter] = pca_data
        
        # 9. 保存共同特征和PCA模型
        pca_model_info = {
            'common_features': common_features,
            'pca_n_components': self.n_pca_components,
            'min_correlation': self.min_correlation,
            'max_feature_correlation': self.max_feature_correlation
        }
        
        with open(os.path.join(self.output_path, "common_features_pca_info.json"), 'w') as f:
            json.dump(pca_model_info, f, indent=4)
        
        self.logger.info(f"共同特征和PCA模型信息已保存至: {os.path.join(self.output_path, 'common_features_pca_info.json')}")
        
        return common_features
    
    def apply_common_features(self, cutter, common_features, use_normalized=False):
        """
        为单个刀具应用共同特征集
        
        参数:
            cutter: 刀具名称
            common_features: 共同特征列表
            use_normalized: 是否使用归一化特征
            
        返回:
            selected_data: 应用共同特征后的数据
        """
        self.logger.info(f"为刀具 {cutter} 应用共同特征集...")
        
        # 确定特征文件路径
        features_file = os.path.join(self.features_path, cutter, f"{cutter}_{'normalized_features' if use_normalized else 'features'}.csv")
        try:
            # 加载特征数据
            features_data = pd.read_csv(features_file)
            self.logger.info(f"成功加载特征数据: {features_file}")
        except Exception as e:
            self.logger.error(f"加载特征数据失败: {e}")
            return None
        
        # 确保目标列存在
        if self.target_column not in features_data.columns:
            self.logger.error(f"目标列 {self.target_column} 不存在于特征数据中")
            return None
        
        # 确保所有共同特征都存在
        missing_features = [f for f in common_features if f not in features_data.columns]
        if missing_features:
            self.logger.error(f"刀具 {cutter} 缺少以下特征: {missing_features}")
            return None
        
        # 应用共同特征
        selected_data = features_data[common_features + [self.target_column, 'cut_num']]
        
        # 保存选择的特征数据
        cutter_output_dir = os.path.join(self.output_path, cutter)
        if not os.path.exists(cutter_output_dir):
            os.makedirs(cutter_output_dir)
        
        output_file = os.path.join(cutter_output_dir, f"{cutter}_selected_feature_data.csv")
        selected_data.to_csv(output_file, index=False)
        self.logger.info(f"应用共同特征后的数据已保存至: {output_file}")
        
        return selected_data

def main():
    """主函数"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='铣刀寿命预测系统 - 特征选择模块')
    
    # 添加命令行参数
    parser.add_argument('--features_path', type=str, default='/Users/xiaohudemac/cursor01/bishe/5_8tool/data/features',
                        help='特征数据路径')
    parser.add_argument('--output_path', type=str, default='/Users/xiaohudemac/cursor01/bishe/5_8tool/data/selected_features',
                        help='输出路径')
    parser.add_argument('--target_column', type=str, default='wear_VB_avg',
                        help='目标变量列名')
    parser.add_argument('--n_features', type=int, default=40,
                        help='通过相关系数初步选择的特征数量')
    parser.add_argument('--n_pca_components', type=int, default=5,
                        help='PCA降维后的组件数量')
    parser.add_argument('--min_correlation', type=float, default=0.85,
                        help='特征需要达到的最小相关系数（皮尔逊或斯皮尔曼之一）')
    parser.add_argument('--max_feature_correlation', type=float, default=0.98,
                        help='特征之间的最大允许相关系数，用于移除冗余特征')
    parser.add_argument('--cutters', type=str, default='c1,c4,c6',
                        help='需要处理的刀具列表，用逗号分隔')
    parser.add_argument('--use_normalized', action='store_true',
                        help='使用归一化特征数据')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 解析刀具列表
    cutters = args.cutters.split(',')
    
    # 初始化特征选择器
    selector = FeatureSelector(
        features_path=args.features_path,
        output_path=args.output_path,
        target_column=args.target_column,
        n_features=args.n_features,
        n_pca_components=args.n_pca_components,
        min_correlation=args.min_correlation,
        max_feature_correlation=args.max_feature_correlation
    )
    
    # 为所有刀具选择相同的特征集
    common_features = selector.select_common_features(cutters, args.use_normalized)
    print(f"\n为所有刀具选择的共同特征: {', '.join(common_features)}")
    
    print("\n特征选择和PCA降维完成！接下来将使用这些特征训练模型...")

if __name__ == "__main__":
    main() 