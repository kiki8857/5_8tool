import pandas as pd
import numpy as np
import json
import os
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 创建最佳模型目录
best_model_dir = '../results/best_model'
if not os.path.exists(best_model_dir):
    os.makedirs(best_model_dir)

# 记录当前时间
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

# 查找所有评估结果文件
eval_files = []
for root, dirs, files in os.walk('../results/random_forest'):
    for file in files:
        if file == 'evaluation.json':
            eval_files.append(os.path.join(root, file))

models = []
for path in eval_files:
    # 提取目录信息
    run_dir = os.path.dirname(path)
    run_name = os.path.basename(run_dir)
    parent_dir = os.path.basename(os.path.dirname(run_dir))
    
    # 判断是否为PCA模型
    is_pca = 'pca' in run_name
    model_type = 'PCA降维' if is_pca else '特征选择'
    
    # 判断是否使用优化参数
    is_optimized = 'best_params' in parent_dir or 'optimized' in parent_dir
    param_type = '优化超参数' if is_optimized else '默认参数'
    
    model_name = f'{model_type}+{param_type}'
    
    models.append({
        'name': model_name,
        'path': path,
        'run_dir': run_dir,
        'model_type': model_type,
        'param_type': param_type
    })

results = []
for model in models:
    with open(model['path'], 'r') as f:
        eval_data = json.load(f)
    
    # 获取特征重要性
    feature_importance_path = os.path.join(model['run_dir'], 'feature_importance.csv')
    top_features = []
    if os.path.exists(feature_importance_path):
        feature_df = pd.read_csv(feature_importance_path, header=None, names=['feature', 'importance'])
        # 检查第一个特征名是否为"Feature"，如果是则跳过
        if feature_df.iloc[0, 0] == 'Feature':
            feature_df = feature_df.iloc[1:].copy()
            feature_df.reset_index(drop=True, inplace=True)
        feature_df = feature_df.sort_values('importance', ascending=False)
        top_features = feature_df.head(5)['feature'].tolist()
    
    results.append({
        '模型': model['name'],
        '模型类型': model['model_type'],
        '参数类型': model['param_type'],
        'R²': eval_data['r2'],
        'RMSE': eval_data['rmse'],
        'MAE': eval_data['mae'],
        'run_dir': model['run_dir'],
        '前五重要特征': top_features
    })

# 创建DataFrame
df = pd.DataFrame(results)

# 找到最佳模型
best_model = df.loc[df['R²'].idxmax()]
best_model_run_dir = best_model['run_dir']

# 复制最佳模型到最佳模型目录
best_model_files = [
    'random_forest_model.joblib',
    'evaluation.json',
    'feature_importance.csv'
]

for file in best_model_files:
    src_file = os.path.join(best_model_run_dir, file)
    if os.path.exists(src_file):
        dst_file = os.path.join(best_model_dir, file)
        shutil.copy2(src_file, dst_file)
        print(f'已复制: {src_file} -> {dst_file}')

# 保存模型比较图
plt.figure(figsize=(10, 6))

# 散点图，比较不同模型的R²和RMSE
plt.subplot(1, 2, 1)
colors = {
    '特征选择+优化超参数': 'red',
    '特征选择+默认参数': 'blue',
    'PCA降维+优化超参数': 'green',
    'PCA降维+默认参数': 'purple'
}

markers = {
    '特征选择+优化超参数': 'o',
    '特征选择+默认参数': 's',
    'PCA降维+优化超参数': '^',
    'PCA降维+默认参数': 'd'
}

for model_name in df['模型'].unique():
    model_data = df[df['模型'] == model_name]
    plt.scatter(model_data['RMSE'], model_data['R²'], 
                label=model_name, 
                color=colors.get(model_name, 'gray'),
                marker=markers.get(model_name, 'o'),
                alpha=0.7, s=100)

# 标注最佳模型点
plt.scatter(best_model['RMSE'], best_model['R²'], 
            color='gold', marker='*', s=300, 
            edgecolor='black', linewidth=1.5, 
            zorder=5, label='最佳模型')

plt.xlabel('均方根误差 (RMSE)')
plt.ylabel('决定系数 (R²)')
plt.title('模型性能比较')
plt.grid(True, alpha=0.3)
plt.legend(loc='lower right')

# 绘制特征重要性条形图（如果有）
plt.subplot(1, 2, 2)
if best_model['模型类型'] == '特征选择' and best_model['前五重要特征']:
    feature_importance_path = os.path.join(best_model_run_dir, 'feature_importance.csv')
    feature_df = pd.read_csv(feature_importance_path, header=None, names=['feature', 'importance'])
    # 检查第一个特征名是否为"Feature"，如果是则跳过
    if feature_df.iloc[0, 0] == 'Feature':
        feature_df = feature_df.iloc[1:].copy()
        feature_df.reset_index(drop=True, inplace=True)
    feature_df = feature_df.sort_values('importance', ascending=False).head(10)
    
    sns.barplot(x='importance', y='feature', data=feature_df, palette='YlOrRd')
    plt.title('最佳模型的前10个重要特征')
    plt.xlabel('特征重要性')
    plt.ylabel('特征名称')
    plt.tight_layout()
else:
    plt.text(0.5, 0.5, '最佳模型未找到特征重要性数据', 
             horizontalalignment='center', verticalalignment='center',
             fontsize=12)
    plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(best_model_dir, 'best_model_summary.png'), dpi=300)

# 生成最佳模型摘要文件
summary = {
    'model_name': best_model['模型'],
    'model_type': best_model['模型类型'],
    'param_type': best_model['参数类型'],
    'metrics': {
        'r2_score': best_model['R²'],
        'rmse': best_model['RMSE'],
        'mae': best_model['MAE']
    },
    'top_features': best_model['前五重要特征'],
    'timestamp': current_time,
    'original_path': best_model_run_dir
}

with open(os.path.join(best_model_dir, 'model_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=4)

print(f'\n最佳模型摘要已保存至: {os.path.join(best_model_dir, "model_summary.json")}')
print(f'最佳模型可视化已保存至: {os.path.join(best_model_dir, "best_model_summary.png")}')

print('\n最佳模型详细信息:')
print(f'模型类型: {best_model["模型"]}')
print(f'决定系数 (R²): {best_model["R²"]:.4f}')
print(f'均方根误差 (RMSE): {best_model["RMSE"]:.4f}')
print(f'平均绝对误差 (MAE): {best_model["MAE"]:.4f}')

if best_model['前五重要特征']:
    print('\n最重要的五个特征:')
    for i, feature in enumerate(best_model['前五重要特征'], 1):
        print(f'{i}. {feature}') 