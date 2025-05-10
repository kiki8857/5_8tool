import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import glob
import numpy as np
import seaborn as sns

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

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
    
    results.append({
        '模型': model['name'],
        '模型类型': model['model_type'],
        '参数类型': model['param_type'],
        'R²': eval_data['r2'],
        'RMSE': eval_data['rmse'],
        'MAE': eval_data['mae']
    })

# 创建DataFrame
df = pd.DataFrame(results)

# 创建图表
plt.figure(figsize=(12, 8))

# 设置配色方案
colors = {'特征选择+优化超参数': 'darkred', 
          '特征选择+默认参数': 'lightcoral',
          'PCA降维+优化超参数': 'darkblue', 
          'PCA降维+默认参数': 'lightblue'}

# 散点图，比较R²和RMSE
plt.subplot(2, 2, 1)
for model_name in df['模型'].unique():
    model_data = df[df['模型'] == model_name]
    plt.scatter(model_data['R²'], model_data['RMSE'], 
                label=model_name, color=colors.get(model_name, 'gray'),
                alpha=0.7, s=100)
plt.xlabel('决定系数 (R²)')
plt.ylabel('均方根误差 (RMSE)')
plt.title('决定系数 vs 均方根误差')
plt.grid(True, alpha=0.3)
plt.legend()

# 箱线图，比较不同特征工程方法的R²分布
plt.subplot(2, 2, 2)
sns.boxplot(x='模型类型', y='R²', data=df, palette='Set2')
plt.title('不同特征工程方法的决定系数分布')
plt.grid(True, alpha=0.3)

# 箱线图，比较不同参数优化方法的R²分布
plt.subplot(2, 2, 3)
sns.boxplot(x='参数类型', y='R²', data=df, palette='Set3')
plt.title('优化参数与默认参数的决定系数分布')
plt.grid(True, alpha=0.3)

# 创建分组条形图，比较各种组合的R²均值
plt.subplot(2, 2, 4)
grouped_df = df.groupby(['模型类型', '参数类型']).mean(numeric_only=True).reset_index()
bar_width = 0.35
x = np.arange(2)  # 两种特征工程方法
default_bars = plt.bar(x - bar_width/2, 
                     grouped_df[grouped_df['参数类型'] == '默认参数']['R²'], 
                     width=bar_width, label='默认参数')
optimized_bars = plt.bar(x + bar_width/2, 
                        grouped_df[grouped_df['参数类型'] == '优化超参数']['R²'], 
                        width=bar_width, label='优化超参数')
plt.xlabel('特征工程方法')
plt.ylabel('平均决定系数 (R²)')
plt.title('特征工程和参数优化组合效果')
plt.xticks(x, grouped_df['模型类型'].unique())
plt.grid(True, alpha=0.3)
plt.legend()

# 添加数值标签
for bars in [default_bars, optimized_bars]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('../results/model_comparison.png', dpi=300)

print('分析图表已保存至: ../results/model_comparison.png')

# 打印最佳模型信息
best_model = df.loc[df['R²'].idxmax()]
print(f'\n最佳模型信息:')
print(f'模型: {best_model["模型"]}')
print(f'决定系数 (R²): {best_model["R²"]:.4f}')
print(f'均方根误差 (RMSE): {best_model["RMSE"]:.4f}')
print(f'平均绝对误差 (MAE): {best_model["MAE"]:.4f}')

# 打印特征工程方法和参数优化的平均效果
print('\n特征工程方法的平均效果:')
print(df.groupby('模型类型')['R²'].mean())

print('\n参数优化的平均效果:')
print(df.groupby('参数类型')['R²'].mean())

print('\n参数优化的提升效果:')
improvement = df.groupby('参数类型')['R²'].mean()['优化超参数'] - df.groupby('参数类型')['R²'].mean()['默认参数']
print(f'R²提升: {improvement:.4f} ({improvement/df.groupby("参数类型")["R²"].mean()["默认参数"]*100:.2f}%)') 