import pandas as pd
import json
import os
import glob

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
    
    model_name = f"{model_type}+{param_type}"
    
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
        feature_df = feature_df.sort_values('importance', ascending=False)
        top_features = feature_df.head(3)['feature'].tolist()
    
    results.append({
        '模型': model['name'],
        '模型类型': model['model_type'],
        '参数类型': model['param_type'],
        'R²': eval_data['r2'],
        'RMSE': eval_data['rmse'],
        'MAE': eval_data['mae'],
        '前三重要特征': ', '.join(top_features) if top_features else 'N/A'
    })

# 创建DataFrame并按R²排序
df = pd.DataFrame(results)
df = df.sort_values('R²', ascending=False)

print('\n铣刀寿命预测模型性能比较：')
print('='*120)
print(df[['模型', 'R²', 'RMSE', 'MAE']].to_string(index=False))
print('='*120)

# 按模型类型分组分析
print('\n按特征工程方法分组：')
print('='*120)
grouped = df.groupby('模型类型').mean(numeric_only=True)
print(grouped[['R²', 'RMSE', 'MAE']])
print('='*120)

# 按参数类型分组分析
print('\n按参数优化方法分组：')
print('='*120)
grouped = df.groupby('参数类型').mean(numeric_only=True)
print(grouped[['R²', 'RMSE', 'MAE']])
print('='*120) 