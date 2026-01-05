import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix, make_scorer
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
import json
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

warnings.filterwarnings('ignore')

def specificity_score(y_true, y_pred, is_multiclass=False):
    """计算特异性，支持多分类"""
    if is_multiclass:
        cm = confusion_matrix(y_true, y_pred)
        specificities = []
        for i in range(len(cm)):
            tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
            fp = np.sum(cm[:, i]) - cm[i, i]
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificities.append(spec)
        return np.mean(specificities)
    else:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0

def npv_score(y_true, y_pred, is_multiclass=False):
    """计算阴性预测值，支持多分类"""
    if is_multiclass:
        cm = confusion_matrix(y_true, y_pred)
        npvs = []
        for i in range(len(cm)):
            tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
            fn = np.sum(cm[i, :]) - cm[i, i]
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            npvs.append(npv)
        return np.mean(npvs)
    else:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fn) if (tn + fn) > 0 else 0

def ppv_score(y_true, y_pred, is_multiclass=False):
    """计算阳性预测值，支持多分类"""
    if is_multiclass:
        cm = confusion_matrix(y_true, y_pred)
        ppvs = []
        for i in range(len(cm)):
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            ppvs.append(ppv)
        return np.mean(ppvs)
    else:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tp / (tp + fp) if (tp + fp) > 0 else 0

def load_saved_models(model_paths_file):
    """加载所有保存的最优模型"""
    print("=== 加载保存的最优模型 ===")
    
    if not os.path.exists(model_paths_file):
        raise FileNotFoundError(f"模型路径文件不存在: {model_paths_file}")
    
    with open(model_paths_file, 'r') as f:
        paths_info = json.load(f)
    
    output_dir = paths_info['output_dir']
    model_paths = paths_info['model_paths']
    n_classes = paths_info['n_classes']
    is_multiclass = paths_info['is_multiclass']
    scaler_path = paths_info.get('scaler_path')
    
    print(f"输出目录: {output_dir}")
    print(f"模型数量: {len(model_paths)}")
    print(f"是否为多分类: {is_multiclass}")
    
    loaded_models = {}
    
    for model_name, model_path in model_paths.items():
        print(f"\n📂 加载模型: {model_name}")
        print(f"路径: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在，跳过: {model_path}")
            continue
        
        try:
            model = joblib.load(model_path)
            loaded_models[model_name] = model
            print(f"✅ 成功加载: {model_name} - {type(model).__name__}")
        except Exception as e:
            print(f"❌ 加载失败: {model_name} - {str(e)}")
    
    print(f"\n✅ 成功加载 {len(loaded_models)} 个模型")
    return loaded_models, output_dir, n_classes, is_multiclass, scaler_path, paths_info

def load_data_for_super_learner(paths_info):
    """加载原始数据用于Super Learner训练，避免数据泄露"""
    print("\n=== 加载原始数据 ===")
    
    original_data_path = paths_info['original_data_path']
    scaler_path = paths_info['scaler_path']
    is_multiclass = paths_info['is_multiclass']
    
    print(f"原始数据路径: {original_data_path}")
    print(f"Scaler路径: {scaler_path}")
    
    if not os.path.exists(original_data_path):
        raise FileNotFoundError(f"原始数据文件不存在: {original_data_path}")
    
    # 加载数据
    df = pd.read_excel(original_data_path)
    
    # 数据预处理（与训练时相同）
    # 处理缺失值
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # 分离特征和目标
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # 处理分类特征
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # 处理目标变量
    if y.dtype == 'object' or y.dtype.name == 'category':
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    # 加载预训练的scaler（关键：避免数据泄露）
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler文件不存在: {scaler_path}")
    
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X)  # 只transform，不fit
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )
    
    print(f"训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, X.columns.tolist(), is_multiclass

def build_optimize_super_learner(loaded_models, X_train, X_test, y_train, y_test, output_dir, is_multiclass, paths_info):
    """构建和优化Super Learner"""
    print("\n" + "="*80)
    print("🚀 构建和优化 Super Learner 集成模型")
    print("="*80)
    
    # 过滤基础模型：只选择CV准确率>0.7的模型
    results_dir = f"{output_dir}/results"
    valid_models = {}
    
    for model_name, model in loaded_models.items():
        result_file = f"{results_dir}/{model_name}_results.json"
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                try:
                    result = json.load(f)
                    cv_accuracy = result.get('cv_accuracy', 0)
                    if cv_accuracy > 0.7:  # 阈值可调整
                        valid_models[model_name] = model
                        print(f"✅ 选择基础模型: {model_name} (CV准确率: {cv_accuracy:.4f})")
                    else:
                        print(f"❌ 跳过基础模型: {model_name} (CV准确率: {cv_accuracy:.4f} < 0.7)")
                except Exception as e:
                    print(f"⚠️ 读取 {model_name} 结果失败: {str(e)}")
                    valid_models[model_name] = model  # 保守起见还是包含
        else:
            valid_models[model_name] = model  # 包含没有结果文件的模型
    
    if len(valid_models) < 2:
        print(f"⚠️ 有效基础模型数量不足 ({len(valid_models)} < 2)，使用所有加载的模型")
        valid_models = loaded_models
    
    # 准备基础模型列表
    estimators = []
    for model_name, model in valid_models.items():
        clean_name = model_name.replace(' ', '_').replace('-', '_').lower()
        estimators.append((clean_name, model))
        print(f"🔧 添加基础模型: {model_name} -> {clean_name}")
    
    print(f"\n🎯 将使用 {len(estimators)} 个基础模型构建Super Learner")
    
    # 动态选择元模型
    if is_multiclass:
        meta_learner = RidgeClassifier(random_state=42)
        print("📊 多分类问题，使用 RidgeClassifier 作为元模型")
    else:
        meta_learner = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')
        print("📊 二分类问题，使用 LogisticRegression 作为元模型")
    
    # 定义Super Learner参数网格
    param_grid = {
        'stack_method': ['auto', 'predict_proba'] if not is_multiclass else ['auto'],
        'passthrough': [True, False],
        'cv': [5, 10]
    }
    
    # 为元模型添加参数
    if is_multiclass:
        param_grid['final_estimator__alpha'] = [0.1, 1.0, 10.0]
    else:
        param_grid['final_estimator__C'] = [0.1, 1.0, 10.0]
        param_grid['final_estimator__class_weight'] = ['balanced', None]
    
    # 创建Super Learner
    super_learner = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1,
        verbose=0
    )
    
    print("\n🔧 开始Super Learner参数优化...")
    
    # 10折交叉验证
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    # 网格搜索
    grid_search = GridSearchCV(
        estimator=super_learner,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    best_super_learner = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"\n✅ Super Learner 优化完成!")
    print(f"最佳参数: {best_params}")
    print(f"最佳交叉验证准确率: {best_score:.4f}")
    print(f"训练耗时: {training_time:.2f}秒")
    
    # 10折交叉验证评估
    scoring = {
        'accuracy': 'accuracy',
        'recall': 'recall_macro' if is_multiclass else 'recall',
        'f1': 'f1_macro' if is_multiclass else 'f1',
        'roc_auc': 'roc_auc_ovr' if is_multiclass else 'roc_auc'
    }
    
    cv_results = cross_validate(
        best_super_learner, X_train, y_train, cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False
    )
    
    # 计算95%置信区间
    def calculate_ci(scores):
        n = len(scores)
        mean = np.mean(scores)
        std = np.std(scores, ddof=1)
        margin = 1.96 * std / np.sqrt(n)
        return mean, mean - margin, mean + margin
    
    cv_metrics = {}
    for metric_name in ['test_accuracy', 'test_recall', 'test_f1']:
        if metric_name in cv_results:
            scores = cv_results[metric_name]
            mean, lower, upper = calculate_ci(scores)
            cv_metrics[metric_name] = {
                'mean': mean,
                'lower_ci': lower,
                'upper_ci': upper,
                'scores': scores.tolist()
            }
    
    # 测试集评估
    y_pred = best_super_learner.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred, average='macro' if is_multiclass else 'binary')
    test_f1 = f1_score(y_test, y_pred, average='macro' if is_multiclass else 'binary')
    
    # 计算其他指标
    test_specificity = specificity_score(y_test, y_pred, is_multiclass)
    test_ppv = ppv_score(y_test, y_pred, is_multiclass)
    test_npv = npv_score(y_test, y_pred, is_multiclass)
    
    # 计算AUC
    test_auc = None
    if hasattr(best_super_learner, 'predict_proba'):
        try:
            y_proba = best_super_learner.predict_proba(X_test)
            if is_multiclass:
                test_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
            else:
                test_auc = roc_auc_score(y_test, y_proba[:, 1])
        except Exception as e:
            print(f"⚠️ AUC计算失败: {str(e)}")
            test_auc = None
    
    print(f"\n📊 Super Learner 测试集性能:")
    print(f"准确率: {test_accuracy:.4f}")
    print(f"召回率: {test_recall:.4f}")
    print(f"F1分数: {test_f1:.4f}")
    if test_auc is not None:
        print(f"AUC: {test_auc:.4f}")
    print(f"特异性: {test_specificity:.4f}")
    print(f"阳性预测值: {test_ppv:.4f}")
    print(f"阴性预测值: {test_npv:.4f}")
    
    # 保存Super Learner
    super_learner_dir = f"{output_dir}/super_learner"
    os.makedirs(super_learner_dir, exist_ok=True)
    
    model_path = f"{super_learner_dir}/super_learner_best.pkl"
    joblib.dump(best_super_learner, model_path)
    print(f"💾 Super Learner 模型已保存: {model_path}")
    
    # 保存参数
    params_path = f"{super_learner_dir}/super_learner_params.json"
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"📝 Super Learner 参数已保存: {params_path}")
    
    # 分析元模型权重
    try:
        if hasattr(best_super_learner.final_estimator_, 'coef_'):
            coef = best_super_learner.final_estimator_.coef_[0] if is_multiclass else best_super_learner.final_estimator_.coef_[0]
            base_model_names = [name for name, _ in estimators]
            
            weights_df = pd.DataFrame({
                'base_model': base_model_names,
                'weight': coef
            }).sort_values('weight', ascending=False)
            
            weights_path = f"{super_learner_dir}/meta_model_weights.csv"
            weights_df.to_csv(weights_path, index=False)
            print(f"📊 元模型权重已保存: {weights_path}")
            
            # 绘制权重图
            plt.figure(figsize=(12, 8))
            sns.barplot(x='weight', y='base_model', data=weights_df)
            plt.title('Super Learner - 元模型权重分析', fontsize=16)
            plt.xlabel('权重', fontsize=12)
            plt.ylabel('基础模型', fontsize=12)
            plt.tight_layout()
            plt.savefig(f"{super_learner_dir}/meta_model_weights.png", dpi=300)
            plt.close()
            print(f"📈 元模型权重图已保存: {super_learner_dir}/meta_model_weights.png")
    except Exception as e:
        print(f"⚠️ 元模型权重分析失败: {str(e)}")
    
    # 保存结果
    super_learner_result = {
        'model_name': 'Super_Learner',
        'base_models': [name for name, _ in estimators],
        'best_params': best_params,
        'best_cv_score': best_score,
        'training_time': training_time,
        'cv_metrics': cv_metrics,
        'test_metrics': {
            'accuracy': test_accuracy,
            'recall': test_recall,
            'f1': test_f1,
            'auc': test_auc,
            'specificity': test_specificity,
            'ppv': test_ppv,
            'npv': test_npv
        },
        'model_path': model_path,
        'params_path': params_path
    }
    
    result_path = f"{super_learner_dir}/super_learner_results.json"
    with open(result_path, 'w') as f:
        json.dump(super_learner_result, f, indent=4, default=str)
    print(f"📊 Super Learner 结果已保存: {result_path}")
    
    # 与单一最佳模型对比
    compare_with_best_single_model(super_learner_result, output_dir, is_multiclass)
    
    return super_learner_result

def compare_with_best_single_model(super_learner_result, output_dir, is_multiclass):
    """与单一最佳模型进行对比"""
    print("\n" + "="*80)
    print("🎯 与单一最佳模型对比")
    print("="*80)
    
    # 加载之前训练的模型结果
    results_dir = f"{output_dir}/results"
    if not os.path.exists(results_dir):
        print("⚠️ 找不到之前的模型结果目录")
        return
    
    # 收集所有模型结果
    model_results = []
    for file in os.listdir(results_dir):
        if file.endswith('_results.json') and file != 'super_learner_results.json':
            file_path = os.path.join(results_dir, file)
            with open(file_path, 'r') as f:
                try:
                    result = json.load(f)
                    if 'test_metrics' in result and 'error' not in result:
                        model_results.append(result)
                except Exception as e:
                    print(f"⚠️ 读取 {file} 失败: {str(e)}")
                    continue
    
    if not model_results:
        print("⚠️ 没有找到有效的模型结果")
        return
    
    # 找出单一最佳模型
    best_single_model = max(model_results, key=lambda x: x['test_metrics']['accuracy'])
    
    print(f"🏆 单一最佳模型: {best_single_model['model_name']}")
    print(f"   测试准确率: {best_single_model['test_metrics']['accuracy']:.4f}")
    print(f"   模型路径: {best_single_model['model_path']}")
    
    print(f"\n🤖 Super Learner:")
    print(f"   测试准确率: {super_learner_result['test_metrics']['accuracy']:.4f}")
    print(f"   模型路径: {super_learner_result['model_path']}")
    
    # 性能对比
    sl_accuracy = super_learner_result['test_metrics']['accuracy']
    single_accuracy = best_single_model['test_metrics']['accuracy']
    
    if sl_accuracy > single_accuracy:
        improvement = (sl_accuracy - single_accuracy) / single_accuracy * 100
        print(f"\n🎉 Super Learner 比单一最佳模型表现更好!")
        print(f"   准确率提升: {improvement:.2f}%")
        print(f"   推荐使用 Super Learner 集成模型")
    else:
        improvement = (single_accuracy - sl_accuracy) / sl_accuracy * 100
        print(f"\n💡 单一最佳模型表现略好")
        print(f"   优势: {improvement:.2f}%")
        print(f"   但 Super Learner 通常更稳定，可根据需求选择")
    
    # 创建对比报告
    comparison_data = {
        'Model': ['Super_Learner', best_single_model['model_name']],
        'Accuracy': [sl_accuracy, single_accuracy],
        'Recall': [
            super_learner_result['test_metrics']['recall'],
            best_single_model['test_metrics']['recall']
        ],
        'F1_Score': [
            super_learner_result['test_metrics']['f1'],
            best_single_model['test_metrics']['f1']
        ],
        'Type': ['Ensemble', 'Single_Model']
    }
    
    if 'auc' in super_learner_result['test_metrics']:
        comparison_data['AUC'] = [
            super_learner_result['test_metrics']['auc'],
            best_single_model['test_metrics'].get('auc', 0)
        ]
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # 保存对比结果
    comparison_path = f"{output_dir}/super_learner/model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"📊 模型对比结果已保存: {comparison_path}")
    
    # 生成对比图表
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='Accuracy', data=comparison_df, hue='Type')
    plt.title('Super Learner vs 最佳单一模型 - 准确率对比', fontsize=16)
    plt.xlabel('模型', fontsize=12)
    plt.ylabel('准确率', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/super_learner/model_comparison.png", dpi=300)
    plt.close()
    print(f"📈 模型对比图已保存: {output_dir}/super_learner/model_comparison.png")

def main():
    """主函数"""
    # 配置
    model_paths_file = "all_model_paths.json"
    
    print("🚀 开始Super Learner构建流程")
    print(f"模型路径文件: {model_paths_file}")
    
    # 1. 加载保存的最优模型
    loaded_models, output_dir, n_classes, is_multiclass, scaler_path, paths_info = load_saved_models(model_paths_file)
    
    if not loaded_models:
        print("❌ 没有加载到任何模型，流程终止")
        return
    
    # 2. 加载原始数据（关键：使用预训练的scaler避免数据泄露）
    X_train, X_test, y_train, y_test, feature_names, is_multiclass = load_data_for_super_learner(paths_info)
    
    # 3. 构建和优化Super Learner
    super_learner_result = build_optimize_super_learner(
        loaded_models, X_train, X_test, y_train, y_test, output_dir, is_multiclass, paths_info
    )
    
    print("\n🎉 Super Learner构建流程完成!")
    print(f"所有结果保存在: {output_dir}/super_learner/")

if __name__ == "__main__":
    main()