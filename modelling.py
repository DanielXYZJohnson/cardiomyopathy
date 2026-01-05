import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import make_scorer, accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
from tqdm import tqdm
import time
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

def calculate_confidence_interval(scores, confidence=0.95):
    """计算95%置信区间"""
    n = len(scores)
    mean = np.mean(scores)
    std = np.std(scores, ddof=1)
    margin = 1.96 * std / np.sqrt(n)  # 95% CI
    return mean, mean - margin, mean + margin

def specificity_score(y_true, y_pred, is_multiclass=False):
    """计算特异性 (Specificity)，支持多分类"""
    if is_multiclass:
        # 计算宏平均特异性
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
    """计算阴性预测值 (NPV)，支持多分类"""
    if is_multiclass:
        # 计算宏平均NPV
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
    """计算阳性预测值 (PPV/Precision)，支持多分类"""
    if is_multiclass:
        # 计算宏平均PPV
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

def load_and_preprocess_data(file_path):
    """读取Excel文件并进行预处理"""
    print("=== 读取Excel文件 ===")
    df = pd.read_excel(file_path)
    print(f"原始数据形状: {df.shape}")
    
    # 处理缺失值
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("存在缺失值，进行填充...")
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
    
    # 分离特征和目标
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    print(f"特征数量: {X.shape[1]}")
    print(f"目标变量唯一值: {y.unique()}")
    
    # 处理分类特征
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        print(f"发现分类特征: {categorical_cols.tolist()}")
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # 处理目标变量
    is_classification = True
    if y.dtype == 'object' or y.dtype.name == 'category':
        le = LabelEncoder()
        y = le.fit_transform(y)
        print(f"编码后的目标变量: {np.unique(y)}")
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )
    
    print(f"训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, X.columns.tolist(), len(np.unique(y)), scaler, file_path

def train_and_evaluate_models(X_train, X_test, y_train, y_test, n_classes, scaler, original_data_path, output_dir):
    """训练、优化和评估所有模型"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/models", exist_ok=True)
    os.makedirs(f"{output_dir}/results", exist_ok=True)
    
    is_multiclass = n_classes > 2
    
    # 定义评分器
    scoring = {
        'accuracy': 'accuracy',
        'recall': 'recall_macro' if is_multiclass else 'recall',
        'f1': 'f1_macro' if is_multiclass else 'f1',
        'roc_auc': 'roc_auc_ovr' if is_multiclass else 'roc_auc'
    }
    
    # 自定义评分器
    custom_scorers = {
        'specificity': make_scorer(lambda y_true, y_pred: specificity_score(y_true, y_pred, is_multiclass)),
        'ppv': make_scorer(lambda y_true, y_pred: ppv_score(y_true, y_pred, is_multiclass)),
        'npv': make_scorer(lambda y_true, y_pred: npv_score(y_true, y_pred, is_multiclass))
    }
    
    # 定义所有模型及其参数网格（优化参数范围）
    models_config = {
        'KNN': {
            'model': KNeighborsClassifier(),
            'param_grid': {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        },
        'GNB': {
            'model': GaussianNB(),
            'param_grid': {
                'var_smoothing': [1e-9, 1e-8, 1e-7]
            }
        },
        'LR': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'param_grid': {
                'C': [0.01, 0.1, 1, 10],
                'solver': ['lbfgs', 'liblinear'] if not is_multiclass else ['lbfgs'],
                'penalty': ['l2']
            }
        },
        'SVM': {
            'model': SVC(random_state=42, probability=True, cache_size=2000),  # 增加缓存
            'param_grid': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 0.1]
            }
        },
        'DT': {
            'model': DecisionTreeClassifier(random_state=42),
            'param_grid': {
                'max_depth': [None, 5, 10],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'criterion': ['gini', 'entropy']
            }
        },
        'RF': {
            'model': RandomForestClassifier(random_state=42, n_jobs=-1),
            'param_grid': {
                'n_estimators': [50, 100],
                'max_depth': [None, 10],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        },
        'ET': {
            'model': ExtraTreesClassifier(random_state=42, n_jobs=-1),
            'param_grid': {
                'n_estimators': [50, 100],
                'max_depth': [None, 10],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        },
        'HGB': {
            'model': HistGradientBoostingClassifier(random_state=42),
            'param_grid': {
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5],
                'min_samples_leaf': [10, 20]
            }
        },
        'XGBoost': {
            'model': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
            'param_grid': {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5],
                'subsample': [0.8, 1.0]
            }
        },
        'LightGBM': {
            'model': lgb.LGBMClassifier(random_state=42),
            'param_grid': {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5],
                'num_leaves': [31, 63]
            }
        },
        'MLP': {
            'model': MLPClassifier(random_state=42, max_iter=500, batch_size=128),  # 限制迭代次数和批量大小
            'param_grid': {
                'hidden_layer_sizes': [(50,), (100,)],
                'activation': ['relu'],
                'alpha': [0.0001, 0.001],
                'learning_rate': ['constant']
            }
        }
    }
    
    results = {}
    all_model_paths = {}
    
    print("=== 开始训练和优化所有模型 ===")
    
    for model_name, config in tqdm(models_config.items(), desc="模型训练进度"):
        print(f"\n{'='*70}")
        print(f"🎯 训练模型: {model_name}")
        print(f"{'='*70}")
        
        try:
            # 10折交叉验证
            cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            
            # 网格搜索
            grid_search = GridSearchCV(
                estimator=config['model'],
                param_grid=config['param_grid'],
                cv=cv,
                scoring='accuracy',
                n_jobs=-1,
                verbose=0  # 减少输出
            )
            
            start_time = time.time()
            grid_search.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            
            print(f"✅ {model_name} 训练完成!")
            print(f"最佳参数: {best_params}")
            print(f"最佳交叉验证准确率: {best_score:.4f}")
            print(f"训练耗时: {training_time:.2f}秒")
            
            # 详细的10折交叉验证评估
            cv_results = cross_validate(
                best_model, X_train, y_train, cv=cv,
                scoring=scoring,
                n_jobs=-1,
                return_train_score=False
            )
            
            # 计算自定义指标
            custom_results = {name: [] for name in custom_scorers.keys()}
            for train_idx, test_idx in cv.split(X_train, y_train):
                X_train_fold, X_test_fold = X_train.iloc[train_idx], X_train.iloc[test_idx]
                y_train_fold, y_test_fold = y_train[train_idx], y_train[test_idx]
                
                fold_model = best_model.fit(X_train_fold, y_train_fold)
                y_pred = fold_model.predict(X_test_fold)
                
                for name, scorer in custom_scorers.items():
                    custom_results[name].append(scorer(y_test_fold, y_pred))
            
            # 计算各指标的95%置信区间
            metrics_with_ci = {}
            for metric_name in ['test_accuracy', 'test_recall', 'test_f1']:
                if metric_name in cv_results:
                    scores = cv_results[metric_name]
                    mean, lower, upper = calculate_confidence_interval(scores)
                    metrics_with_ci[metric_name] = {
                        'mean': mean,
                        'lower_ci': lower,
                        'upper_ci': upper,
                        'scores': scores.tolist()
                    }
            
            # AUC需要特殊处理
            if 'test_roc_auc' in cv_results:
                auc_scores = cv_results['test_roc_auc']
                mean_auc, lower_auc, upper_auc = calculate_confidence_interval(auc_scores)
                metrics_with_ci['test_roc_auc'] = {
                    'mean': mean_auc,
                    'lower_ci': lower_auc,
                    'upper_ci': upper_auc,
                    'scores': auc_scores.tolist()
                }
            
            # 计算自定义指标的CI
            for name, scores in custom_results.items():
                if scores:  # 确保有分数
                    mean, lower, upper = calculate_confidence_interval(scores)
                    metrics_with_ci[name] = {
                        'mean': mean,
                        'lower_ci': lower,
                        'upper_ci': upper,
                        'scores': scores
                    }
            
            # 测试集评估
            y_pred = best_model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)
            test_recall = recall_score(y_test, y_pred, average='macro' if is_multiclass else 'binary')
            test_f1 = f1_score(y_test, y_pred, average='macro' if is_multiclass else 'binary')
            
            # 计算AUC
            test_auc = None
            if hasattr(best_model, 'predict_proba'):
                try:
                    y_proba = best_model.predict_proba(X_test)
                    if is_multiclass:
                        test_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
                    else:
                        test_auc = roc_auc_score(y_test, y_proba[:, 1])
                except Exception as e:
                    print(f"⚠️ AUC计算失败: {str(e)}")
                    test_auc = None
            
            # 计算自定义指标
            test_specificity = specificity_score(y_test, y_pred, is_multiclass)
            test_ppv = ppv_score(y_test, y_pred, is_multiclass)
            test_npv = npv_score(y_test, y_pred, is_multiclass)
            
            print(f"\n📊 {model_name} 测试集性能:")
            print(f"准确率: {test_accuracy:.4f}")
            print(f"召回率: {test_recall:.4f}")
            print(f"F1分数: {test_f1:.4f}")
            if test_auc is not None:
                print(f"AUC: {test_auc:.4f}")
            print(f"特异性: {test_specificity:.4f}")
            print(f"阳性预测值: {test_ppv:.4f}")
            print(f"阴性预测值: {test_npv:.4f}")
            
            # 保存模型
            model_path = f"{output_dir}/models/{model_name.replace(' ', '_')}_best.pkl"
            joblib.dump(best_model, model_path)
            print(f"💾 模型已保存: {model_path}")
            
            # 保存参数
            params_path = f"{output_dir}/models/{model_name.replace(' ', '_')}_params.json"
            with open(params_path, 'w') as f:
                json.dump(best_params, f, indent=4)
            print(f"📝 参数已保存: {params_path}")
            
            # 保存结果
            model_result = {
                'model_name': model_name,
                'best_params': best_params,
                'best_cv_score': best_score,
                'training_time': training_time,
                'cv_metrics': metrics_with_ci,
                'test_metrics': {
                    'accuracy': test_accuracy,
                    'recall': test_recall,
                    'f1': test_f1,
                    'auc': test_auc,
                    'specificity': test_specificity,
                    'ppv': test_ppv,
                    'npv': test_npv
                },
                'cv_accuracy': best_score,  # 用于后续过滤
                'model_path': model_path,
                'params_path': params_path
            }
            
            results[model_name] = model_result
            all_model_paths[model_name] = model_path
            
            # 保存单个模型结果
            result_path = f"{output_dir}/results/{model_name}_results.json"
            with open(result_path, 'w') as f:
                json.dump(model_result, f, indent=4, default=str)
            print(f"📊 结果已保存: {result_path}")
            
        except Exception as e:
            print(f"❌ {model_name} 训练失败: {str(e)}")
            results[model_name] = {'error': str(e)}
    
    # 保存scaler
    scaler_path = f"{output_dir}/scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler已保存: {scaler_path}")
    
    # 创建综合结果报告
    create_comprehensive_report(results, output_dir, is_multiclass)
    
    # 保存所有模型路径
    paths_info = {
        'model_paths': all_model_paths,
        'output_dir': output_dir,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'n_classes': n_classes,
        'is_multiclass': is_multiclass,
        'original_data_path': original_data_path,
        'scaler_path': scaler_path
    }
    
    paths_path = f"{output_dir}/all_model_paths.json"
    with open(paths_path, 'w') as f:
        json.dump(paths_info, f, indent=4)
    print(f"\n✅ 所有模型路径信息已保存: {paths_path}")
    
    return results, paths_info

def create_comprehensive_report(results, output_dir, is_multiclass):
    """创建综合报告"""
    # 准备数据
    report_data = []
    for model_name, result in results.items():
        if 'error' in result:
            report_data.append({
                'Model': model_name,
                'Status': 'Failed',
                'Error': result['error']
            })
            continue
        
        cv_metrics = result['cv_metrics']
        test_metrics = result['test_metrics']
        
        # 提取95% CI
        accuracy_ci = cv_metrics.get('test_accuracy', {})
        recall_ci = cv_metrics.get('test_recall', {})
        f1_ci = cv_metrics.get('test_f1', {})
        auc_ci = cv_metrics.get('test_roc_auc', {})
        
        row = {
            'Model': model_name,
            'Status': 'Success',
            'CV_Accuracy_Mean': accuracy_ci.get('mean', 0),
            'CV_Accuracy_95%CI': f"{accuracy_ci.get('lower_ci', 0):.4f} - {accuracy_ci.get('upper_ci', 0):.4f}",
            'CV_Recall_Mean': recall_ci.get('mean', 0),
            'CV_Recall_95%CI': f"{recall_ci.get('lower_ci', 0):.4f} - {recall_ci.get('upper_ci', 0):.4f}",
            'CV_F1_Mean': f1_ci.get('mean', 0),
            'CV_F1_95%CI': f"{f1_ci.get('lower_ci', 0):.4f} - {f1_ci.get('upper_ci', 0):.4f}",
            'Test_Accuracy': test_metrics['accuracy'],
            'Test_Recall': test_metrics['recall'],
            'Test_F1': test_metrics['f1'],
            'Test_Specificity': test_metrics['specificity'],
            'Test_PPV': test_metrics['ppv'],
            'Test_NPV': test_metrics['npv'],
            'Training_Time(s)': result['training_time']
        }
        
        if 'test_roc_auc' in cv_metrics:
            row['CV_AUC_Mean'] = auc_ci.get('mean', 0)
            row['CV_AUC_95%CI'] = f"{auc_ci.get('lower_ci', 0):.4f} - {auc_ci.get('upper_ci', 0):.4f}"
            row['Test_AUC'] = test_metrics['auc']
        
        report_data.append(row)
    
    # 创建DataFrame
    report_df = pd.DataFrame(report_data)
    
    # 保存CSV报告
    csv_path = f"{output_dir}/results/comprehensive_report.csv"
    report_df.to_csv(csv_path, index=False)
    print(f"✅ 综合报告CSV已保存: {csv_path}")
    
    # 生成可视化
    if 'Test_Accuracy' in report_df.columns:
        plt.figure(figsize=(15, 8))
        successful_models = report_df[report_df['Status'] == 'Success']
        if not successful_models.empty:
            sns.barplot(x='Model', y='Test_Accuracy', data=successful_models)
            plt.title('各模型测试集准确率对比', fontsize=16)
            plt.xlabel('模型', fontsize=12)
            plt.ylabel('准确率', fontsize=12)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/results/model_accuracy_comparison.png", dpi=300)
            plt.close()
            print(f"📈 准确率对比图已保存: {output_dir}/results/model_accuracy_comparison.png")

def main():
    """主函数"""
    # 配置
    excel_file_path = "data.xlsx"
    output_dir = f"model_training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("🚀 开始模型训练流程")
    print(f"Excel文件: {excel_file_path}")
    print(f"输出目录: {output_dir}")
    
    # 1. 加载和预处理数据
    X_train, X_test, y_train, y_test, feature_names, n_classes, scaler, original_data_path = load_and_preprocess_data(excel_file_path)
    
    # 2. 保存预处理信息
    preprocessing_info = {
        'feature_names': feature_names,
        'n_classes': n_classes,
        'is_multiclass': n_classes > 2,
        'train_shape': X_train.shape,
        'test_shape': X_test.shape,
        'class_distribution': pd.Series(y_train).value_counts().to_dict(),
        'original_data_path': original_data_path
    }
    
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/preprocessing_info.json", 'w') as f:
        json.dump(preprocessing_info, f, indent=4)
    print(f"✅ 预处理信息已保存: {output_dir}/preprocessing_info.json")
    
    # 3. 训练和评估所有模型
    results, paths_info = train_and_evaluate_models(X_train, X_test, y_train, y_test, n_classes, scaler, original_data_path, output_dir)
    
    print("\n🎉 模型训练流程完成!")
    print(f"所有结果保存在: {output_dir}")
    print(f"最优模型路径信息: {output_dir}/all_model_paths.json")

if __name__ == "__main__":
    main()